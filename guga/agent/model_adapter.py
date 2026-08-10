from __future__ import annotations

import json
from typing import Callable
from uuid import uuid4

from guga.agent.state import AgentTaskState, PlanStep, ToolAction, Verification
from guga.tools import ToolRegistry
from guga.types import GenerationConfig


class AgentProtocolError(RuntimeError):
    pass


class AgentModelAdapter:
    def __init__(
        self,
        model,
        generation: GenerationConfig,
        system_prompt: str,
        tools: ToolRegistry,
    ) -> None:
        self.model = model
        self.generation = generation
        self.system_prompt = system_prompt
        self.tools = tools

    def create_plan(self, state: AgentTaskState) -> list[PlanStep]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是智能体任务规划器。只输出一个 JSON 对象，不执行工具。"
                    "格式为 {\"steps\":[{\"id\":...,\"description\":...,"
                    "\"expected_result\":...,\"verification_method\":...,"
                    "\"allowed_tools\":[...]}]}。每一步必须可执行、可验证。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": state.get("user_request", ""),
                        "frozen_context": state.get("task_context", ""),
                        "previous_plan": state.get("plan", []),
                        "evidence": state.get("evidence", []),
                        "available_tools": self.tools.openai_tools(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = self._generate_json(messages, self._validate_plan_payload)
        return payload["steps"]

    def choose_action(self, state: AgentTaskState) -> ToolAction:
        step = state["plan"][state["current_step_index"]]
        return self._choose_action_for_prompt(
            state,
            (
                "执行当前已批准步骤。只选择一个工具调用。"
                "不得选择 allowed_tools 之外的工具。"
            ),
            allowed=set(step["allowed_tools"]),
        )

    def choose_recovery_action(self, state: AgentTaskState) -> ToolAction:
        step = state["plan"][state["current_step_index"]]
        return self._choose_action_for_prompt(
            state,
            (
                "上一次工具调用在完成记录写入前中断。"
                "选择一个工具检查当前状态是否已经满足 expected_result。"
                "不要重复原始修改动作；只返回一个诊断调用。"
            ),
            allowed=set(step["allowed_tools"]),
        )

    def verify_result(self, state: AgentTaskState) -> Verification:
        step = state["plan"][state["current_step_index"]]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是任务结果验证器。只输出一个 JSON 对象："
                    "matched、reason、requires_replan、blocked。"
                    "matched 表示实际证据满足预期；requires_replan 表示原计划必须改变；"
                    "blocked 表示缺少输入、工具、权限或结果仍无法判断。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": state.get("user_request", ""),
                        "step": step,
                        "attempt": state.get("attempt", 0),
                        "action": state.get("current_action", {}),
                        "actual_result": state.get("tool_result", {}),
                        "evidence": state.get("evidence", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._generate_json(messages, self._validate_verification)

    def render_final(self, state: AgentTaskState) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "请根据以下任务状态和可验证证据生成最终回复。"
                    "说明完成了什么、如何验证，或为什么失败/阻塞。"
                    "不要输出原始工具 JSON。\n"
                    + json.dumps(
                        {
                            "request": state.get("user_request", ""),
                            "status": state.get("status", ""),
                            "plan": state.get("plan", []),
                            "evidence": state.get("evidence", []),
                            "verification": state.get("verification", {}),
                            "trace_ref": state.get("trace_ref", ""),
                        },
                        ensure_ascii=False,
                    )
                ),
            },
        ]
        text = str(self.model.generate_reply(messages, self.generation)).strip()
        if not text:
            raise AgentProtocolError("final response is empty")
        return text

    def _choose_action_for_prompt(
        self,
        state: AgentTaskState,
        instruction: str,
        *,
        allowed: set[str],
    ) -> ToolAction:
        step = state["plan"][state["current_step_index"]]
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": state.get("user_request", ""),
                        "frozen_context": state.get("task_context", ""),
                        "step": step,
                        "attempt": state.get("attempt", 0) + 1,
                        "evidence": state.get("evidence", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        native = getattr(self.model, "generate_reply_with_tools", None)
        if callable(native):
            last_error = "agent action must contain exactly one tool call"
            for retry in range(3):
                attempt_messages = self._retry_messages(messages, last_error, retry)
                response = native(
                    attempt_messages,
                    self.generation,
                    self.tools.openai_tools(names=allowed),
                )
                if len(response.tool_calls) != 1:
                    last_error = "agent action must contain exactly one tool call"
                    continue
                call = response.tool_calls[0]
                if call.name not in allowed:
                    last_error = f"tool outside approved step: {call.name}"
                    continue
                if not isinstance(call.arguments, dict):
                    last_error = "tool arguments must be an object"
                    continue
                return {
                    "call_id": call.id,
                    "tool_name": call.name,
                    "arguments": call.arguments,
                    "reason": str(response.content or "").strip(),
                }
            raise AgentProtocolError(last_error)

        schemas = self.tools.openai_tools(names=allowed)
        local_messages = [
            {
                "role": "system",
                "content": (
                    instruction
                    + " 你没有原生 tool_calls 接口。只输出 JSON："
                    + '{"tool_name":"...","arguments":{},"reason":"..."}。'
                ),
            },
            {
                "role": "user",
                "content": messages[1]["content"]
                + "\navailable_tools="
                + json.dumps(schemas, ensure_ascii=False),
            },
        ]

        def validate(payload: dict) -> dict:
            tool_name = self._required_text(payload, "tool_name")
            if tool_name not in allowed:
                raise AgentProtocolError(f"tool outside approved step: {tool_name}")
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                raise AgentProtocolError("tool arguments must be an object")
            return {
                "call_id": f"local_{uuid4().hex[:12]}",
                "tool_name": tool_name,
                "arguments": arguments,
                "reason": self._required_text(payload, "reason"),
            }

        return self._generate_json(local_messages, validate)

    def _generate_json(self, messages: list[dict], validator: Callable[[dict], dict]) -> dict:
        last_error = "empty response"
        structured = getattr(self.model, "generate_structured_reply", None)
        for retry in range(3):
            attempt_messages = self._retry_messages(messages, last_error, retry)
            try:
                if callable(structured):
                    reply = structured(attempt_messages, self.generation)
                    raw = str(getattr(reply, "content", reply)).strip()
                else:
                    raw = str(self.model.generate_reply(attempt_messages, self.generation)).strip()
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise AgentProtocolError("response must be a JSON object")
                return validator(payload)
            except (json.JSONDecodeError, AgentProtocolError, TypeError, ValueError) as exc:
                last_error = str(exc)
        raise AgentProtocolError(last_error)

    def _validate_plan_payload(self, payload: dict) -> dict:
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise AgentProtocolError("plan steps must be a non-empty list")
        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        known_tools = self.tools.names()
        for raw in raw_steps:
            if not isinstance(raw, dict):
                raise AgentProtocolError("plan step must be an object")
            step_id = self._required_text(raw, "id")
            if step_id in seen_ids:
                raise AgentProtocolError(f"duplicate plan step id: {step_id}")
            seen_ids.add(step_id)
            allowed = raw.get("allowed_tools")
            if not isinstance(allowed, list) or not allowed or not all(isinstance(name, str) for name in allowed):
                raise AgentProtocolError(f"allowed_tools must be a non-empty string list: {step_id}")
            unknown = [name for name in allowed if name not in known_tools]
            if unknown:
                raise AgentProtocolError(f"unknown tool in plan: {unknown[0]}")
            steps.append(
                {
                    "id": step_id,
                    "description": self._required_text(raw, "description"),
                    "expected_result": self._required_text(raw, "expected_result"),
                    "verification_method": self._required_text(raw, "verification_method"),
                    "allowed_tools": list(dict.fromkeys(allowed)),
                }
            )
        return {"steps": steps}

    def _validate_verification(self, payload: dict) -> Verification:
        for field in ("matched", "requires_replan", "blocked"):
            if type(payload.get(field)) is not bool:
                raise AgentProtocolError(f"verification {field} must be boolean")
        matched = payload["matched"]
        requires_replan = payload["requires_replan"]
        blocked = payload["blocked"]
        if matched and (requires_replan or blocked):
            raise AgentProtocolError("contradictory verification result")
        return {
            "matched": matched,
            "reason": self._required_text(payload, "reason"),
            "requires_replan": requires_replan,
            "blocked": blocked,
        }

    @staticmethod
    def _required_text(payload: dict, field: str) -> str:
        value = str(payload.get(field, "")).strip()
        if not value:
            raise AgentProtocolError(f"{field} is required")
        return value

    @staticmethod
    def _retry_messages(messages: list[dict], last_error: str, retry: int) -> list[dict]:
        if retry == 0:
            return messages
        return [
            messages[0],
            {
                "role": "user",
                "content": (
                    str(messages[1]["content"])
                    + "\n上一响应未通过协议校验："
                    + last_error
                    + "。请只返回符合要求的结果。"
                ),
            },
        ]
