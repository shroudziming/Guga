from __future__ import annotations

import json
import unittest

from guga.agent.model_adapter import AgentModelAdapter, AgentProtocolError
from guga.models.structured import StructuredReply
from guga.tools import ToolCall, ToolModelResponse, ToolRegistry, ToolSpec
from guga.types import GenerationConfig


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="guga_read_file",
                description="Read file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                handler=lambda args: {"content": args["path"]},
            ),
            ToolSpec(
                name="guga_run_command",
                description="Run command",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler=lambda args: {"command": args["command"]},
            ),
        ]
    )


def _state() -> dict:
    return {
        "task_id": "task_1",
        "user_request": "读取 README",
        "task_context": "记忆中用户偏好简洁结果",
        "plan": [
            {
                "id": "step_1",
                "description": "读取 README",
                "expected_result": "获得 README 内容",
                "verification_method": "工具返回非空 content",
                "allowed_tools": ["guga_read_file"],
            }
        ],
        "current_step_index": 0,
        "attempt": 0,
        "evidence": [],
    }


class NativeModel:
    def __init__(self, structured_outputs: list[str], tool_output: ToolModelResponse) -> None:
        self.structured_outputs = structured_outputs
        self.tool_output = tool_output
        self.structured_calls = 0
        self.tool_schemas: list[dict] = []
        self.messages_seen: list[list[dict]] = []

    def generate_structured_reply(self, messages, gen):
        self.messages_seen.append(messages)
        output = self.structured_outputs[self.structured_calls]
        self.structured_calls += 1
        return StructuredReply(content=output)

    def generate_reply_with_tools(self, messages, gen, tools):
        self.messages_seen.append(messages)
        self.tool_schemas = tools
        return self.tool_output

    def generate_reply(self, messages, gen):
        self.messages_seen.append(messages)
        return "[happy]任务完成"


class LocalModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.messages_seen: list[list[dict]] = []

    def generate_reply(self, messages, gen):
        self.messages_seen.append(messages)
        output = self.outputs[self.calls]
        self.calls += 1
        return output


class AgentModelAdapterTest(unittest.TestCase):
    def test_planner_requires_workspace_inspection_before_operational_tools(self) -> None:
        valid = json.dumps(
            {
                "steps": [
                    {
                        "id": "step_1",
                        "description": "读取 README",
                        "expected_result": "获得内容",
                        "verification_method": "content 非空",
                        "allowed_tools": ["guga_read_file"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        model = NativeModel([valid], ToolModelResponse(content="", tool_calls=[]))
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        adapter.create_plan(_state())

        planner_prompt = model.messages_seen[0][0]["content"]
        self.assertIn("guga_workspace", planner_prompt)
        self.assertIn("inspect", planner_prompt)
        self.assertIn("set 或 reset 后", planner_prompt)

    def test_plan_retries_invalid_json_without_counting_tool_attempts(self) -> None:
        valid = json.dumps(
            {
                "steps": [
                    {
                        "id": "step_1",
                        "description": "读取 README",
                        "expected_result": "获得内容",
                        "verification_method": "content 非空",
                        "allowed_tools": ["guga_read_file"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        model = NativeModel(
            ["not-json", valid],
            ToolModelResponse(content="", tool_calls=[]),
        )
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        plan = adapter.create_plan(_state())

        self.assertEqual(plan[0]["verification_method"], "content 非空")
        self.assertEqual(model.structured_calls, 2)
        self.assertIn("记忆中用户偏好简洁结果", json.dumps(model.messages_seen[0], ensure_ascii=False))

    def test_plan_rejects_unknown_tool_after_three_protocol_attempts(self) -> None:
        invalid = json.dumps(
            {
                "steps": [
                    {
                        "id": "step_1",
                        "description": "越界",
                        "expected_result": "完成",
                        "verification_method": "检查结果",
                        "allowed_tools": ["unknown_tool"],
                    }
                ]
            }
        )
        model = NativeModel([invalid, invalid, invalid], ToolModelResponse(content="", tool_calls=[]))
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        with self.assertRaisesRegex(AgentProtocolError, "unknown_tool"):
            adapter.create_plan(_state())

        self.assertEqual(model.structured_calls, 3)

    def test_native_tool_calling_receives_only_approved_schema(self) -> None:
        model = NativeModel(
            [],
            ToolModelResponse(
                content="读取当前文件",
                tool_calls=[ToolCall(id="call_1", name="guga_read_file", arguments={"path": "README.md"})],
            ),
        )
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        action = adapter.choose_action(_state())

        self.assertEqual(action["tool_name"], "guga_read_file")
        self.assertEqual(action["arguments"], {"path": "README.md"})
        self.assertEqual([schema["function"]["name"] for schema in model.tool_schemas], ["guga_read_file"])

    def test_local_model_uses_validated_json_action_fallback(self) -> None:
        model = LocalModel(
            [
                json.dumps(
                    {
                        "tool_name": "guga_read_file",
                        "arguments": {"path": "README.md"},
                        "reason": "先读取文件",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        action = adapter.choose_action(_state())

        self.assertEqual(action["tool_name"], "guga_read_file")
        self.assertEqual(action["reason"], "先读取文件")
        self.assertTrue(action["call_id"].startswith("local_"))
        self.assertIn("guga_read_file", json.dumps(model.messages_seen[0], ensure_ascii=False))

    def test_invalid_tool_arguments_are_repaired_before_execution(self) -> None:
        model = LocalModel(
            [
                json.dumps(
                    {
                        "tool_name": "guga_read_file",
                        "arguments": {},
                        "reason": "missing required path",
                    }
                ),
                json.dumps(
                    {
                        "tool_name": "guga_read_file",
                        "arguments": {"path": "README.md"},
                        "reason": "repaired arguments",
                    }
                ),
            ]
        )
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())

        action = adapter.choose_action(_state())

        self.assertEqual({"path": "README.md"}, action["arguments"])
        self.assertEqual(2, model.calls)

    def test_verification_rejects_contradictory_result(self) -> None:
        contradictory = json.dumps(
            {
                "matched": True,
                "reason": "通过",
                "requires_replan": True,
                "blocked": False,
            },
            ensure_ascii=False,
        )
        model = NativeModel(
            [contradictory, contradictory, contradictory],
            ToolModelResponse(content="", tool_calls=[]),
        )
        adapter = AgentModelAdapter(model, GenerationConfig(), "人格", _registry())
        state = _state()
        state["tool_result"] = {"ok": True, "content": "README"}

        with self.assertRaisesRegex(AgentProtocolError, "contradictory"):
            adapter.verify_result(state)

    def test_final_reply_uses_persona_and_task_evidence(self) -> None:
        model = LocalModel(["[happy]任务完成"])
        adapter = AgentModelAdapter(model, GenerationConfig(), "企鹅人格", _registry())
        state = _state()
        state.update({"status": "completed", "trace_ref": "agent-run://default/task_1/trace.jsonl"})

        reply = adapter.render_final(state)

        self.assertEqual(reply, "[happy]任务完成")
        prompt = json.dumps(model.messages_seen[0], ensure_ascii=False)
        self.assertIn("企鹅人格", prompt)
        self.assertIn("agent-run://default/task_1/trace.jsonl", prompt)


if __name__ == "__main__":
    unittest.main()
