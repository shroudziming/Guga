from __future__ import annotations

from typing import Any, Callable

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from guga.agent.model_adapter import AgentModelAdapter, AgentProtocolError
from guga.agent.state import AgentTaskState
from guga.agent.trace import ExecutionTraceStore
from guga.tools import ToolCall, ToolRegistry


def build_agent_task_graph(
    adapter: AgentModelAdapter,
    tools: ToolRegistry,
    trace: ExecutionTraceStore,
    checkpointer=None,
):
    """Build the approval-gated, checkpointable task execution graph."""

    graph = StateGraph(AgentTaskState)

    def emit(payload: dict[str, Any]) -> None:
        get_stream_writer()(payload)

    def create_plan(state: AgentTaskState) -> dict[str, Any]:
        task_id = state["task_id"]
        trace.append_once(
            task_id,
            "task_created",
            {"goal": state["user_request"], "status": "planning"},
            event_id=f"{task_id}:created",
        )
        revision = state.get("plan_revision", 0) + 1
        emit({"type": "planning", "task_id": task_id, "revision": revision})
        try:
            plan = adapter.create_plan(state)
        except AgentProtocolError as exc:
            return _blocked_update(state, f"无法生成有效计划：{exc}")
        trace.append_once(
            task_id,
            "plan_created",
            {"revision": revision, "plan": plan},
            event_id=f"{task_id}:plan:r{revision}",
        )
        return {
            "status": "awaiting_approval",
            "plan": plan,
            "plan_revision": revision,
            "current_step_index": 0,
            "attempt": 0,
            "recovery_required": False,
        }

    def request_approval(state: AgentTaskState) -> dict[str, Any]:
        revision = state["plan_revision"]
        task_id = state["task_id"]
        emit(
            {
                "type": "awaiting_approval",
                "task_id": task_id,
                "revision": revision,
                "plan": state["plan"],
            }
        )
        response = interrupt(
            {
                "type": "plan_approval",
                "task_id": task_id,
                "revision": revision,
                "plan": state["plan"],
            }
        )
        approved = response if isinstance(response, bool) else bool(response.get("approved", False))
        if not approved:
            reason = "用户拒绝了当前计划"
            trace.append_once(
                task_id,
                "task_rejected",
                {"revision": revision, "reason": reason},
                event_id=f"{task_id}:rejected:r{revision}",
            )
            return {
                "status": "failed",
                "verification": _verification(reason),
            }
        trace.append_once(
            task_id,
            "plan_approved",
            {"revision": revision},
            event_id=f"{task_id}:approved:r{revision}",
        )
        return {"status": "executing", "approved_revision": revision}

    def choose_action(state: AgentTaskState) -> dict[str, Any]:
        step = _current_step(state)
        emit(
            {
                "type": "step_started",
                "task_id": state["task_id"],
                "step_id": step["id"],
                "attempt": state.get("attempt", 0) + 1,
            }
        )
        try:
            action = adapter.choose_action(state)
        except AgentProtocolError as exc:
            message = str(exc)
            if "outside approved step" in message:
                return {
                    "status": "revised_plan",
                    "verification": _verification(message, requires_replan=True),
                }
            return _blocked_update(state, f"连续三次无法生成有效工具调用：{message}")
        if action["tool_name"] not in set(step["allowed_tools"]):
            return {
                "status": "revised_plan",
                "verification": _verification(
                    f"工具 {action['tool_name']} 不在当前步骤许可范围内",
                    requires_replan=True,
                ),
            }
        return {"current_action": action}

    def execute_action(state: AgentTaskState) -> dict[str, Any]:
        attempt = state.get("attempt", 0) + 1
        execution_id = _execution_id(state, attempt)
        existing = trace.execution_status(state["task_id"], execution_id)
        if existing["state"] == "finished":
            result = existing["result"]
            emit(
                {
                    "type": "tool_reused",
                    "task_id": state["task_id"],
                    "step_id": _current_step(state)["id"],
                    "attempt": attempt,
                }
            )
            return {
                "attempt": attempt,
                "tool_result": result,
                "recovery_required": False,
                "evidence": _append_evidence(state, execution_id, result, reused=True),
            }
        if existing["state"] == "started":
            emit(
                {
                    "type": "recovery_check",
                    "task_id": state["task_id"],
                    "step_id": _current_step(state)["id"],
                    "attempt": attempt,
                }
            )
            return {
                "attempt": attempt,
                "recovery_required": True,
                "recovery_execution_id": execution_id,
            }

        action = state["current_action"]
        trace.append_once(
            state["task_id"],
            "tool_call_started",
            {
                "execution_id": execution_id,
                "revision": state["plan_revision"],
                "step_id": _current_step(state)["id"],
                "attempt": attempt,
                "tool_name": action["tool_name"],
                "arguments": action["arguments"],
            },
            event_id=f"{execution_id}:started",
        )
        emit(
            {
                "type": "tool_started",
                "task_id": state["task_id"],
                "step_id": _current_step(state)["id"],
                "tool_name": action["tool_name"],
                "attempt": attempt,
            }
        )
        result = tools.execute(ToolCall(action["call_id"], action["tool_name"], action["arguments"]))
        trace.append_once(
            state["task_id"],
            "tool_call_finished",
            {"execution_id": execution_id, "result": result},
            event_id=f"{execution_id}:finished",
        )
        return {
            "attempt": attempt,
            "tool_result": result,
            "recovery_required": False,
            "evidence": _append_evidence(state, execution_id, result),
        }

    def inspect_interrupted_action(state: AgentTaskState) -> dict[str, Any]:
        try:
            action = adapter.choose_recovery_action(state)
        except AgentProtocolError as exc:
            return _blocked_update(state, f"无法生成恢复检查：{exc}")
        step = _current_step(state)
        if action["tool_name"] not in set(step["allowed_tools"]):
            return _blocked_update(state, "恢复检查使用了计划外工具")
        recovery_id = state["recovery_execution_id"] + ":inspection"
        trace.append_once(
            state["task_id"],
            "recovery_check_started",
            {
                "execution_id": recovery_id,
                "original_execution_id": state["recovery_execution_id"],
                "tool_name": action["tool_name"],
                "arguments": action["arguments"],
            },
            event_id=f"{recovery_id}:started",
        )
        result = tools.execute(ToolCall(action["call_id"], action["tool_name"], action["arguments"]))
        trace.append_once(
            state["task_id"],
            "recovery_check_finished",
            {"execution_id": recovery_id, "result": result},
            event_id=f"{recovery_id}:finished",
        )
        return {
            "current_action": action,
            "tool_result": result,
            "recovery_required": False,
            "evidence": _append_evidence(state, recovery_id, result, recovery=True),
        }

    def verify_action(state: AgentTaskState) -> dict[str, Any]:
        prior_verifications = sum(
            row.get("event") == "verification_finished" for row in trace.load(state["task_id"])
        )
        adapter_state = dict(state)
        adapter_state["verification_count"] = prior_verifications
        try:
            verification = adapter.verify_result(adapter_state)
        except AgentProtocolError as exc:
            return _blocked_update(state, f"连续三次无法生成有效验证结果：{exc}")
        event_id = (
            f"{state['task_id']}:verify:r{state['plan_revision']}:"
            f"{_current_step(state)['id']}:a{state['attempt']}"
        )
        trace.append_once(
            state["task_id"],
            "verification_finished",
            {
                "revision": state["plan_revision"],
                "step_id": _current_step(state)["id"],
                "attempt": state["attempt"],
                "verification": verification,
            },
            event_id=event_id,
        )
        emit(
            {
                "type": "verification_finished",
                "task_id": state["task_id"],
                "step_id": _current_step(state)["id"],
                "attempt": state["attempt"],
                "matched": verification["matched"],
                "reason": verification["reason"],
            }
        )
        return {"verification": verification}

    def advance_step(state: AgentTaskState) -> dict[str, Any]:
        next_index = state["current_step_index"] + 1
        if next_index >= len(state["plan"]):
            return {"status": "completed"}
        return {
            "current_step_index": next_index,
            "attempt": 0,
            "recovery_required": False,
        }

    def revise_plan(state: AgentTaskState) -> dict[str, Any]:
        trace.append_once(
            state["task_id"],
            "plan_revision_requested",
            {
                "revision": state["plan_revision"],
                "reason": state.get("verification", {}).get("reason", "计划需要修订"),
            },
            event_id=f"{state['task_id']}:revise:r{state['plan_revision']}",
        )
        emit(
            {
                "type": "plan_revision_requested",
                "task_id": state["task_id"],
                "revision": state["plan_revision"],
            }
        )
        return {
            "status": "revised_plan",
            "current_step_index": 0,
            "attempt": 0,
            "recovery_required": False,
        }

    def complete(state: AgentTaskState) -> dict[str, Any]:
        return _terminal_update(adapter, trace, state, "completed", "task_completed")

    def fail(state: AgentTaskState) -> dict[str, Any]:
        return _terminal_update(adapter, trace, state, "failed", "task_failed")

    def block(state: AgentTaskState) -> dict[str, Any]:
        return _terminal_update(adapter, trace, state, "blocked", "task_blocked")

    graph.add_node("planning", create_plan)
    graph.add_node("awaiting_approval", request_approval)
    graph.add_node("choose_action", choose_action)
    graph.add_node("executing", execute_action)
    graph.add_node("recovering", inspect_interrupted_action)
    graph.add_node("verifying", verify_action)
    graph.add_node("advance_step", advance_step)
    graph.add_node("revised_plan", revise_plan)
    graph.add_node("completed", complete)
    graph.add_node("failed", fail)
    graph.add_node("blocked", block)

    graph.add_edge(START, "planning")
    graph.add_conditional_edges(
        "planning",
        lambda state: "blocked" if state["status"] == "blocked" else "awaiting_approval",
    )
    graph.add_conditional_edges(
        "awaiting_approval",
        lambda state: "choose_action" if state["status"] == "executing" else "failed",
    )
    graph.add_conditional_edges(
        "choose_action",
        _route_action_selection,
        {
            "execute": "executing",
            "revise": "revised_plan",
            "block": "blocked",
        },
    )
    graph.add_conditional_edges(
        "executing",
        lambda state: "recovering" if state.get("recovery_required") else "verifying",
    )
    graph.add_conditional_edges(
        "recovering",
        lambda state: "blocked" if state["status"] == "blocked" else "verifying",
    )
    graph.add_conditional_edges(
        "verifying",
        _route_verification,
        {
            "advance": "advance_step",
            "retry": "choose_action",
            "revise": "revised_plan",
            "fail": "failed",
            "block": "blocked",
        },
    )
    graph.add_conditional_edges(
        "advance_step",
        lambda state: "completed" if state["status"] == "completed" else "choose_action",
    )
    graph.add_edge("revised_plan", "planning")
    graph.add_edge("completed", END)
    graph.add_edge("failed", END)
    graph.add_edge("blocked", END)
    return graph.compile(checkpointer=checkpointer, name="guga_agent_task")


def _current_step(state: AgentTaskState) -> dict[str, Any]:
    return state["plan"][state["current_step_index"]]


def _execution_id(state: AgentTaskState, attempt: int) -> str:
    return (
        f"{state['task_id']}:r{state['plan_revision']}:"
        f"{_current_step(state)['id']}:a{attempt}"
    )


def _append_evidence(
    state: AgentTaskState,
    execution_id: str,
    result: dict[str, Any],
    *,
    reused: bool = False,
    recovery: bool = False,
) -> list[dict[str, Any]]:
    return [
        *state.get("evidence", []),
        {
            "execution_id": execution_id,
            "step_id": _current_step(state)["id"],
            "result": result,
            "reused": reused,
            "recovery": recovery,
        },
    ]


def _verification(
    reason: str,
    *,
    requires_replan: bool = False,
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "matched": False,
        "reason": reason,
        "requires_replan": requires_replan,
        "blocked": blocked,
    }


def _blocked_update(state: AgentTaskState, reason: str) -> dict[str, Any]:
    return {"status": "blocked", "verification": _verification(reason, blocked=True)}


def _route_action_selection(state: AgentTaskState) -> str:
    if state["status"] == "blocked":
        return "block"
    if state["status"] == "revised_plan":
        return "revise"
    return "execute"


def _route_verification(state: AgentTaskState) -> str:
    verification = state["verification"]
    if verification["matched"]:
        return "advance"
    if verification["blocked"]:
        return "block"
    if verification["requires_replan"]:
        return "revise"
    if state.get("recovery_required"):
        return "block"
    if state["attempt"] >= state.get("max_attempts", 3):
        return "fail"
    return "retry"


def _terminal_update(
    adapter: AgentModelAdapter,
    trace: ExecutionTraceStore,
    state: AgentTaskState,
    status: str,
    event: str,
) -> dict[str, Any]:
    terminal_state = dict(state)
    terminal_state["status"] = status
    try:
        response = adapter.render_final(terminal_state)
    except AgentProtocolError:
        reason = state.get("verification", {}).get("reason", "任务运行结束")
        response = f"任务状态：{status}。{reason}"
    trace.append_once(
        state["task_id"],
        event,
        {
            "status": status,
            "summary": response,
            "trace_ref": state["trace_ref"],
        },
        event_id=f"{state['task_id']}:terminal:{status}",
    )
    get_stream_writer()(
        {"type": "task_terminal", "task_id": state["task_id"], "status": status}
    )
    return {"status": status, "final_response": response}
