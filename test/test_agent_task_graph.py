from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from guga.agent.graph import build_agent_task_graph
from guga.agent.model_adapter import AgentProtocolError
from guga.agent.trace import ExecutionTraceStore
from guga.tools import ToolRegistry, ToolSpec


class FakeAdapter:
    def __init__(self) -> None:
        self.plans = [[self.step("step-1")]]
        self.actions = [self.action("work")]
        self.recovery_actions = [self.action("inspect")]
        self.verifications = [self.verification(matched=True)]
        self.plan_calls = 0
        self.action_calls = 0
        self.recovery_calls = 0

    @staticmethod
    def step(step_id: str) -> dict:
        return {
            "id": step_id,
            "description": "执行并验证",
            "expected_result": "状态符合预期",
            "verification_method": "使用 inspect 检查当前状态",
            "allowed_tools": ["work", "inspect"],
        }

    @staticmethod
    def action(tool_name: str) -> dict:
        return {
            "call_id": f"call-{tool_name}",
            "tool_name": tool_name,
            "arguments": {"value": tool_name},
            "reason": "test",
        }

    @staticmethod
    def verification(
        *,
        matched: bool,
        requires_replan: bool = False,
        blocked: bool = False,
    ) -> dict:
        return {
            "matched": matched,
            "reason": "deterministic verifier result",
            "requires_replan": requires_replan,
            "blocked": blocked,
        }

    def create_plan(self, state: dict) -> list[dict]:
        index = min(self.plan_calls, len(self.plans) - 1)
        self.plan_calls += 1
        return self.plans[index]

    def choose_action(self, state: dict) -> dict:
        index = min(self.action_calls, len(self.actions) - 1)
        self.action_calls += 1
        action = self.actions[index]
        if isinstance(action, Exception):
            raise action
        return action

    def choose_recovery_action(self, state: dict) -> dict:
        index = min(self.recovery_calls, len(self.recovery_actions) - 1)
        self.recovery_calls += 1
        return self.recovery_actions[index]

    def verify_result(self, state: dict) -> dict:
        index = min(state.get("verification_count", 0), len(self.verifications) - 1)
        state["verification_count"] = index + 1
        return self.verifications[index]

    def render_final(self, state: dict) -> str:
        return f"terminal:{state['status']}"


class AgentTaskGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.trace = ExecutionTraceStore(self.root / "traces", "default")
        self.calls: list[str] = []
        self.tools = ToolRegistry(
            [
                ToolSpec("work", "work", {"type": "object"}, self._handler("work")),
                ToolSpec("inspect", "inspect", {"type": "object"}, self._handler("inspect")),
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _handler(self, name: str):
        def handle(arguments: dict) -> dict:
            self.calls.append(name)
            return {"ok": True, "observed": arguments.get("value")}

        return handle

    def _state(self, task_id: str = "task-1") -> dict:
        return {
            "task_id": task_id,
            "agent_id": "default",
            "session_id": "session-1",
            "user_request": "执行测试任务",
            "task_context": "frozen context",
            "status": "planning",
            "plan": [],
            "plan_revision": 0,
            "approved_revision": 0,
            "current_step_index": 0,
            "attempt": 0,
            "max_attempts": 3,
            "evidence": [],
            "trace_ref": self.trace.trace_ref(task_id),
        }

    def _graph(self, adapter: FakeAdapter):
        return build_agent_task_graph(adapter, self.tools, self.trace, MemorySaver())

    @staticmethod
    def _config(task_id: str = "task-1") -> dict:
        return {"configurable": {"thread_id": task_id}}

    def test_unapproved_plan_interrupts_before_any_tool_call(self) -> None:
        adapter = FakeAdapter()
        result = self._graph(adapter).invoke(self._state(), self._config())

        self.assertEqual([], self.calls)
        self.assertEqual("awaiting_approval", result["status"])
        self.assertIn("__interrupt__", result)

    def test_approved_step_completes_and_records_trace(self) -> None:
        adapter = FakeAdapter()
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("completed", result["status"])
        self.assertEqual(["work"], self.calls)
        self.assertEqual("terminal:completed", result["final_response"])
        events = [row["event"] for row in self.trace.load("task-1")]
        self.assertIn("plan_approved", events)
        self.assertIn("tool_call_finished", events)
        self.assertIn("verification_finished", events)
        self.assertIn("task_completed", events)

    def test_third_execution_mismatch_enters_failed(self) -> None:
        adapter = FakeAdapter()
        adapter.actions = [adapter.action("work")] * 3
        adapter.verifications = [adapter.verification(matched=False)] * 3
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("failed", result["status"])
        self.assertEqual(["work", "work", "work"], self.calls)
        self.assertEqual(3, result["attempt"])

    def test_revised_plan_requires_a_second_approval(self) -> None:
        adapter = FakeAdapter()
        adapter.plans = [[adapter.step("old")], [adapter.step("new")]]
        adapter.verifications = [
            adapter.verification(matched=False, requires_replan=True),
            adapter.verification(matched=True),
        ]
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        revised = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("awaiting_approval", revised["status"])
        self.assertEqual(2, revised["plan_revision"])
        self.assertEqual(1, revised["approved_revision"])
        self.assertIn("__interrupt__", revised)
        self.assertEqual(["work"], self.calls)

        completed = graph.invoke(Command(resume=True), self._config())
        self.assertEqual("completed", completed["status"])
        self.assertEqual(["work", "work"], self.calls)

    def test_protocol_failure_blocks_without_running_a_tool(self) -> None:
        adapter = FakeAdapter()
        adapter.actions = [AgentProtocolError("invalid action after three repairs")]
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("blocked", result["status"])
        self.assertEqual([], self.calls)

    def test_existing_finished_execution_is_reused(self) -> None:
        adapter = FakeAdapter()
        execution_id = "task-1:r1:step-1:a1"
        self.trace.append_once(
            "task-1",
            "tool_call_started",
            {"execution_id": execution_id},
            event_id=f"{execution_id}:started",
        )
        self.trace.append_once(
            "task-1",
            "tool_call_finished",
            {"execution_id": execution_id, "result": {"ok": True, "reused": True}},
            event_id=f"{execution_id}:finished",
        )
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("completed", result["status"])
        self.assertEqual([], self.calls)
        self.assertTrue(result["tool_result"]["reused"])

    def test_started_execution_is_inspected_instead_of_repeated(self) -> None:
        adapter = FakeAdapter()
        execution_id = "task-1:r1:step-1:a1"
        self.trace.append_once(
            "task-1",
            "tool_call_started",
            {"execution_id": execution_id},
            event_id=f"{execution_id}:started",
        )
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("completed", result["status"])
        self.assertEqual(["inspect"], self.calls)
        self.assertEqual(1, adapter.recovery_calls)

    def test_ambiguous_recovery_enters_blocked(self) -> None:
        adapter = FakeAdapter()
        adapter.verifications = [adapter.verification(matched=False, blocked=True)]
        execution_id = "task-1:r1:step-1:a1"
        self.trace.append_once(
            "task-1",
            "tool_call_started",
            {"execution_id": execution_id},
            event_id=f"{execution_id}:started",
        )
        graph = self._graph(adapter)
        graph.invoke(self._state(), self._config())

        result = graph.invoke(Command(resume=True), self._config())

        self.assertEqual("blocked", result["status"])
        self.assertEqual(["inspect"], self.calls)


if __name__ == "__main__":
    unittest.main()
