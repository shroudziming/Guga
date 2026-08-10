from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.agent.runner import AgentTaskRunner
from guga.agent.trace import ExecutionTraceStore
from guga.tools import ToolRegistry, ToolSpec
from guga.types import MemoryContext
from guga.workspace import WorkspaceContext


class FakeMemoryManager:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, str]] = []
        self.outcomes = []

    def prepare_context(self, user_text: str, session_id: str) -> MemoryContext:
        self.prepared.append((user_text, session_id))
        return MemoryContext(archival_memories=["remembered fact"], user_portrait="user model")

    def compose_system_prompt(
        self,
        base_prompt: str,
        memory_context: MemoryContext,
        task_mode: str = "Conversation",
    ) -> str:
        return f"{task_mode}|{base_prompt}|{memory_context.archival_memories[0]}"

    def record_task_outcome(self, outcome) -> bool:
        if any(item.task_id == outcome.task_id for item in self.outcomes):
            return False
        self.outcomes.append(outcome)
        return True


class FakeAdapter:
    system_prompt = "persona prompt"

    def create_plan(self, state: dict) -> list[dict]:
        return [
            {
                "id": "step-1",
                "description": "run work",
                "expected_result": "done",
                "verification_method": "inspect result",
                "allowed_tools": ["work"],
            }
        ]

    def choose_action(self, state: dict) -> dict:
        return {
            "call_id": "call-1",
            "tool_name": "work",
            "arguments": {},
            "reason": "execute",
        }

    def choose_recovery_action(self, state: dict) -> dict:
        return self.choose_action(state)

    def verify_result(self, state: dict) -> dict:
        return {
            "matched": True,
            "reason": "verified",
            "requires_replan": False,
            "blocked": False,
        }

    def render_final(self, state: dict) -> str:
        return "[happy]任务完成"


class AgentTaskRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory = FakeMemoryManager()
        self.calls = 0
        self.workspace = WorkspaceContext(self.root)
        self.tools = ToolRegistry(
            [
                ToolSpec(
                    "work",
                    "work",
                    {"type": "object", "properties": {}},
                    self._work,
                )
            ],
            workspace=self.workspace,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _work(self, arguments: dict) -> dict:
        self.calls += 1
        return {"ok": True, "value": "done"}

    def _runner(self, memory=None) -> AgentTaskRunner:
        return AgentTaskRunner(
            FakeAdapter(),
            self.tools,
            memory or self.memory,
            agent_id="default",
            runs_root=self.root / "agent_runs" / "default",
            expression_tags=("happy",),
        )

    def test_start_freezes_context_and_waits_for_approval(self) -> None:
        runner = self._runner()
        try:
            events = list(runner.start("read the file", "session-1", task_id="task-1"))
            state = runner.get_task("task-1")
        finally:
            runner.close()

        self.assertEqual([("read the file", "session-1")], self.memory.prepared)
        self.assertIn("Agent Task|persona prompt|remembered fact", state["task_context"])
        self.assertEqual("awaiting_approval", state["status"])
        self.assertEqual(0, self.calls)
        self.assertEqual("awaiting_approval", events[-1].type)
        self.assertEqual("step-1", events[-1].payload["plan"][0]["id"])

    def test_start_invalidates_previous_workspace_confirmation(self) -> None:
        self.workspace.inspect()
        runner = self._runner()
        try:
            list(runner.start("new task", "session-1", task_id="task-1"))
        finally:
            runner.close()

        self.assertFalse(self.workspace.confirmed)

    def test_only_one_unfinished_task_is_allowed_per_agent(self) -> None:
        runner = self._runner()
        try:
            list(runner.start("first", "session-1", task_id="task-1"))
            with self.assertRaisesRegex(RuntimeError, "unfinished task"):
                list(runner.start("second", "session-1", task_id="task-2"))
            pending = runner.list_tasks()
        finally:
            runner.close()

        self.assertEqual(["task-1"], [row["task_id"] for row in pending])

    def test_sqlite_checkpoint_resumes_after_runner_restart(self) -> None:
        first = self._runner()
        list(first.start("restartable", "session-1", task_id="task-1"))
        first.close()

        second = self._runner()
        try:
            events = list(second.resume("task-1", approved=True))
            state = second.get_task("task-1")
        finally:
            second.close()

        self.assertEqual("completed", state["status"])
        self.assertEqual(1, self.calls)
        self.assertEqual("terminal", events[-1].type)

    def test_terminal_outcome_is_clean_and_references_trace(self) -> None:
        runner = self._runner()
        try:
            list(runner.start("remember outcome", "session-1", task_id="task-1"))
            events = list(runner.resume("task-1", approved=True))
            list(runner.resume("task-1", approved=True))
        finally:
            runner.close()

        self.assertEqual("任务完成", events[-1].message)
        self.assertEqual(1, len(self.memory.outcomes))
        outcome = self.memory.outcomes[0]
        self.assertEqual("任务完成", outcome.summary)
        self.assertEqual("agent-run://default/task-1/trace.jsonl", outcome.trace_ref)
        self.assertEqual(("work",), outcome.tools_used)

    def test_custom_stream_exposes_tool_attempt_and_verification(self) -> None:
        runner = self._runner()
        try:
            list(runner.start("stream progress", "session-1", task_id="task-1"))
            events = list(runner.resume("task-1", approved=True))
        finally:
            runner.close()

        tool_event = next(event for event in events if event.type == "tool_started")
        verification = next(event for event in events if event.type == "verification_finished")
        self.assertEqual("work", tool_event.payload["tool_name"])
        self.assertEqual(1, tool_event.payload["attempt"])
        self.assertTrue(verification.payload["matched"])


if __name__ == "__main__":
    unittest.main()
