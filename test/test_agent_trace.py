from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.agent.outcome import TaskOutcome
from guga.agent.trace import ExecutionTraceStore


class TaskOutcomeTest(unittest.TestCase):
    def test_as_dict_serializes_tools_as_json_list(self) -> None:
        outcome = TaskOutcome(
            task_id="task_1",
            goal="读取文件",
            status="completed",
            summary="读取成功",
            trace_ref="agent-run://default/task_1/trace.jsonl",
            completed_at="2026-08-10T12:00:00+08:00",
            tools_used=("guga_read_file",),
        )

        payload = outcome.as_dict()

        self.assertEqual(payload["task_id"], "task_1")
        self.assertEqual(payload["tools_used"], ["guga_read_file"])


class ExecutionTraceStoreTest(unittest.TestCase):
    def test_append_once_preserves_order_and_rejects_duplicate_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExecutionTraceStore(Path(tmp), agent_id="default")

            self.assertTrue(
                store.append_once(
                    "task_1",
                    "task_created",
                    {"goal": "读取文件"},
                    event_id="task_1:created",
                )
            )
            self.assertTrue(
                store.append_once(
                    "task_1",
                    "tool_call_started",
                    {"execution_id": "task_1:r1:s1:a1"},
                    event_id="task_1:r1:s1:a1:started",
                )
            )
            self.assertFalse(
                store.append_once(
                    "task_1",
                    "tool_call_started",
                    {"execution_id": "task_1:r1:s1:a1"},
                    event_id="task_1:r1:s1:a1:started",
                )
            )

            rows = store.load("task_1")

        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["goal"], "读取文件")
        self.assertIn("created_at", rows[0])

    def test_logical_reference_resolves_inside_agent_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ExecutionTraceStore(root, agent_id="gentle")

            trace_ref = store.trace_ref("task_2")
            resolved = store.resolve(trace_ref)

        self.assertEqual(trace_ref, "agent-run://gentle/task_2/trace.jsonl")
        self.assertEqual(resolved, (root / "task_2" / "trace.jsonl").resolve())

    def test_execution_status_distinguishes_absent_started_and_finished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExecutionTraceStore(Path(tmp), agent_id="default")
            execution_id = "task_1:r1:s1:a1"

            self.assertEqual(store.execution_status("task_1", execution_id)["state"], "absent")
            store.append_once(
                "task_1",
                "tool_call_started",
                {"execution_id": execution_id},
                event_id=f"{execution_id}:started",
            )
            self.assertEqual(store.execution_status("task_1", execution_id)["state"], "started")
            store.append_once(
                "task_1",
                "tool_call_finished",
                {"execution_id": execution_id, "result": {"ok": True, "value": 7}},
                event_id=f"{execution_id}:finished",
            )

            status = store.execution_status("task_1", execution_id)

        self.assertEqual(status, {"state": "finished", "result": {"ok": True, "value": 7}})

    def test_list_unfinished_uses_terminal_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExecutionTraceStore(Path(tmp), agent_id="default")
            for task_id, goal in (("task_1", "任务一"), ("task_2", "任务二")):
                store.append_once(
                    task_id,
                    "task_created",
                    {"goal": goal},
                    event_id=f"{task_id}:created",
                )
            store.append_once(
                "task_2",
                "task_completed",
                {"status": "completed"},
                event_id="task_2:terminal",
            )

            pending = store.list_unfinished()

        self.assertEqual(pending, [{"task_id": "task_1", "goal": "任务一", "status": "pending"}])

    def test_trace_file_is_jsonl_for_developer_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExecutionTraceStore(Path(tmp), agent_id="default")
            store.append_once(
                "task_1",
                "plan_created",
                {"revision": 1, "steps": [{"id": "s1"}]},
                event_id="task_1:plan:1",
            )

            raw = Path(store.resolve(store.trace_ref("task_1"))).read_text(encoding="utf-8")
            payload = json.loads(raw.strip())

        self.assertEqual(payload["event"], "plan_created")
        self.assertEqual(payload["steps"], [{"id": "s1"}])


if __name__ == "__main__":
    unittest.main()
