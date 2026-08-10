from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.agent.outcome import TaskOutcome
from guga.memory.manager import MemoryManager
from guga.memory.task_outcomes import TaskOutcomeStore


def _outcome() -> TaskOutcome:
    return TaskOutcome(
        task_id="task_1",
        goal="运行测试",
        status="completed",
        summary="测试通过",
        trace_ref="agent-run://default/task_1/trace.jsonl",
        completed_at="2026-08-10T12:00:00+08:00",
        tools_used=("guga_run_command",),
    )


class TaskOutcomeStoreTest(unittest.TestCase):
    def test_append_is_idempotent_by_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "task_outcomes.jsonl"
            store = TaskOutcomeStore(target)

            self.assertTrue(store.append(_outcome()))
            self.assertFalse(store.append(_outcome()))

            rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "task_1")
        self.assertEqual(rows[0]["tools_used"], ["guga_run_command"])
        self.assertEqual(rows[0]["trace_ref"], "agent-run://default/task_1/trace.jsonl")

    def test_append_rejects_incomplete_outcome(self) -> None:
        incomplete = TaskOutcome(
            task_id="",
            goal="运行测试",
            status="completed",
            summary="测试通过",
            trace_ref="agent-run://default/task_1/trace.jsonl",
            completed_at="2026-08-10T12:00:00+08:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskOutcomeStore(Path(tmp) / "task_outcomes.jsonl")

            with self.assertRaisesRegex(ValueError, "task_id"):
                store.append(incomplete)


class TaskOutcomeMemoryBridgeTest(unittest.TestCase):
    def test_manager_records_outcome_without_creating_user_semantic_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MemoryManager(memory_root=root, enable_semantic=False)

            self.assertTrue(manager.record_task_outcome(_outcome()))
            self.assertFalse(manager.record_task_outcome(_outcome()))

            outcome_rows = [
                json.loads(line)
                for line in (root / "task_outcomes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            semantic_rows = []
            if (root / "semantic_events.jsonl").exists():
                semantic_rows = [
                    json.loads(line)
                    for line in (root / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()
                ]

        self.assertEqual(len(outcome_rows), 1)
        self.assertFalse(any(row.get("task_id") == "task_1" for row in semantic_rows))

    def test_task_mode_prompt_keeps_retrieved_context_but_changes_mode_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            context = manager.prepare_context("检查之前的计划", "sess_task")

            prompt = manager.compose_system_prompt(
                "人格内容",
                context,
                task_mode="Agent Task",
            )

        self.assertIn("[Task Mode: Agent Task]", prompt)
        self.assertIn("[Persona Skill]\n人格内容", prompt)


if __name__ == "__main__":
    unittest.main()
