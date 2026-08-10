from __future__ import annotations

import unittest

from guga.agent.cli import TaskCommandController, format_task_event
from guga.agent.runner import TaskRunEvent


class FakeRunner:
    def __init__(self) -> None:
        self.started = []
        self.resumed = []
        self.tasks = [{"task_id": "task-1", "goal": "goal", "status": "awaiting_approval"}]

    def start(self, request: str, session_id: str):
        self.started.append((request, session_id))
        yield TaskRunEvent(
            "awaiting_approval",
            "task-1",
            payload={"revision": 1, "plan": [_plan_step()]},
        )

    def resume(self, task_id: str, *, approved: bool):
        self.resumed.append((task_id, approved))
        yield TaskRunEvent("terminal", task_id, "done", {"status": "completed"})

    def list_tasks(self):
        return list(self.tasks)

    def get_task(self, task_id: str):
        if task_id != "task-1":
            raise KeyError(task_id)
        return {
            "task_id": task_id,
            "status": "awaiting_approval",
            "plan_revision": 1,
            "plan": [_plan_step()],
        }


def _plan_step() -> dict:
    return {
        "id": "step-1",
        "description": "读取配置文件",
        "expected_result": "获得目标值",
        "verification_method": "重新读取并比较",
        "allowed_tools": ["guga_read_file"],
    }


class AgentCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = FakeRunner()
        self.controller = TaskCommandController(self.runner)

    def test_task_starts_and_selects_task_for_approval(self) -> None:
        events = list(self.controller.handle("/task 读取文件", "session-1"))
        list(self.controller.handle("/approve", "session-1"))

        self.assertEqual([("读取文件", "session-1")], self.runner.started)
        self.assertEqual([("task-1", True)], self.runner.resumed)
        self.assertEqual("awaiting_approval", events[-1].type)

    def test_tasks_and_resume_explicitly_select_checkpoint(self) -> None:
        listed = list(self.controller.handle("/tasks", "session-1"))
        selected = list(self.controller.handle("/resume task-1", "session-1"))
        list(self.controller.handle("/reject", "session-1"))

        self.assertEqual("task_list", listed[0].type)
        self.assertEqual("task_selected", selected[0].type)
        self.assertEqual([("task-1", False)], self.runner.resumed)

    def test_approve_without_selected_task_uses_only_pending_task(self) -> None:
        list(self.controller.handle("/approve", "session-1"))
        self.assertEqual([("task-1", True)], self.runner.resumed)

    def test_regular_chat_is_not_a_task_command(self) -> None:
        self.assertFalse(self.controller.is_task_command("你好"))
        self.assertTrue(self.controller.is_task_command("/task 做事"))

    def test_plan_and_progress_format_is_concise(self) -> None:
        plan_text = format_task_event(
            TaskRunEvent(
                "awaiting_approval",
                "task-1",
                payload={"revision": 1, "plan": [_plan_step()]},
            )
        )
        tool_text = format_task_event(
            TaskRunEvent(
                "tool_started",
                "task-1",
                "调用 guga_read_file，第 1 次尝试",
                {
                    "tool_name": "guga_read_file",
                    "attempt": 1,
                    "arguments": {"path": "secret.txt"},
                    "result": {"content": "secret"},
                },
            )
        )

        self.assertIn("读取配置文件", plan_text)
        self.assertIn("预期：获得目标值", plan_text)
        self.assertIn("验证：重新读取并比较", plan_text)
        self.assertIn("/approve", plan_text)
        self.assertEqual("调用 guga_read_file，第 1 次尝试", tool_text)
        self.assertNotIn("secret.txt", tool_text)
        self.assertNotIn("content", tool_text)


if __name__ == "__main__":
    unittest.main()
