from __future__ import annotations

from collections.abc import Iterator

from guga.agent.runner import AgentTaskRunner, TaskRunEvent


_TASK_COMMANDS = {"/task", "/tasks", "/resume", "/approve", "/reject"}


class TaskCommandController:
    def __init__(self, runner: AgentTaskRunner) -> None:
        self.runner = runner
        self.selected_task_id: str | None = None

    @staticmethod
    def is_task_command(text: str) -> bool:
        command = str(text).strip().split(maxsplit=1)[0] if str(text).strip() else ""
        return command in _TASK_COMMANDS

    def handle(self, text: str, session_id: str) -> Iterator[TaskRunEvent]:
        normalized = str(text).strip()
        command, _, argument = normalized.partition(" ")
        argument = argument.strip()
        if command not in _TASK_COMMANDS:
            raise ValueError(f"not a task command: {command}")

        if command == "/task":
            if not argument:
                raise ValueError("usage: /task <任务>")
            for event in self.runner.start(argument, session_id):
                self.selected_task_id = event.task_id
                yield event
            return

        if command == "/tasks":
            tasks = self.runner.list_tasks()
            yield TaskRunEvent(
                "task_list",
                self.selected_task_id or "",
                payload={"tasks": tasks},
            )
            return

        if command == "/resume":
            if not argument:
                raise ValueError("usage: /resume <task_id>")
            state = self.runner.get_task(argument)
            self.selected_task_id = argument
            yield TaskRunEvent(
                "task_selected",
                argument,
                payload={
                    "status": state.get("status", ""),
                    "revision": state.get("plan_revision", 0),
                    "plan": state.get("plan", []),
                },
            )
            return

        if argument:
            raise ValueError(f"usage: {command}")
        task_id = self.selected_task_id or self._only_pending_task_id()
        approved = command == "/approve"
        for event in self.runner.resume(task_id, approved=approved):
            yield event

    def _only_pending_task_id(self) -> str:
        tasks = self.runner.list_tasks()
        if not tasks:
            raise RuntimeError("没有可恢复的任务")
        if len(tasks) > 1:
            raise RuntimeError("请先使用 /resume <task_id> 选择任务")
        self.selected_task_id = tasks[0]["task_id"]
        return self.selected_task_id


def format_task_event(event: TaskRunEvent) -> str:
    if event.type in {"awaiting_approval", "task_selected"}:
        revision = event.payload.get("revision", "?")
        lines = [f"任务 {event.task_id} · 第 {revision} 版计划"]
        for index, step in enumerate(event.payload.get("plan", []), start=1):
            tools = ", ".join(step.get("allowed_tools", [])) or "无"
            lines.extend(
                [
                    f"{index}. {step.get('description', '')}",
                    f"   预期：{step.get('expected_result', '')}",
                    f"   验证：{step.get('verification_method', '')}",
                    f"   工具：{tools}",
                ]
            )
        lines.append("输入 /approve 批准，或 /reject 拒绝。")
        return "\n".join(lines)

    if event.type == "task_list":
        tasks = event.payload.get("tasks", [])
        if not tasks:
            return "没有未结束任务。"
        lines = ["未结束任务："]
        lines.extend(
            f"- {task.get('task_id', '')} [{task.get('status', '')}] {task.get('goal', '')}"
            for task in tasks
        )
        return "\n".join(lines)

    if event.type == "terminal":
        status = event.payload.get("status", "")
        trace_ref = event.payload.get("trace_ref", "")
        suffix = f"\nTrace: {trace_ref}" if trace_ref else ""
        return f"任务状态：{status}\n{event.message}{suffix}".strip()

    return event.message or event.type
