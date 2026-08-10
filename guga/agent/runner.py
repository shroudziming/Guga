from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from guga.agent.graph import build_agent_task_graph
from guga.agent.outcome import TaskOutcome
from guga.agent.trace import ExecutionTraceStore
from guga.memory.time_utils import now_beijing_iso
from guga.persona import PersonaOutputParser, PersonaText
from guga.tools import ToolRegistry
from guga.utils.paths import agent_runs_dir


_TERMINAL_STATUSES = {"completed", "failed", "blocked"}


@dataclass(frozen=True)
class TaskRunEvent:
    type: str
    task_id: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class AgentTaskRunner:
    """Own a persistent LangGraph runtime for one persona/agent."""

    def __init__(
        self,
        adapter,
        tools: ToolRegistry,
        memory_manager,
        *,
        agent_id: str,
        runs_root: Path | None = None,
        expression_tags: tuple[str, ...] = (),
    ) -> None:
        self.adapter = adapter
        self.tools = tools
        self.memory_manager = memory_manager
        self.agent_id = str(agent_id).strip()
        if not self.agent_id:
            raise ValueError("agent_id is required")
        self.runs_root = Path(runs_root or agent_runs_dir(self.agent_id)).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.expression_tags = tuple(expression_tags)
        self.trace = ExecutionTraceStore(self.runs_root, self.agent_id)
        self.checkpoint_path = self.runs_root / "checkpoints.sqlite3"
        self._connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        self._checkpointer.setup()
        self.graph = build_agent_task_graph(adapter, tools, self.trace, self._checkpointer)
        self._closed = False

    def start(
        self,
        user_request: str,
        session_id: str,
        *,
        task_id: str | None = None,
    ) -> Iterator[TaskRunEvent]:
        self._ensure_open()
        request = str(user_request).strip()
        if not request:
            raise ValueError("task request is required")
        pending = self.list_tasks()
        if pending:
            raise RuntimeError(
                f"agent has an unfinished task: {pending[0]['task_id']}; resume or reject it first"
            )
        resolved_task_id = task_id or f"task_{uuid4().hex[:16]}"
        context = self.memory_manager.prepare_context(request, session_id)
        frozen_context = self.memory_manager.compose_system_prompt(
            self.adapter.system_prompt,
            context,
            task_mode="Agent Task",
        )
        state = {
            "task_id": resolved_task_id,
            "agent_id": self.agent_id,
            "session_id": session_id,
            "user_request": request,
            "task_context": frozen_context,
            "status": "planning",
            "plan": [],
            "plan_revision": 0,
            "approved_revision": 0,
            "current_step_index": 0,
            "attempt": 0,
            "max_attempts": 3,
            "evidence": [],
            "trace_ref": self.trace.trace_ref(resolved_task_id),
            "recovery_required": False,
        }
        yield from self._stream(state, resolved_task_id)

    def resume(self, task_id: str, *, approved: bool) -> Iterator[TaskRunEvent]:
        self._ensure_open()
        snapshot = self._snapshot(task_id)
        if not snapshot:
            raise KeyError(f"unknown task: {task_id}")
        if snapshot.get("status") in _TERMINAL_STATUSES:
            yield self._terminal_event(task_id, snapshot)
            self._record_outcome(snapshot)
            return
        yield from self._stream(Command(resume={"approved": approved}), task_id)

    def list_tasks(self) -> list[dict[str, str]]:
        self._ensure_open()
        rows = self.trace.list_unfinished()
        for row in rows:
            snapshot = self._snapshot(row["task_id"])
            if snapshot:
                row["status"] = str(snapshot.get("status", "pending"))
        return rows

    def get_task(self, task_id: str) -> dict[str, Any]:
        self._ensure_open()
        snapshot = self._snapshot(task_id)
        if not snapshot:
            raise KeyError(f"unknown task: {task_id}")
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True

    def __enter__(self) -> AgentTaskRunner:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _stream(self, graph_input, task_id: str) -> Iterator[TaskRunEvent]:
        config = self._config(task_id)
        for mode, payload in self.graph.stream(
            graph_input,
            config,
            stream_mode=["updates", "custom"],
            version="v1",
        ):
            if mode != "custom" or not isinstance(payload, dict):
                continue
            event_type = str(payload.get("type", "progress"))
            yield TaskRunEvent(
                type=event_type,
                task_id=task_id,
                message=self._progress_message(payload),
                payload=payload,
            )
        snapshot = self._snapshot(task_id)
        if snapshot.get("status") in _TERMINAL_STATUSES:
            self._record_outcome(snapshot)
            yield self._terminal_event(task_id, snapshot)

    def _record_outcome(self, state: dict[str, Any]) -> None:
        clean_summary = self._clean_persona_text(str(state.get("final_response", "")))
        tools_used: list[str] = []
        for row in self.trace.load(state["task_id"]):
            if row.get("event") != "tool_call_started":
                continue
            tool_name = str(row.get("tool_name", ""))
            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)
        self.memory_manager.record_task_outcome(
            TaskOutcome(
                task_id=state["task_id"],
                goal=state["user_request"],
                status=state["status"],
                summary=clean_summary,
                trace_ref=state["trace_ref"],
                completed_at=now_beijing_iso(),
                tools_used=tuple(tools_used),
            )
        )

    def _terminal_event(self, task_id: str, state: dict[str, Any]) -> TaskRunEvent:
        clean_summary = self._clean_persona_text(str(state.get("final_response", "")))
        return TaskRunEvent(
            type="terminal",
            task_id=task_id,
            message=clean_summary,
            payload={
                "status": state.get("status", ""),
                "trace_ref": state.get("trace_ref", ""),
            },
        )

    def _clean_persona_text(self, text: str) -> str:
        parser = PersonaOutputParser(self.expression_tags)
        events = [*parser.feed(text), *parser.flush()]
        return "".join(event.text for event in events if isinstance(event, PersonaText)).strip()

    def _snapshot(self, task_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(task_id))
        return dict(snapshot.values or {})

    @staticmethod
    def _config(task_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": task_id}}

    @staticmethod
    def _progress_message(payload: dict[str, Any]) -> str:
        event_type = payload.get("type")
        if event_type == "planning":
            return f"正在制定第 {payload.get('revision')} 版计划"
        if event_type == "awaiting_approval":
            return f"第 {payload.get('revision')} 版计划等待批准"
        if event_type == "step_started":
            return f"步骤 {payload.get('step_id')}，第 {payload.get('attempt')} 次尝试"
        if event_type == "tool_started":
            return (
                f"调用 {payload.get('tool_name')}，"
                f"第 {payload.get('attempt')} 次尝试"
            )
        if event_type == "verification_finished":
            outcome = "通过" if payload.get("matched") else "未通过"
            return f"验证{outcome}：{payload.get('reason', '')}"
        if event_type == "recovery_check":
            return "检测中断调用的实际状态"
        if event_type == "plan_revision_requested":
            return "执行结果要求修订计划"
        return str(event_type or "任务状态已更新")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("task runner is closed")
