from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock

from guga.memory.time_utils import now_beijing_iso


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_TERMINAL_EVENTS = {"task_completed", "task_failed", "task_blocked", "task_rejected"}


class ExecutionTraceStore:
    def __init__(self, root: Path, agent_id: str) -> None:
        self.root = Path(root).resolve()
        self.agent_id = self._validate_id(agent_id, "agent_id")
        self._lock = RLock()

    def trace_ref(self, task_id: str) -> str:
        task_id = self._validate_id(task_id, "task_id")
        return f"agent-run://{self.agent_id}/{task_id}/trace.jsonl"

    def resolve(self, trace_ref: str) -> Path:
        prefix = f"agent-run://{self.agent_id}/"
        if not trace_ref.startswith(prefix):
            raise ValueError(f"trace reference does not belong to agent {self.agent_id}: {trace_ref}")
        relative = trace_ref[len(prefix) :]
        parts = Path(relative).parts
        if len(parts) != 2 or parts[1] != "trace.jsonl":
            raise ValueError(f"invalid trace reference: {trace_ref}")
        task_id = self._validate_id(parts[0], "task_id")
        target = (self.root / task_id / "trace.jsonl").resolve()
        if self.root not in target.parents:
            raise ValueError(f"trace reference escapes agent root: {trace_ref}")
        return target

    def append_once(
        self,
        task_id: str,
        event: str,
        payload: dict,
        *,
        event_id: str,
    ) -> bool:
        task_id = self._validate_id(task_id, "task_id")
        if not event.strip():
            raise ValueError("trace event is required")
        if not event_id.strip():
            raise ValueError("trace event_id is required")
        if not isinstance(payload, dict):
            raise ValueError("trace payload must be an object")

        target = self.resolve(self.trace_ref(task_id))
        with self._lock:
            rows = self._load_path(target)
            if any(row.get("event_id") == event_id for row in rows):
                return False
            row = {
                "sequence": len(rows) + 1,
                "event_id": event_id,
                "event": event.strip(),
                "task_id": task_id,
                "agent_id": self.agent_id,
                "created_at": now_beijing_iso(),
                **payload,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True

    def load(self, task_id: str) -> list[dict]:
        target = self.resolve(self.trace_ref(task_id))
        with self._lock:
            return self._load_path(target)

    def execution_status(self, task_id: str, execution_id: str) -> dict:
        started = False
        for row in self.load(task_id):
            if row.get("execution_id") != execution_id:
                continue
            if row.get("event") == "tool_call_finished":
                return {"state": "finished", "result": row.get("result", {})}
            if row.get("event") == "tool_call_started":
                started = True
        return {"state": "started" if started else "absent"}

    def list_unfinished(self) -> list[dict[str, str]]:
        pending: list[dict[str, str]] = []
        if not self.root.exists():
            return pending
        for trace_file in sorted(self.root.glob("*/trace.jsonl")):
            rows = self._load_path(trace_file)
            created = next((row for row in rows if row.get("event") == "task_created"), None)
            if created is None or any(row.get("event") in _TERMINAL_EVENTS for row in rows):
                continue
            pending.append(
                {
                    "task_id": str(created.get("task_id", "")),
                    "goal": str(created.get("goal", "")),
                    "status": "pending",
                }
            )
        return pending

    def _load_path(self, target: Path) -> list[dict]:
        if not target.exists():
            return []
        rows: list[dict] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"trace row must be an object: {target}")
            rows.append(payload)
        return rows

    @staticmethod
    def _validate_id(value: str, field: str) -> str:
        normalized = str(value).strip()
        if not normalized or _SAFE_ID.fullmatch(normalized) is None:
            raise ValueError(f"invalid {field}: {value!r}")
        return normalized
