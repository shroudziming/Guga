from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from guga.agent.outcome import TaskOutcome


class TaskOutcomeStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self._lock = RLock()

    def append(self, outcome: TaskOutcome) -> bool:
        self._validate(outcome)
        with self._lock:
            rows = self.load_all()
            if any(row.get("task_id") == outcome.task_id for row in rows):
                return False
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome.as_dict(), ensure_ascii=False) + "\n")
            return True

    def load_all(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        rows: list[dict] = []
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"task outcome row must be an object: {self.file_path}")
            rows.append(payload)
        return rows

    @staticmethod
    def _validate(outcome: TaskOutcome) -> None:
        for field in ("task_id", "goal", "status", "summary", "trace_ref", "completed_at"):
            if not str(getattr(outcome, field, "")).strip():
                raise ValueError(f"task outcome {field} is required")
