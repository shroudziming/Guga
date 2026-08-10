from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    goal: str
    status: str
    summary: str
    trace_ref: str
    completed_at: str
    tools_used: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tools_used"] = list(self.tools_used)
        return payload
