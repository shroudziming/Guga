from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from guga.memory.time_utils import now_beijing_iso


class GugaUserModelStore:
    """Store one agent's evidence-backed working understanding of the user."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> dict:
        if not self.file_path.exists():
            return {"schema_version": 1, "updated_at": "", "insights": []}
        payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("guga user model must be an object")
        payload.setdefault("schema_version", 1)
        payload.setdefault("updated_at", "")
        payload.setdefault("insights", [])
        return payload

    def apply_operations(self, operations: list[dict]) -> list[dict]:
        model = self.load()
        insights = [item for item in model["insights"] if isinstance(item, dict)]
        written: list[dict] = []
        now = now_beijing_iso()
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("user model operation must be an object")
            action = str(operation.get("operation", "upsert")).strip()
            if action not in {"upsert", "deactivate"}:
                raise ValueError(f"unsupported user model operation: {action}")
            source_event_ids = _event_ids(operation.get("source_event_ids"))
            if action == "deactivate":
                target_id = str(operation.get("id", "")).strip()
                for insight in insights:
                    if insight.get("id") == target_id:
                        insight["status"] = "inactive"
                        insight["updated_at"] = now
                        written.append(insight)
                        break
                continue
            statement = str(operation.get("statement", "")).strip()
            kind = str(operation.get("kind", "")).strip()
            stability = str(operation.get("stability", "")).strip()
            if not statement or not kind or not stability or not source_event_ids:
                raise ValueError("user model upsert requires statement, kind, stability, and source_event_ids")
            insight_id = str(operation.get("id", "")).strip() or f"gum_{uuid4().hex}"
            insight = {
                "id": insight_id,
                "statement": statement,
                "kind": kind,
                "confidence": _clamp(operation.get("confidence"), 0.7),
                "stability": stability,
                "source_event_ids": source_event_ids,
                "status": "active",
                "updated_at": now,
            }
            for index, existing in enumerate(insights):
                if existing.get("id") == insight_id:
                    insights[index] = insight
                    break
            else:
                insights.append(insight)
            written.append(insight)
        model["insights"] = insights
        model["updated_at"] = now
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return written


def _event_ids(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        event_id = str(item).strip()
        if event_id and event_id not in result:
            result.append(event_id)
    return result


def _clamp(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(0.0, min(number, 1.0))
