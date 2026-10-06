from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from guga.memory_sources import normalize_memory_sources
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
        insights: list[dict] = []
        for item in payload["insights"]:
            if not isinstance(item, dict):
                continue
            if "source_event_ids" in item:
                raise ValueError("source_event_ids is no longer supported")
            normalized = dict(item)
            normalized["sources"] = normalize_memory_sources(item.get("sources"))
            insights.append(normalized)
        payload["insights"] = insights
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
            if "source_event_ids" in operation:
                raise ValueError("source_event_ids is no longer supported")
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
            if not statement or not kind or not stability:
                raise ValueError("user model upsert requires statement, kind, and stability")
            sources = normalize_memory_sources(operation.get("sources"))
            insight_id = str(operation.get("id", "")).strip() or f"gum_{uuid4().hex}"
            insight = {
                "id": insight_id,
                "statement": statement,
                "kind": kind,
                "confidence": _clamp(operation.get("confidence"), 0.7),
                "stability": stability,
                "sources": sources,
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
        model["schema_version"] = 2
        model["insights"] = insights
        model["updated_at"] = now
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return written

def _clamp(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(0.0, min(number, 1.0))
