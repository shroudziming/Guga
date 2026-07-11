from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from guga.memory.time_utils import now_beijing_iso, resolve_event_time


_OPERATIONS = {"create", "update", "replace", "cancel", "ignore"}
_INACTIVE_REASONS = {"replaced", "cancelled", "expired", "invalidated"}
_REFLECTION_KEYS = {"appraisal", "felt_response", "relational_intent", "interpretation_confidence"}


@dataclass(frozen=True)
class EventApplyResult:
    created_event_ids: list[str]
    updated_event_ids: list[str]
    deactivated_event_ids: list[str]


class SemanticEventStore:
    """Persist authoritative semantic events and their lifecycle state."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load_all(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        rows: list[dict] = []
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def load_active(self) -> list[dict]:
        return [row for row in self.load_all() if row.get("status") == "active"]

    def apply_operations(
        self,
        *,
        operations: list[dict],
        session_id: str,
        include_guga_reflection: bool,
    ) -> EventApplyResult:
        rows = copy.deepcopy(self.load_all())
        created: list[str] = []
        updated: list[str] = []
        deactivated: list[str] = []
        now = now_beijing_iso()

        for operation_payload in operations:
            if not isinstance(operation_payload, dict):
                raise ValueError("semantic event operation must be an object")
            operation = str(operation_payload.get("operation", "")).strip()
            if operation not in _OPERATIONS:
                raise ValueError(f"unsupported semantic event operation: {operation}")
            if operation == "ignore":
                continue
            if operation == "create":
                event = self._new_event(operation_payload, session_id, now, include_guga_reflection)
                rows.append(event)
                created.append(event["id"])
                continue

            target_id = str(operation_payload.get("target_event_id", "")).strip()
            target = self._find(rows, target_id)
            if target is None:
                raise ValueError(f"semantic event target not found: {target_id}")

            if operation == "cancel":
                target["status"] = "inactive"
                target["inactive_reason"] = "cancelled"
                target["updated_at"] = now
                target["source_message_ids"] = _merge_ids(target.get("source_message_ids"), operation_payload.get("source_message_ids"))
                deactivated.append(target_id)
                continue

            if operation == "replace":
                target["status"] = "inactive"
                target["inactive_reason"] = "replaced"
                target["updated_at"] = now
                deactivated.append(target_id)
                event = self._new_event(
                    operation_payload,
                    session_id,
                    now,
                    include_guga_reflection,
                    replaces_event_id=target_id,
                )
                rows.append(event)
                created.append(event["id"])
                continue

            self._update_event(target, operation_payload, session_id, now, include_guga_reflection)
            updated.append(target_id)

        self._write_rows(rows)
        return EventApplyResult(created, updated, deactivated)

    def _new_event(
        self,
        payload: dict,
        session_id: str,
        now: str,
        include_guga_reflection: bool,
        replaces_event_id: str | None = None,
    ) -> dict:
        event_kind = _required_text(payload, "event_kind")
        subject = _required_text(payload, "subject")
        entity = _required_text(payload, "entity")
        description = _required_text(payload, "description")
        reference_created_at = str(payload.get("reference_created_at") or now).strip()
        end_unknown = bool(payload.get("end_unknown", False))
        resolved = resolve_event_time(str(payload.get("time_expression", "")), reference_created_at, end_unknown)
        event = {
            "id": f"evt_{uuid4().hex}",
            "type": "semantic_event",
            "event_kind": event_kind,
            "subject": subject,
            "entity": entity,
            "description": description,
            "start_at": resolved.start_at,
            "end_at": resolved.end_at,
            "end_unknown": resolved.end_unknown,
            "time_expression": str(payload.get("time_expression", "")).strip(),
            "time_source": resolved.time_source,
            "time_granularity": resolved.time_granularity,
            "reference_created_at": reference_created_at,
            "status": "active",
            "inactive_reason": None,
            "replaces_event_id": replaces_event_id,
            "source_session_id": session_id,
            "source_message_ids": _merge_ids([], payload.get("source_message_ids")),
            "confidence": _clamp(payload.get("confidence"), 0.8),
            "created_at": now,
            "updated_at": now,
        }
        reflection = _reflection(payload.get("guga_reflection"), include_guga_reflection)
        if reflection is not None:
            event["guga_reflection"] = reflection
        return event

    def _update_event(
        self,
        event: dict,
        payload: dict,
        session_id: str,
        now: str,
        include_guga_reflection: bool,
    ) -> None:
        for key in ("event_kind", "subject", "entity", "description"):
            if key in payload and str(payload[key]).strip():
                event[key] = str(payload[key]).strip()
        if "time_expression" in payload:
            reference_created_at = str(payload.get("reference_created_at") or event["reference_created_at"]).strip()
            resolved = resolve_event_time(
                str(payload.get("time_expression", "")),
                reference_created_at,
                bool(payload.get("end_unknown", event["end_unknown"])),
            )
            event.update(
                {
                    "start_at": resolved.start_at,
                    "end_at": resolved.end_at,
                    "end_unknown": resolved.end_unknown,
                    "time_expression": str(payload.get("time_expression", "")).strip(),
                    "time_source": resolved.time_source,
                    "time_granularity": resolved.time_granularity,
                    "reference_created_at": reference_created_at,
                }
            )
        event["source_session_id"] = session_id
        event["source_message_ids"] = _merge_ids(event.get("source_message_ids"), payload.get("source_message_ids"))
        event["confidence"] = _clamp(payload.get("confidence"), event.get("confidence", 0.8))
        event["updated_at"] = now
        reflection = _reflection(payload.get("guga_reflection"), include_guga_reflection)
        if reflection is not None:
            event["guga_reflection"] = reflection

    def _find(self, rows: list[dict], event_id: str) -> dict | None:
        for row in rows:
            if str(row.get("id", "")) == event_id:
                return row
        return None

    def _write_rows(self, rows: list[dict]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )


def _required_text(payload: dict, key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"semantic event {key} is required")
    return value


def _merge_ids(existing: object, incoming: object) -> list[str]:
    values: list[str] = []
    for source in (existing, incoming):
        if isinstance(source, str):
            source = [source]
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in values:
                values.append(text)
    return values


def _clamp(value: object, fallback: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    return max(0.0, min(number, 1.0))


def _reflection(value: object, include: bool) -> dict | None:
    if not include or not isinstance(value, dict):
        return None
    if set(value) != _REFLECTION_KEYS:
        raise ValueError("guga_reflection has unsupported fields")
    return {
        "appraisal": str(value["appraisal"]).strip(),
        "felt_response": str(value["felt_response"]).strip(),
        "relational_intent": str(value["relational_intent"]).strip(),
        "interpretation_confidence": _clamp(value["interpretation_confidence"], 0.5),
    }
