from __future__ import annotations

from typing import Any


ALLOWED_MEMORY_SOURCE_TYPES = {"semantic_event", "event_summary", "message"}


def normalize_memory_sources(
    value: object,
    *,
    legacy_event_ids: object = None,
) -> list[dict[str, str]]:
    """Return de-duplicated structured provenance, with legacy event-id support."""
    raw_sources = value if isinstance(value, list) else []
    if not raw_sources and value is None:
        raw_sources = [
            {"type": "semantic_event", "id": event_id}
            for event_id in _ids(legacy_event_ids)
        ]

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("type", "")).strip()
        source_id = str(item.get("id", "")).strip()
        key = (source_type, source_id)
        if source_type not in ALLOWED_MEMORY_SOURCE_TYPES or not source_id or key in seen:
            continue
        seen.add(key)
        sources.append({"type": source_type, "id": source_id})
    return sources


def normalize_payload_sources(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy a stored payload into the unified source representation."""
    normalized = dict(payload)
    normalized["sources"] = normalize_memory_sources(
        payload.get("sources"),
        legacy_event_ids=payload.get("source_event_ids"),
    )
    normalized.pop("source_event_ids", None)
    return normalized


def _ids(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        source_id = str(item).strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result
