from __future__ import annotations

from guga.memory_sources import normalize_memory_sources


def active_event_ids(events: list[dict]) -> set[str]:
    return {
        str(event.get("id", ""))
        for event in events
        if event.get("status") == "active" and str(event.get("id", ""))
    }


def uses_only_active_event_sources(payload: dict, active_ids: set[str]) -> bool:
    if payload.get("status", "active") != "active":
        return False

    if payload.get("type") == "semantic_event":
        return str(payload.get("id", "")) in active_ids

    sources = normalize_memory_sources(payload.get("sources"))
    source_ids = [item["id"] for item in sources if item["type"] == "semantic_event"]
    if source_ids and not set(source_ids).issubset(active_ids):
        return False

    if payload.get("type") == "event_summary":
        if _ids(payload.get("deactivated_event_ids")):
            return False
        if "covered_event_ids" in payload:
            covered_ids = _ids(payload.get("covered_event_ids"))
            return bool(covered_ids) and set(covered_ids).issubset(active_ids)

    return True


def _ids(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
