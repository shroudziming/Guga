from __future__ import annotations


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

    if "source_event_ids" in payload:
        source_ids = _ids(payload.get("source_event_ids"))
        return bool(source_ids) and set(source_ids).issubset(active_ids)

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
