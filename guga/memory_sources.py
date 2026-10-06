from __future__ import annotations

ALLOWED_MEMORY_SOURCE_TYPES = {"semantic_event", "event_summary", "message"}


def normalize_memory_sources(value: object) -> list[dict[str, str]]:
    """Return de-duplicated structured memory provenance."""
    raw_sources = value if isinstance(value, list) else []
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
