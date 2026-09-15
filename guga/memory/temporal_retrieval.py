from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from typing import Any

from guga.memory.time_utils import BEIJING_TZ, parse_datetime
from guga.rag.embedder import BaseEmbedder


_RELATIONS = {"overlap", "before", "after"}
_GRANULARITIES = {"minute", "hour", "day", "week", "month", "year", "range", "session"}


@dataclass(frozen=True)
class TimeQuery:
    original_expression: str
    start_at: datetime
    end_at: datetime
    relation: str
    granularity: str

    def as_dict(self) -> dict[str, str]:
        return {
            "original_expression": self.original_expression,
            "start_at": self.start_at.astimezone(BEIJING_TZ).isoformat(),
            "end_at": self.end_at.astimezone(BEIJING_TZ).isoformat(),
            "relation": self.relation,
            "granularity": self.granularity,
        }


class TemporalQueryValidationError(ValueError):
    def __init__(self, code: str, message: str, query_index: int) -> None:
        super().__init__(message)
        self.code = code
        self.query_index = query_index

    def as_tool_error(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "query_index": self.query_index}


def parse_time_queries(raw_queries: object) -> list[TimeQuery]:
    if not isinstance(raw_queries, list) or not raw_queries:
        raise TemporalQueryValidationError(
            "EMPTY_TIME_QUERIES",
            "time_queries must be a non-empty array",
            -1,
        )
    parsed_queries: list[TimeQuery] = []
    for index, raw in enumerate(raw_queries):
        if not isinstance(raw, dict):
            raise TemporalQueryValidationError(
                "INVALID_TIME_QUERY",
                "time query must be an object",
                index,
            )
        start_raw = str(raw.get("start_at", "")).strip()
        end_raw = str(raw.get("end_at", "")).strip()
        try:
            start_source = datetime.fromisoformat(start_raw)
            end_source = datetime.fromisoformat(end_raw)
        except ValueError as exc:
            raise TemporalQueryValidationError(
                "INVALID_ISO_TIME",
                "start_at and end_at must be ISO 8601 timestamps",
                index,
            ) from exc
        if start_source.tzinfo is None or end_source.tzinfo is None:
            raise TemporalQueryValidationError(
                "INVALID_TIMEZONE",
                "start_at and end_at must include timezone",
                index,
            )
        start = parse_datetime(start_raw)
        end = parse_datetime(end_raw)
        if start is None or end is None:
            raise TemporalQueryValidationError(
                "INVALID_ISO_TIME",
                "start_at and end_at must be ISO 8601 timestamps",
                index,
            )
        if start >= end:
            raise TemporalQueryValidationError(
                "INVALID_TIME_RANGE",
                "start_at must be earlier than end_at",
                index,
            )
        relation = str(raw.get("relation", "overlap")).strip()
        granularity = str(raw.get("granularity", "range")).strip()
        if relation not in _RELATIONS:
            raise TemporalQueryValidationError(
                "INVALID_RELATION",
                f"unsupported relation: {relation}",
                index,
            )
        if granularity not in _GRANULARITIES:
            raise TemporalQueryValidationError(
                "INVALID_GRANULARITY",
                f"unsupported granularity: {granularity}",
                index,
            )
        parsed_queries.append(
            TimeQuery(
                str(raw.get("original_expression", "")).strip(),
                start,
                end,
                relation,
                granularity,
            )
        )
    return parsed_queries


def search_temporal_records(
    queries: list[TimeQuery],
    records: list[dict[str, Any]],
    semantic_query: str,
    embedder: BaseEmbedder | None,
    top_k: int,
) -> dict[str, Any]:
    ranking_degraded = False
    query_results: list[dict[str, Any]] = []

    for query in queries:
        candidates = _matching_candidates(query, records)
        if embedder is not None and candidates:
            try:
                vectors = embedder.encode([semantic_query] + [item["text"] for item in candidates])
                scores = _cosine_scores(vectors, len(candidates))
            except Exception:
                ranking_degraded = True
            else:
                for candidate, score in zip(candidates, scores):
                    candidate["semantic_score"] = score

        candidates.sort(key=_ranking_key)
        for candidate in candidates:
            candidate.pop("_updated_at")
        query_results.append(
            {
                "query": query.as_dict(),
                "matches": candidates[: max(0, top_k)],
            }
        )

    return {"queries": query_results, "ranking_degraded": ranking_degraded}


def _matching_candidates(
    query: TimeQuery,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        for time_value in record.get("time_values", []):
            if not isinstance(time_value, dict) or not _time_value_matches(query, record, time_value):
                continue
            candidates.append(
                {
                    "record_type": str(record["record_type"]),
                    "record_id": str(record["record_id"]),
                    "text": str(record["text"]),
                    "status": str(record.get("status", "active")),
                    "time_fields": dict(record.get("time_fields", {})),
                    "matched_time_field": str(time_value["matched_time_field"]),
                    "time_meaning": str(time_value["time_meaning"]),
                    "semantic_score": None,
                    "_updated_at": _sortable_timestamp(record.get("updated_at")),
                }
            )
    return candidates


def _time_value_matches(
    query: TimeQuery,
    record: dict[str, Any],
    time_value: dict[str, Any],
) -> bool:
    start = _parse_aware_datetime(time_value.get("start_at"))
    if start is None:
        return False

    if time_value.get("is_point") is True:
        if time_value.get("open_end") is True:
            return False
        return _point_matches(query, start)

    if time_value.get("open_end") is True:
        if record.get("record_type") != "archival_memory" or time_value.get("end_at") is not None:
            return False
        return _open_interval_matches(query, start)

    end = _parse_aware_datetime(time_value.get("end_at"))
    if end is None or start >= end:
        return False
    return _closed_interval_matches(query, start, end)


def _parse_aware_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    try:
        source = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if source.tzinfo is None or source.utcoffset() is None:
        return None
    return parse_datetime(raw)


def _point_matches(query: TimeQuery, point: datetime) -> bool:
    if query.relation == "overlap":
        return query.start_at <= point < query.end_at
    if query.relation == "before":
        return point < query.start_at
    return point >= query.end_at


def _closed_interval_matches(query: TimeQuery, start: datetime, end: datetime) -> bool:
    if query.relation == "overlap":
        return start < query.end_at and query.start_at < end
    if query.relation == "before":
        return end <= query.start_at
    return start >= query.end_at


def _open_interval_matches(query: TimeQuery, start: datetime) -> bool:
    if query.relation == "overlap":
        return start < query.end_at
    if query.relation == "before":
        return False
    return start >= query.end_at


def _cosine_scores(vectors: object, candidate_count: int) -> list[float]:
    if not isinstance(vectors, list) or len(vectors) != candidate_count + 1:
        raise ValueError("embedder returned an unexpected vector count")
    normalized = [_normalize_vector(vector) for vector in vectors]
    dimensions = {len(vector) for vector in normalized}
    if dimensions == {0} or len(dimensions) != 1:
        raise ValueError("embedder returned incompatible vector dimensions")
    query_vector = normalized[0]
    return [sum(left * right for left, right in zip(query_vector, vector)) for vector in normalized[1:]]


def _normalize_vector(vector: object) -> list[float]:
    if not isinstance(vector, (list, tuple)):
        raise ValueError("embedding vector must be a sequence")
    values = [float(value) for value in vector]
    if not all(isfinite(value) for value in values):
        raise ValueError("embedding vector values must be finite")
    norm = sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for value in values]
    return [value / norm for value in values]


def _sortable_timestamp(value: object) -> float:
    parsed = parse_datetime(str(value or ""))
    return parsed.timestamp() if parsed is not None else float("-inf")


def _ranking_key(match: dict[str, Any]) -> tuple[float, float, str]:
    score = match["semantic_score"]
    score_key = -score if score is not None else 0.0
    updated_key = -match["_updated_at"]
    return score_key, updated_key, match["record_id"]
