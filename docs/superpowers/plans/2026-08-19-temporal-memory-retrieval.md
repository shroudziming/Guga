# Temporal Memory Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-resolved, tool-driven temporal memory retrieval to the existing Agent Loop while persisting each successful time interpretation on the corresponding session turn.

**Architecture:** Introduce a pure temporal query/matching module, expose it through `MemoryManager.search_time_context`, and register `memory_time_search` as a normal tool in every `ChatSession`. The existing Tool Calling loop remains authoritative: any tool call continues the loop, and the absence of a time call never implies that the response is final. Ordinary semantic retrieval remains independent and stops performing its current regex-based date filtering.

**Tech Stack:** Python 3, dataclasses, JSONL session/memory stores, `sentence-transformers` BGE-M3 embeddings, FAISS-backed existing RAG pipeline, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-18-temporal-memory-retrieval-consolidation-design.md`

## Global Constraints

- Interpret every interval as left-closed and right-open: $[start\_at, end\_at)$.
- Require timezone-aware ISO 8601 values and normalize them to `+08:00`.
- Keep semantic retrieval and temporal retrieval independent; do not intersect their result sets.
- Match every normalized query against every supported temporal field, and preserve the matched field name and meaning on each result.
- Search new canonical data only. Do not add compatibility inference for incomplete legacy temporal fields.
- Include inactive semantic events in temporal matching and preserve lifecycle state in results.
- Use BGE-M3 only to order records that already passed structural time filtering.
- Preserve all existing non-time tools and the existing maximum Agent Loop tool-round limit.
- Do not solve cross-layer duplicate memory in this plan.
- Use TDD and commit each independently verified task with the repository commit format.

---

### Task 1: Canonical time-query validation and interval search

**Files:**
- Create: `guga/memory/temporal_retrieval.py`
- Create: `test/test_temporal_retrieval.py`

**Interfaces:**
- Consumes: `guga.memory.time_utils.parse_datetime`, `guga.memory.time_utils.format_beijing`, and an optional `guga.rag.embedder.BaseEmbedder`.
- Produces: `TimeQuery`, `TemporalQueryValidationError`, `parse_time_queries(raw_queries)`, and `search_temporal_records(queries, records, semantic_query, embedder, top_k)`.
- Canonical record input keys: `record_type`, `record_id`, `text`, `status`, `time_values`, `updated_at`, and `time_fields`. Each `time_values` item contains `start_at`, optional `end_at`, `is_point`, `open_end`, `matched_time_field`, and `time_meaning`.
- Search output: `{"queries": [{"query": <canonical query>, "matches": [<structured match>]}], "ranking_degraded": bool}`.

- [ ] **Step 1: Write failing contract, boundary, lifecycle, and ranking tests**

```python
from __future__ import annotations

import unittest

from guga.memory.temporal_retrieval import (
    TemporalQueryValidationError,
    parse_time_queries,
    search_temporal_records,
)


class KeywordEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "面试" in text else [0.0, 1.0] for text in texts]


class BrokenEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")


class TemporalRetrievalTest(unittest.TestCase):
    def test_rejects_naive_or_empty_intervals_with_query_index(self) -> None:
        with self.assertRaises(TemporalQueryValidationError) as caught:
            parse_time_queries(
                [{
                    "original_expression": "8月20号",
                    "start_at": "2026-08-20T00:00:00",
                    "end_at": "2026-08-21T00:00:00+08:00",
                    "relation": "overlap",
                    "granularity": "day",
                }]
            )
        self.assertEqual(caught.exception.code, "INVALID_TIMEZONE")
        self.assertEqual(caught.exception.query_index, 0)

    def test_half_open_overlap_excludes_event_starting_at_query_end(self) -> None:
        query = parse_time_queries([{
            "original_expression": "8月20号",
            "start_at": "2026-08-20T00:00:00+08:00",
            "end_at": "2026-08-21T00:00:00+08:00",
            "relation": "overlap",
            "granularity": "day",
        }])
        records = [{
            "record_type": "semantic_event",
            "record_id": "evt_boundary",
            "text": "第二天开始的安排",
            "status": "active",
            "time_values": [{
                "start_at": "2026-08-21T00:00:00+08:00",
                "end_at": "2026-08-21T01:00:00+08:00",
                "is_point": False,
                "open_end": False,
                "matched_time_field": "start_at/end_at",
                "time_meaning": "事件实际发生区间",
            }],
            "updated_at": "2026-08-18T10:00:00+08:00",
            "time_fields": {},
        }]
        result = search_temporal_records(query, records, "有什么安排", None, 5)
        self.assertEqual(result["queries"][0]["matches"], [])

    def test_keeps_cancelled_event_and_ranks_time_candidates_with_embedder(self) -> None:
        query = parse_time_queries([{
            "original_expression": "8月20号",
            "start_at": "2026-08-20T00:00:00+08:00",
            "end_at": "2026-08-21T00:00:00+08:00",
            "relation": "overlap",
            "granularity": "day",
        }])
        records = [
            {
                "record_type": "semantic_event",
                "record_id": "evt_dinner",
                "text": "晚上聚餐",
                "status": "active",
                "time_values": [{"start_at": "2026-08-20T18:00:00+08:00", "end_at": "2026-08-20T19:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}],
                "updated_at": "2026-08-18T09:00:00+08:00",
                "time_fields": {},
            },
            {
                "record_type": "semantic_event",
                "record_id": "evt_interview",
                "text": "参加面试",
                "status": "cancelled",
                "time_values": [{"start_at": "2026-08-20T14:00:00+08:00", "end_at": "2026-08-20T15:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}],
                "updated_at": "2026-08-19T09:00:00+08:00",
                "time_fields": {},
            },
        ]
        result = search_temporal_records(query, records, "面试安排", KeywordEmbedder(), 5)
        matches = result["queries"][0]["matches"]
        self.assertEqual([item["record_id"] for item in matches], ["evt_interview", "evt_dinner"])
        self.assertEqual(matches[0]["status"], "cancelled")
        self.assertEqual(matches[0]["semantic_score"], 1.0)

    def test_created_at_candidate_is_annotated_and_embedding_failure_degrades(self) -> None:
        query = parse_time_queries([{
            "original_expression": "昨天聊的内容",
            "start_at": "2026-08-18T00:00:00+08:00",
            "end_at": "2026-08-19T00:00:00+08:00",
            "relation": "overlap",
            "granularity": "day",
        }])
        records = [{
            "record_type": "conversation_turn",
            "record_id": "turn_1",
            "text": "用户询问了面试",
            "status": "active",
            "time_values": [{
                "start_at": "2026-08-18T12:00:00+08:00",
                "end_at": None,
                "is_point": True,
                "open_end": False,
                "matched_time_field": "created_at",
                "time_meaning": "这条消息是在该时间创建的，不代表消息内容中的事件发生于此时",
            }],
            "updated_at": "2026-08-18T12:00:00+08:00",
            "time_fields": {"created_at": "2026-08-18T12:00:00+08:00"},
        }]
        result = search_temporal_records(query, records, "昨天聊了什么", BrokenEmbedder(), 5)
        self.assertEqual(result["queries"][0]["matches"][0]["record_id"], "turn_1")
        self.assertEqual(result["queries"][0]["matches"][0]["matched_time_field"], "created_at")
        self.assertTrue(result["ranking_degraded"])

    def test_multiple_queries_group_before_and_after_results(self) -> None:
        queries = parse_time_queries([
            {"original_expression": "8月20号之前", "start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00", "relation": "before", "granularity": "day"},
            {"original_expression": "8月20号之后", "start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00", "relation": "after", "granularity": "day"},
        ])
        records = [
            {"record_type": "semantic_event", "record_id": "evt_before", "text": "之前的事情", "status": "active", "time_values": [{"start_at": "2026-08-19T09:00:00+08:00", "end_at": "2026-08-19T10:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}], "updated_at": "2026-08-19T10:00:00+08:00", "time_fields": {}},
            {"record_type": "semantic_event", "record_id": "evt_after", "text": "之后的事情", "status": "active", "time_values": [{"start_at": "2026-08-21T09:00:00+08:00", "end_at": "2026-08-21T10:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}], "updated_at": "2026-08-21T10:00:00+08:00", "time_fields": {}},
        ]
        result = search_temporal_records(queries, records, "事情", None, 5)
        self.assertEqual(len(result["queries"]), 2)
        self.assertEqual(result["queries"][0]["matches"][0]["record_id"], "evt_before")
        self.assertEqual(result["queries"][1]["matches"][0]["record_id"], "evt_after")

    def test_open_ended_archival_interval_overlaps_future_query(self) -> None:
        query = parse_time_queries([{"original_expression": "8月20号", "start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00", "relation": "overlap", "granularity": "day"}])
        records = [{
            "record_type": "archival_memory",
            "record_id": "mem_open",
            "text": "用户正在准备面试",
            "status": "active",
            "time_values": [{"start_at": "2026-08-18T10:00:00+08:00", "end_at": None, "is_point": False, "open_end": True, "matched_time_field": "valid_at/invalid_at", "time_meaning": "事实成立的有效区间"}],
            "updated_at": "2026-08-18T10:00:00+08:00",
            "time_fields": {"valid_at": "2026-08-18T10:00:00+08:00", "invalid_at": ""},
        }]
        result = search_temporal_records(query, records, "面试", None, 5)
        self.assertEqual(result["queries"][0]["matches"][0]["record_id"], "mem_open")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify the module is missing**

Run: `python -m unittest test.test_temporal_retrieval -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'guga.memory.temporal_retrieval'`.

- [ ] **Step 3: Implement the canonical query and pure search module**

Create the following public contract in `guga/memory/temporal_retrieval.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from guga.memory.time_utils import format_beijing, parse_datetime
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
            "start_at": format_beijing(self.start_at),
            "end_at": format_beijing(self.end_at),
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
        raise TemporalQueryValidationError("EMPTY_TIME_QUERIES", "time_queries must be a non-empty array", -1)
    parsed_queries: list[TimeQuery] = []
    for index, raw in enumerate(raw_queries):
        if not isinstance(raw, dict):
            raise TemporalQueryValidationError("INVALID_TIME_QUERY", "time query must be an object", index)
        start_raw = str(raw.get("start_at", "")).strip()
        end_raw = str(raw.get("end_at", "")).strip()
        try:
            start_source = datetime.fromisoformat(start_raw)
            end_source = datetime.fromisoformat(end_raw)
        except ValueError as exc:
            raise TemporalQueryValidationError("INVALID_ISO_TIME", "start_at and end_at must be ISO 8601 timestamps", index) from exc
        if start_source.tzinfo is None or end_source.tzinfo is None:
            raise TemporalQueryValidationError("INVALID_TIMEZONE", "start_at and end_at must include timezone", index)
        start = parse_datetime(start_raw)
        end = parse_datetime(end_raw)
        if start is None or end is None:
            raise TemporalQueryValidationError("INVALID_ISO_TIME", "start_at and end_at must be ISO 8601 timestamps", index)
        if start >= end:
            raise TemporalQueryValidationError("INVALID_TIME_RANGE", "start_at must be earlier than end_at", index)
        relation = str(raw.get("relation", "overlap")).strip()
        granularity = str(raw.get("granularity", "range")).strip()
        if relation not in _RELATIONS:
            raise TemporalQueryValidationError("INVALID_RELATION", f"unsupported relation: {relation}", index)
        if granularity not in _GRANULARITIES:
            raise TemporalQueryValidationError("INVALID_GRANULARITY", f"unsupported granularity: {granularity}", index)
        parsed_queries.append(TimeQuery(str(raw.get("original_expression", "")).strip(), start, end, relation, granularity))
    return parsed_queries
```

Complete the same file with these exact search rules:

- Iterate every record's `time_values`; do not choose a field category before matching.
- Closed intervals require valid timezone-aware starts and ends with `start < end`. Only `record_type=archival_memory` may use `{end_at: None, open_end: True}` for intentionally open-ended validity.
- Point values such as `created_at` use `query.start_at <= point < query.end_at` for overlap.
- `before` accepts a closed interval only when `candidate_end <= query.start_at`; a point must be `< query.start_at`.
- `after` accepts an interval only when `candidate_start >= query.end_at`; a point must be `>= query.end_at`.
- An open-ended archival interval overlaps when `candidate_start < query.end_at`, never satisfies `before`, and satisfies `after` only when its start is at or after `query.end_at`.
- Create one match for every matching time value. Do not deduplicate records that match multiple fields in this phase.
- Encode `[semantic_query] + candidate_texts` once per query and compute normalized-vector dot products.
- On any embedder exception, set `ranking_degraded=True`, leave `semantic_score=None`, and sort by `updated_at` descending followed by `record_id` ascending.
- On successful embedding, sort by `semantic_score` descending, then `updated_at` descending, then `record_id` ascending.
- Merge all matches, apply one global semantic sort, and return only the first total `top_k` matches for each query. Do not reserve per-field quotas.

Use this match shape:

```python
match = {
    "record_type": str(record["record_type"]),
    "record_id": str(record["record_id"]),
    "text": str(record["text"]),
    "status": str(record.get("status", "active")),
    "time_fields": dict(record.get("time_fields", {})),
    "matched_time_field": str(time_value["matched_time_field"]),
    "time_meaning": str(time_value["time_meaning"]),
    "semantic_score": score,
}
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest test.test_temporal_retrieval -v`

Expected: all six tests PASS.

- [ ] **Step 5: Commit the pure retrieval engine**

```powershell
git add guga/memory/temporal_retrieval.py test/test_temporal_retrieval.py
git commit -m "feat(memory): 增加规范时间区间检索"
```

### Task 2: Load all temporal memory layers through MemoryManager

**Files:**
- Modify: `guga/memory/manager.py`
- Modify: `test/test_memory_manager.py`

**Interfaces:**
- Consumes: `parse_time_queries` and `search_temporal_records` from Task 1.
- Produces: `MemoryManager.search_time_context(*, session_id: str, user_text: str, reference_time: str, time_queries: object) -> dict[str, Any]`.
- Produces private canonical loader: `MemoryManager._load_temporal_records() -> list[dict]`.

Add `from unittest.mock import patch` to `test/test_memory_manager.py` for the storage-failure test below.

- [ ] **Step 1: Write failing manager-layer tests with new-format records only**

Add tests that seed all four layers without using legacy field inference:

```python
    def test_search_time_context_reads_layers_and_preserves_event_lifecycle(self) -> None:
        event = {
            "id": "evt_interview",
            "type": "semantic_event",
            "description": "参加面试",
            "start_at": "2026-08-20T14:00:00+08:00",
            "end_at": "2026-08-20T15:00:00+08:00",
            "status": "inactive",
            "inactive_reason": "cancelled",
            "updated_at": "2026-08-19T09:00:00+08:00",
        }
        self.manager.semantic_event_file.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
        result = self.manager.search_time_context(
            session_id="sess_time",
            user_text="8月20号有什么面试安排",
            reference_time="2026-08-19T10:00:00+08:00",
            time_queries=[{
                "original_expression": "8月20号",
                "start_at": "2026-08-20T00:00:00+08:00",
                "end_at": "2026-08-21T00:00:00+08:00",
                "relation": "overlap",
                "granularity": "day",
            }],
        )
        self.assertTrue(result["ok"])
        match = result["queries"][0]["matches"][0]
        self.assertEqual(match["record_id"], "evt_interview")
        self.assertEqual(match["status"], "cancelled")

    def test_search_time_context_returns_structured_validation_error(self) -> None:
        result = self.manager.search_time_context(
            session_id="sess_invalid",
            user_text="查询安排",
            reference_time="2026-08-19T10:00:00+08:00",
            time_queries=[{
                "original_expression": "错误区间",
                "start_at": "2026-08-21T00:00:00+08:00",
                "end_at": "2026-08-20T00:00:00+08:00",
                "relation": "overlap",
                "granularity": "day",
            }],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_TIME_RANGE")
        self.assertEqual(result["error"]["query_index"], 0)

    def test_search_time_context_reports_storage_failure_instead_of_empty_matches(self) -> None:
        with patch.object(self.manager, "_load_temporal_records", side_effect=OSError("disk unavailable")):
            result = self.manager.search_time_context(
                session_id="sess_io",
                user_text="8月20号有什么事情",
                reference_time="2026-08-19T10:00:00+08:00",
                time_queries=[{
                    "original_expression": "8月20号",
                    "start_at": "2026-08-20T00:00:00+08:00",
                    "end_at": "2026-08-21T00:00:00+08:00",
                    "relation": "overlap",
                    "granularity": "day",
                }],
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "TEMPORAL_STORE_READ_FAILED")
```

Add this session-row test for the unified time-query path:

```python
    def test_search_time_context_searches_turn_semantic_time_and_message_created_at(self) -> None:
        session_dir = self.memory_root / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "id": "msg_prior",
            "session_id": "sess_prior",
            "role": "user",
            "content": "我8月20号有个面试",
            "created_at": "2026-08-18T10:00:00+08:00",
            "metadata": {"turn": {"turn_id": "turn_msg_prior", "temporal_context": {
                "source_message_id": "msg_prior",
                "reference_time": "2026-08-18T10:00:00+08:00",
                "time_queries": [{
                    "original_expression": "8月20号",
                    "start_at": "2026-08-20T00:00:00+08:00",
                    "end_at": "2026-08-21T00:00:00+08:00",
                    "relation": "overlap",
                    "granularity": "day",
                }],
            }}},
        }
        (session_dir / "sess_prior.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        semantic = self.manager.search_time_context(
            session_id="sess_now",
            user_text="8月20号有什么",
            reference_time="2026-08-19T10:00:00+08:00",
            time_queries=[{**row["metadata"]["turn"]["temporal_context"]["time_queries"][0]}],
        )
        conversation_day = self.manager.search_time_context(
            session_id="sess_now",
            user_text="8月18号聊了什么",
            reference_time="2026-08-19T10:00:00+08:00",
            time_queries=[{
                "original_expression": "8月18号聊的内容",
                "start_at": "2026-08-18T00:00:00+08:00",
                "end_at": "2026-08-19T00:00:00+08:00",
                "relation": "overlap",
                "granularity": "day",
            }],
        )
        self.assertEqual(semantic["queries"][0]["matches"][0]["record_id"], "turn_msg_prior")
        self.assertEqual(semantic["queries"][0]["matches"][0]["matched_time_field"], "temporal_context.time_queries")
        self.assertEqual(conversation_day["queries"][0]["matches"][0]["record_id"], "turn_msg_prior")
        self.assertEqual(conversation_day["queries"][0]["matches"][0]["matched_time_field"], "created_at")
```

- [ ] **Step 2: Run the focused manager tests and verify the method is missing**

Run: `python -m unittest test.test_memory_manager.MemoryManagerTest.test_search_time_context_reads_layers_and_preserves_event_lifecycle test.test_memory_manager.MemoryManagerTest.test_search_time_context_returns_structured_validation_error -v`

Expected: FAIL with `AttributeError: 'MemoryManager' object has no attribute 'search_time_context'`.

- [ ] **Step 3: Implement strict layer loading and the public tool handler target**

Add imports:

```python
from typing import Any

from guga.memory.temporal_retrieval import (
    TemporalQueryValidationError,
    parse_time_queries,
    search_temporal_records,
)
```

Implement this public method on `MemoryManager`:

```python
    def search_time_context(
        self,
        *,
        session_id: str,
        user_text: str,
        reference_time: str,
        time_queries: object,
    ) -> dict[str, Any]:
        started = perf_counter()
        try:
            queries = parse_time_queries(time_queries)
        except TemporalQueryValidationError as exc:
            self._debug(session_id, f"time_search status=invalid error={json.dumps(exc.as_tool_error(), ensure_ascii=False)}")
            return {"ok": False, "type": "time_context", "error": exc.as_tool_error()}
        try:
            records = self._load_temporal_records()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            error = {"code": "TEMPORAL_STORE_READ_FAILED", "message": str(exc), "query_index": -1}
            self._debug(session_id, f"time_search status=storage_failed error={json.dumps(error, ensure_ascii=False)}")
            return {"ok": False, "type": "time_context", "error": error}
        embedder = self.rag_pipeline.embedder if self.rag_pipeline is not None else None
        result = search_temporal_records(queries, records, user_text, embedder, self.top_k)
        returned_ids = [
            match["record_id"]
            for group in result["queries"]
            for match in group["matches"]
        ]
        elapsed_ms = int((perf_counter() - started) * 1000)
        self._debug(
            session_id,
            "time_search " + json.dumps({
                "status": "success",
                "reference_time": reference_time,
                "time_queries": [query.as_dict() for query in queries],
                "candidate_count": len(records),
                "returned_memory_ids": returned_ids,
                "ranking_model": DEFAULT_RAG_EMBEDDING_MODEL,
                "ranking_degraded": result["ranking_degraded"],
                "duration_ms": elapsed_ms,
            }, ensure_ascii=False, separators=(",", ":")),
        )
        return {"ok": True, "type": "time_context", "reference_time": reference_time, **result}
```

Implement `_load_temporal_records()` with these exact mappings:

- `semantic_events.jsonl`: load all rows, require valid `start_at` and `end_at` with `start < end`, set `matched_time_field=start_at/end_at` and `time_meaning=事件实际发生区间`, map `inactive_reason=cancelled` to output status `cancelled`, and retain other lifecycle reasons.
- `archival_memory.jsonl`: require `valid_at`; map a non-empty `invalid_at` to a closed interval and an empty `invalid_at` to `{end_at: None, open_end: True}`. Set `matched_time_field=valid_at/invalid_at` and `time_meaning=事实成立的有效区间`. Map stored `type=episodic` to output `record_type=archival_memory`.
- `event_summaries.jsonl`: require both `time_window_start` and `time_window_end` with `start < end`; set `matched_time_field=time_window_start/time_window_end` and `time_meaning=事件总结覆盖的时间范围`.
- `sessions/*.jsonl`: inspect only user-message rows. Add one interval value for every `metadata.turn.temporal_context.time_queries` item with `matched_time_field=temporal_context.time_queries`, and add the message `created_at` as an independent point with `matched_time_field=created_at`.
- Do not substitute `day`, `semantic_day`, `valid_at`, or parsed text when a semantic interval is missing. `created_at` remains a separately labelled message-creation point and never masquerades as semantic event time.
- Format semantic-event text from `description`, archival/event-summary text from `summary`, and conversation text from `content`.

- [ ] **Step 4: Run manager and pure temporal tests**

Run: `python -m unittest test.test_temporal_retrieval test.test_memory_manager -v`

Expected: all tests PASS, including validation, lifecycle, session semantic time, `created_at` matching, field annotations, and global ranking.

- [ ] **Step 5: Commit MemoryManager temporal loading**

```powershell
git add guga/memory/manager.py test/test_memory_manager.py
git commit -m "feat(memory): 接入多层时间记忆召回"
```

### Task 3: Register the time tool and persist complete turn traces

**Files:**
- Create: `guga/chat/turn_trace.py`
- Modify: `guga/tools.py`
- Modify: `guga/chat/session.py`
- Modify: `guga/memory/manager.py`
- Modify: `test/test_tool_calling.py`
- Modify: `test/test_chat_session_rag_flow.py`

**Interfaces:**
- Consumes: `MemoryManager.search_time_context` from Task 2.
- Produces: `memory_time_search_tool(handler: ToolHandler) -> ToolSpec`.
- Produces: `TurnTrace.record_tool(call, result)` and `TurnTrace.as_metadata(assistant_message_id)`.
- Produces: `_SessionStore.merge_message_metadata(session_id: str, message_id: str, metadata: dict) -> None` and `MemoryManager.attach_turn_metadata(*, session_id: str, user_message_id: str, metadata: dict) -> None`.

- [ ] **Step 1: Write failing tests for mixed tools and persisted turn metadata**

Add a fake tool-capable model whose first response returns both a time call and a normal test-tool call, then returns final text:

```python
    def test_time_tool_and_other_tool_share_agent_loop_and_persist_turn_trace(self) -> None:
        class MixedToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call_time", name="memory_time_search", arguments={
                                "time_queries": [{
                                    "original_expression": "8月20号",
                                    "start_at": "2026-08-20T00:00:00+08:00",
                                    "end_at": "2026-08-21T00:00:00+08:00",
                                    "relation": "overlap",
                                    "granularity": "day",
                                }]
                            }),
                            ToolCall(id="call_other", name="guga_test_tool", arguments={"query": "面试"}),
                        ],
                    )
                return ToolModelResponse(content="8月20号有面试安排。", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            manager.semantic_event_file.write_text(json.dumps({
                "id": "evt_interview",
                "type": "semantic_event",
                "description": "参加面试",
                "start_at": "2026-08-20T14:00:00+08:00",
                "end_at": "2026-08-20T15:00:00+08:00",
                "status": "active",
                "updated_at": "2026-08-19T09:00:00+08:00",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            registry = ToolRegistry([ToolSpec(
                name="guga_test_tool",
                description="test",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=lambda args: {"result": f"handled {args['query']}"},
            )])
            session = ChatSession(
                model=MixedToolModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_trace",
                tool_registry=registry,
            )
            answer = session.reply("8月20号有什么面试安排", finalize_memory=False, created_at="2026-08-19T10:00:00+08:00")
            self.assertEqual(answer, "8月20号有面试安排。")
            rows = [json.loads(line) for line in (Path(tmp) / "sessions" / "sess_trace.jsonl").read_text(encoding="utf-8").splitlines()]
            user = next(row for row in rows if row["role"] == "user")
            turn = user["metadata"]["turn"]
            self.assertEqual(turn["assistant_message_id"], next(row["id"] for row in rows if row["role"] == "assistant"))
            self.assertEqual(turn["temporal_context"]["time_queries"][0]["start_at"], "2026-08-20T00:00:00+08:00")
            self.assertEqual([item["tool"] for item in turn["tool_interactions"]], ["memory_time_search", "guga_test_tool"])
```

Add this second test proving a non-time tool still continues the Agent Loop:

```python
    def test_non_time_tool_continues_loop_without_temporal_context(self) -> None:
        class OtherToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_other", name="guga_test_tool", arguments={"query": "项目"})],
                    )
                return ToolModelResponse(content="工具处理完成。", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            model = OtherToolModel()
            registry = ToolRegistry([ToolSpec(
                name="guga_test_tool",
                description="test",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=lambda args: {"result": args["query"]},
            )])
            session = ChatSession(model=model, system_prompt="test", generation=GenerationConfig(), memory_manager=manager, session_id="sess_other", tool_registry=registry)
            self.assertEqual(session.reply("读取项目", finalize_memory=False), "工具处理完成。")
            user = json.loads((Path(tmp) / "sessions" / "sess_other.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(model.calls, 2)
            self.assertNotIn("temporal_context", user["metadata"]["turn"])
```

- [ ] **Step 2: Run the two new integration tests and verify the time tool is unknown**

Run: `python -m unittest test.test_tool_calling.ToolCallingTest.test_time_tool_and_other_tool_share_agent_loop_and_persist_turn_trace -v`

Expected: FAIL because `memory_time_search` is not registered and no `metadata.turn` is persisted.

- [ ] **Step 3: Implement `TurnTrace` without storing full tool-result bodies**

Create `guga/chat/turn_trace.py` with this public shape:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guga.tools import ToolCall


@dataclass
class TurnTrace:
    turn_id: str
    user_message_id: str
    reference_time: str
    time_queries: list[dict[str, Any]] = field(default_factory=list)
    tool_interactions: list[dict[str, Any]] = field(default_factory=list)

    def record_tool(self, call: ToolCall, result: dict[str, Any]) -> None:
        returned_ids = [
            str(match.get("record_id", ""))
            for group in (result.get("queries", []) or [])
            if isinstance(group, dict)
            for match in (group.get("matches", []) or [])
            if isinstance(match, dict) and str(match.get("record_id", ""))
        ]
        self.tool_interactions.append({
            "tool_call_id": call.id,
            "tool": call.name,
            "arguments": dict(call.arguments),
            "ok": bool(result.get("ok")),
            "returned_memory_ids": returned_ids,
            "error": dict(result.get("error", {})) if isinstance(result.get("error"), dict) else str(result.get("error", "")),
        })
        if call.name == "memory_time_search" and result.get("ok") is True:
            self.time_queries.extend(
                dict(group["query"])
                for group in (result.get("queries", []) or [])
                if isinstance(group, dict) and isinstance(group.get("query"), dict)
            )

    def as_metadata(self, assistant_message_id: str) -> dict[str, Any]:
        turn = {
            "turn_id": self.turn_id,
            "assistant_message_id": assistant_message_id,
            "tool_interactions": list(self.tool_interactions),
        }
        if self.time_queries:
            turn["temporal_context"] = {
                "source_message_id": self.user_message_id,
                "reference_time": self.reference_time,
                "time_queries": list(self.time_queries),
            }
        return {"turn": turn}
```

- [ ] **Step 4: Add the tool schema and remove the regex parsing tool from the default registry**

In `guga/tools.py`, remove `_time_parse_tool`, its time-utils imports, and `registry.add(_time_parse_tool())`. Add:

```python
def memory_time_search_tool(handler: ToolHandler) -> ToolSpec:
    query_properties = {
        "original_expression": {"type": "string"},
        "start_at": {"type": "string", "description": "Timezone-aware ISO 8601 inclusive start."},
        "end_at": {"type": "string", "description": "Timezone-aware ISO 8601 exclusive end."},
        "relation": {"type": "string", "enum": ["overlap", "before", "after"]},
        "granularity": {"type": "string", "enum": ["minute", "hour", "day", "week", "month", "year", "range", "session"]},
    }
    return ToolSpec(
        name="memory_time_search",
        description="Search memory by one or more absolute time intervals resolved from the user's expression and current context.",
        parameters={
            "type": "object",
            "properties": {
                "time_queries": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": query_properties,
                        "required": ["original_expression", "start_at", "end_at", "relation", "granularity"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["time_queries"],
            "additionalProperties": False,
        },
        handler=handler,
    )
```

- [ ] **Step 5: Persist turn metadata atomically in the existing session JSONL**

Add `_SessionStore.merge_message_metadata` in `guga/memory/manager.py`. It must rewrite the same user row rather than append a third row:

```python
    def merge_message_metadata(self, session_id: str, message_id: str, metadata: dict) -> None:
        target = self.session_dir / f"{session_id}.jsonl"
        rows = []
        found = False
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("id", "")) == message_id:
                row["metadata"] = {**dict(row.get("metadata", {}) or {}), **metadata}
                found = True
            rows.append(row)
        if not found:
            raise ValueError(f"session message not found: {message_id}")
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        temporary.replace(target)
```

Expose `MemoryManager.attach_turn_metadata(session_id, user_message_id, metadata)`. Call the session-store method and also copy `turn_id`, `temporal_context`, and `tool_interactions` into the in-memory `_turn_state[session_id]` so asynchronous consolidation receives the same trace.

Use this method body and widen `_turn_state` values from `dict[str, str]` to `dict[str, Any]`:

```python
    def attach_turn_metadata(self, *, session_id: str, user_message_id: str, metadata: dict) -> None:
        self.session_store.merge_message_metadata(session_id, user_message_id, metadata)
        turn = dict(metadata.get("turn", {}) or {})
        with self._turn_state_lock:
            state = self._turn_state.setdefault(session_id, {})
            state["turn_id"] = str(turn.get("turn_id", f"turn_{user_message_id}"))
            state["temporal_context"] = dict(turn.get("temporal_context", {}) or {})
            state["tool_interactions"] = list(turn.get("tool_interactions", []) or [])
```

- [ ] **Step 6: Wire the time tool, one reference time, and trace recording into both reply modes**

In `ChatSession.__init__`, register the tool even when a caller supplied a custom registry:

```python
        self.tool_registry = tool_registry or default_tool_registry()
        self._active_turn_user_text = ""
        self._active_turn_reference_time = ""
        self.tool_registry.add(memory_time_search_tool(self._execute_memory_time_search))
```

Add the handler:

```python
    def _execute_memory_time_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._active_turn_user_text or not self._active_turn_reference_time:
            return {"ok": False, "type": "time_context", "error": {"code": "NO_ACTIVE_TURN", "message": "time search requires an active turn", "query_index": -1}}
        return self.memory_manager.search_time_context(
            session_id=self.session_id,
            user_text=self._active_turn_user_text,
            reference_time=self._active_turn_reference_time,
            time_queries=args.get("time_queries"),
        )
```

Add `created_at: str | None = None` to `reply_stream`, then apply the following to `reply` and `reply_stream`:

1. Compute `reference_time = created_at or now_beijing_iso()` once.
2. Save the user message using that exact timestamp and capture `user_message_id`.
3. Create `TurnTrace(turn_id=f"turn_{user_message_id}", user_message_id=user_message_id, reference_time=reference_time)`.
4. Set the two active-turn fields before model generation.
5. Pass the trace into `_generate_reply_with_optional_tools` or `_generate_reply_with_optional_tools_stream`.
6. Immediately after every `tool_registry.execute(call)`, call `trace.record_tool(call, result)` for all tools, not only the time tool.
7. Save the assistant message, capture its ID, then call `self.memory_manager.attach_turn_metadata(session_id=self.session_id, user_message_id=user_message_id, metadata=trace.as_metadata(assistant_message_id))` before `finalize_turn_async`.
8. Clear the two active-turn fields in a `finally` block around generation.

Change `_tool_system_prompt` to accept `reference_time` and append these rules:

```text
[Current Time]
reference_time=<exact turn timestamp>
timezone=Asia/Hong_Kong (+08:00)

[Temporal Memory Tool]
When the user states, changes, cancels, compares, or asks about time-bound information, resolve every relevant expression to an absolute timezone-aware left-closed/right-open interval and call memory_time_search.
Use one normalized interval for all relevant time fields. Interpret the result labels as follows:
- start_at/end_at: when an event occurs.
- valid_at/invalid_at: when a fact is valid.
- time_window_start/time_window_end: the period covered by an event summary.
- temporal_context.time_queries: time mentioned or resolved in that conversation turn.
- created_at: when the message was created, not when an event described by the message occurred.
Use matched_time_field and time_meaning together with the user's question to decide which evidence is relevant.
If a time reference cannot be resolved from the current time and conversation context, ask the user to clarify instead of inventing an interval.
The time tool is one normal tool among all available tools. Continue the Agent Loop after any tool call, and finish only when no tool_calls remain.
```

- [ ] **Step 7: Run tool-loop, session, and streaming tests**

Run: `python -m unittest test.test_tool_calling test.test_chat_session_rag_flow test.test_chat_history -v`

Expected: all tests PASS; existing non-time tools still continue the loop, mixed tool calls both execute, streaming traces persist, and session files retain exactly two message rows per completed turn.

- [ ] **Step 8: Commit tool and turn integration**

```powershell
git add guga/chat/turn_trace.py guga/tools.py guga/chat/session.py guga/memory/manager.py test/test_tool_calling.py test/test_chat_session_rag_flow.py
git commit -m "feat(chat): 接入时间工具与轮次追踪"
```

### Task 4: Remove the legacy regex date route from ordinary semantic retrieval

**Files:**
- Modify: `guga/memory/manager.py`
- Modify: `test/test_memory_time_utils.py`
- Modify: `test/test_memory_manager.py`

**Interfaces:**
- Consumes: the tool-driven temporal path completed in Tasks 1–3.
- Produces: ordinary `prepare_context` behavior that is semantic/profile-only and never structurally filters by a regex-resolved calendar day.

- [ ] **Step 1: Replace the old date-boost assertion with separation tests**

Remove `test_date_query_uses_semantic_day_for_retrieval_boost` and add:

```python
    def test_ordinary_query_plan_does_not_route_explicit_date_to_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            plan = manager._build_query_plan("8月20号有什么事情", "sess_plain", [])
            self.assertEqual(plan.route, "hybrid")
            self.assertEqual(plan.reason, "default_hybrid")

    def test_default_tool_registry_no_longer_exposes_regex_time_parser(self) -> None:
        names = [item["function"]["name"] for item in default_tool_registry().openai_tools()]
        self.assertNotIn("guga_parse_time", names)
```

Add this manager assertion that explicit dates no longer create a structural date filter:

```python
    def test_explicit_date_keeps_ordinary_retrieval_in_hybrid_route(self) -> None:
        plan = self.manager._build_query_plan("8月20号有什么事情", "sess_plain", [])
        records = [
            {"id": "evt_20", "summary": "8月20日面试", "type": "semantic_event"},
            {"id": "evt_21", "summary": "8月21日聚餐", "type": "semantic_event"},
        ]
        self.assertEqual(plan.route, "hybrid")
        self.assertEqual([record["id"] for record in records], ["evt_20", "evt_21"])
```

- [ ] **Step 2: Run the replacement tests and verify the old date route still activates**

Run: `python -m unittest test.test_memory_time_utils test.test_memory_manager -v`

Expected: FAIL because `_build_query_plan` still returns `date_window` for the explicit date query.

- [ ] **Step 3: Remove temporal filtering and boosting from the ordinary retrieval branch**

Apply these exact structural changes in `guga/memory/manager.py`:

- Reduce `_QueryPlan` to `route`, `reason`, and no temporal/day/session fields.
- Remove `_date_context_by_session`.
- Make `_build_query_plan` return `portrait` only for portrait queries and `hybrid` for every other query.
- Stop loading raw session rows in `prepare_context`; time-matched session retrieval now belongs to `memory_time_search`, which searches both turn semantic time and labelled message `created_at`.
- Remove `_records_for_query_plan`, `_record_matches_day`, `_extract_query_day_with_source`, `_mentions_recent_current`, `_mentions_last_session`, `_latest_non_current_session_id`, `_apply_time_score_adjustments`, `_apply_time_score_components`, `_record_day`, and `_dedupe_semantic_event_overlaps`.
- Remove `time_hints` from `_merge_memory_hits` and its callers. Preserve semantic score, current-turn weakening, retention, layer quotas, and source validity unchanged.
- Remove direct `extract_semantic_time`/`now_beijing` imports from `manager.py` when they become unused. Keep ingestion helpers in `time_utils.py` because consolidation migration is handled in the second implementation plan.
- Keep user-portrait routing unchanged.

The resulting query plan must be exactly:

```python
    def _build_query_plan(self, query: str, session_id: str, records: list[dict]) -> _QueryPlan:
        if self._mentions_user_portrait(query.strip()):
            return _QueryPlan(route="portrait", reason="long_term_user_profile")
        return _QueryPlan(route="hybrid", reason="default_hybrid")
```

- [ ] **Step 4: Run retrieval and Agent Loop regression tests**

Run: `python -m unittest test.test_memory_time_utils test.test_memory_manager test.test_chat_session_rag_flow test.test_tool_calling test.test_rag_pipeline -v`

Expected: all tests PASS. No ordinary semantic retrieval test should depend on regex date parsing or temporal score boosts.

- [ ] **Step 5: Run the complete non-live suite**

Run: `python -m unittest discover -s test -p "test_*.py" -v`

Expected: all non-live tests PASS; tests guarded by `GUGA_RUN_LIVE_API_TESTS` remain skipped.

- [ ] **Step 6: Commit legacy-route removal**

```powershell
git add guga/memory/manager.py test/test_memory_time_utils.py test/test_memory_manager.py
git commit -m "refactor(memory): 分离语义与时间检索"
```

- [ ] **Step 7: Push the completed retrieval plan commits**

Run: `git status --short`

Expected: no unintended tracked changes.

Run: `git push`

Expected: the current branch is pushed successfully.
