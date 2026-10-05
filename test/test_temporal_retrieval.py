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


class MalformedEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


class TemporalRetrievalTest(unittest.TestCase):
    def test_timezone_overflow_returns_indexed_query_validation_error(self) -> None:
        valid = {"start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00"}
        for field, extreme in (("start_at", "0001-01-01T00:00:00+14:00"),
                               ("end_at", "9999-12-31T23:59:59-12:00")):
            with self.subTest(field=field):
                with self.assertRaises(TemporalQueryValidationError) as caught:
                    parse_time_queries([valid, {**valid, field: extreme}])
                self.assertEqual(caught.exception.code, "INVALID_ISO_TIME")
                self.assertEqual(caught.exception.query_index, 1)

    def test_normalizes_query_timezone_without_losing_subsecond_interval(self) -> None:
        queries = parse_time_queries([{
            "original_expression": "瞬间",
            "start_at": "2026-08-20T00:00:00.100000Z",
            "end_at": "2026-08-20T00:00:00.200000Z",
            "relation": "overlap",
            "granularity": "minute",
        }])

        canonical = queries[0].as_dict()

        self.assertEqual(canonical["start_at"], "2026-08-20T08:00:00.100000+08:00")
        self.assertEqual(canonical["end_at"], "2026-08-20T08:00:00.200000+08:00")

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

    def test_malformed_embedding_output_degrades_to_recency_ranking(self) -> None:
        query = parse_time_queries([{"original_expression": "8月20号", "start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00", "relation": "overlap", "granularity": "day"}])
        records = [
            {"record_type": "semantic_event", "record_id": "older", "text": "较早", "status": "active", "time_values": [{"start_at": "2026-08-20T09:00:00+08:00", "end_at": "2026-08-20T10:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}], "updated_at": "2026-08-18T10:00:00+08:00", "time_fields": {}},
            {"record_type": "semantic_event", "record_id": "newer", "text": "较晚", "status": "active", "time_values": [{"start_at": "2026-08-20T11:00:00+08:00", "end_at": "2026-08-20T12:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}], "updated_at": "2026-08-19T10:00:00+08:00", "time_fields": {}},
        ]

        result = search_temporal_records(query, records, "安排", MalformedEmbedder(), 5)

        self.assertTrue(result["ranking_degraded"])
        self.assertEqual([item["record_id"] for item in result["queries"][0]["matches"]], ["newer", "older"])
        self.assertIsNone(result["queries"][0]["matches"][0]["semantic_score"])

    def test_each_matching_time_value_competes_in_one_global_top_k(self) -> None:
        query = parse_time_queries([{"original_expression": "8月20号", "start_at": "2026-08-20T00:00:00+08:00", "end_at": "2026-08-21T00:00:00+08:00", "relation": "overlap", "granularity": "day"}])
        records = [
            {
                "record_type": "conversation_turn",
                "record_id": "turn_multi",
                "text": "普通安排",
                "status": "active",
                "time_values": [
                    {"start_at": "2026-08-20T09:00:00+08:00", "end_at": None, "is_point": True, "open_end": False, "matched_time_field": "created_at", "time_meaning": "消息创建时间"},
                    {"start_at": "2026-08-20T10:00:00+08:00", "end_at": None, "is_point": True, "open_end": False, "matched_time_field": "updated_at", "time_meaning": "消息更新时间"},
                ],
                "updated_at": "2026-08-20T10:00:00+08:00",
                "time_fields": {"created_at": "2026-08-20T09:00:00+08:00", "updated_at": "2026-08-20T10:00:00+08:00"},
            },
            {
                "record_type": "semantic_event",
                "record_id": "evt_interview",
                "text": "参加面试",
                "status": "active",
                "time_values": [{"start_at": "2026-08-20T14:00:00+08:00", "end_at": "2026-08-20T15:00:00+08:00", "is_point": False, "open_end": False, "matched_time_field": "start_at/end_at", "time_meaning": "事件实际发生区间"}],
                "updated_at": "2026-08-19T10:00:00+08:00",
                "time_fields": {},
            },
        ]

        result = search_temporal_records(query, records, "面试安排", KeywordEmbedder(), 3)

        matches = result["queries"][0]["matches"]
        self.assertEqual([item["record_id"] for item in matches], ["evt_interview", "turn_multi", "turn_multi"])
        self.assertEqual([item["matched_time_field"] for item in matches[1:]], ["created_at", "updated_at"])
        self.assertEqual([item["semantic_score"] for item in matches], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
