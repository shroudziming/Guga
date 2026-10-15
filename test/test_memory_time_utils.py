from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.event_summary_store import EventSummaryStore
from guga.memory.manager import MemoryManager
from guga.memory.semantic_events import EventApplyResult
from guga.memory.summarizer import MemoryBankSummarizer
from guga.memory.time_utils import extract_semantic_time, parse_datetime
from guga.tools import default_tool_registry


class SummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "Low-level memory consolidation" in prompt:
            return json.dumps(
                {
                    "timeline_facts": [
                        {
                            "action": "upsert",
                            "subject": "user",
                            "predicate": "has_time_bound_plan",
                            "object": "和导师见面",
                            "summary": "用户在2026年6月20日要和导师见面",
                            "semantic_day": "2026-06-20",
                            "confidence": 0.9,
                            "source_message_ids": [],
                        }
                    ],
                    "event_summaries": [
                        {
                            "action": "upsert",
                            "scope": "batch",
                            "summary": "用户在2026年6月20日要和导师见面",
                            "source_message_ids": [],
                            "confidence": 0.9,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "High-level memory consolidation" in prompt:
            return json.dumps(
                {
                    "decision": "update_high_level_memory",
                    "archival_updates": [
                        {
                            "topic": "schedule",
                            "summary": "用户在2026年6月20日要和导师见面",
                            "importance": 0.8,
                            "confidence": 0.9,
                            "source_message_ids": [],
                        }
                    ],
                    "profile_updates": [],
                    "personality_insight_updates": [],
                    "reason": "time-bound plan",
                },
                ensure_ascii=False,
            )
        if "Memory route classifier" in prompt:
            return json.dumps(
                [
                    {
                        "target": "archival_memory",
                        "label": "time_bound_plan",
                        "content": "用户在2026年6月20日要和导师见面",
                        "topic": "schedule",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                    {
                        "target": "timeline_fact",
                        "label": "time_bound_plan",
                        "content": "用户在2026年6月20日要和导师见面",
                        "confidence": 0.9,
                    },
                ],
                ensure_ascii=False,
            )
        if "用户画像整理器" in prompt:
            return ""
        return "- 用户在2026年6月20日要和导师见面。"


class MemoryTimeUtilsTest(unittest.TestCase):
    def test_event_summary_uses_covered_event_exclusive_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventSummaryStore(Path(tmp_dir) / "event_summaries.jsonl")
            result = EventApplyResult(["evt_1", "evt_2"], [], [])
            summary = store.upsert_batch_summary(
                session_id="sess_summary",
                batch_seq=1,
                payload={"summary": "两项安排", "confidence": 0.9},
                source_message_ids=["msg_1"],
                event_result=result,
                covered_events=[
                    {
                        "id": "evt_1",
                        "start_at": "2026-08-20T14:00:00+08:00",
                        "end_at": "2026-08-20T15:00:00+08:00",
                    },
                    {
                        "id": "evt_2",
                        "start_at": "2026-08-21T09:00:00+08:00",
                        "end_at": "2026-08-21T10:00:00+08:00",
                    },
                ],
            )
            self.assertEqual(summary["time_window_start"], "2026-08-20T14:00:00+08:00")
            self.assertEqual(summary["time_window_end"], "2026-08-21T10:00:00+08:00")

    def test_event_summary_rejects_missing_covered_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventSummaryStore(Path(tmp_dir) / "event_summaries.jsonl")
            with self.assertRaisesRegex(ValueError, "covered canonical event intervals"):
                store.upsert_batch_summary(
                    session_id="sess_summary",
                    batch_seq=1,
                    payload={"summary": "无来源总结"},
                    source_message_ids=["msg_1"],
                    event_result=EventApplyResult([], [], []),
                    covered_events=[],
                )

    def test_parse_datetime_returns_none_when_timezone_normalization_overflows(self) -> None:
        for raw in ("0001-01-01T00:00:00+14:00", "9999-12-31T23:59:59-12:00"):
            for value in (raw, datetime.fromisoformat(raw)):
                with self.subTest(value=value):
                    self.assertIsNone(parse_datetime(value))

    def test_extract_semantic_time_uses_reference_time_for_relative_weekday(self) -> None:
        result = extract_semantic_time("我下周三要和导师见面", reference_time="2026-06-12T17:09:37+08:00")

        self.assertIsNotNone(result)
        valid_at, source, granularity = result
        self.assertEqual(valid_at.isoformat(timespec="seconds"), "2026-06-17T00:00:00+08:00")
        self.assertEqual(source, "semantic_relative_weekday")
        self.assertEqual(granularity, "date")

    def test_ordinary_query_plan_does_not_route_explicit_date_to_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)

            plan = manager._build_query_plan("8月20号有什么事情", "sess_plain", [])

            self.assertEqual(plan.route, "hybrid")
            self.assertEqual(plan.reason, "default_hybrid")

    def test_default_tool_registry_no_longer_exposes_regex_time_parser(self) -> None:
        names = [item["function"]["name"] for item in default_tool_registry().openai_tools()]

        self.assertNotIn("guga_parse_time", names)


if __name__ == "__main__":
    unittest.main()
