from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.memory.time_utils import extract_semantic_time
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
