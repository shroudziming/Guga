from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.event_summary_store import EventSummaryStore
from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.memory.time_utils import extract_semantic_time
from guga.rag.schemas import RetrievalHit


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

    def test_date_query_uses_semantic_day_for_retrieval_boost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            archival = memory_root / "archival_memory.jsonl"
            archival.write_text(
                json.dumps(
                    {
                        "id": "mem_meeting",
                        "type": "episodic",
                        "summary": "用户计划在2026年6月20日与导师见面",
                        "raw_excerpt": "真实测试：我在2026年6月20日要和导师见面，请你记住。",
                        "created_at": "2026-06-12T17:09:37+08:00",
                        "last_recalled_at": "2099-01-01T00:00:00+08:00",
                        "valid_at": "2026-06-20T00:00:00+08:00",
                        "semantic_day": "2026-06-20",
                        "time_source": "semantic_explicit_date",
                        "memory_strength": 1,
                        "source_session_id": "sess_time",
                        "source_message_ids": ["msg_time"],
                        "importance": 0.7,
                        "confidence": 1.0,
                        "status": "active",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manager = MemoryManager(memory_root=memory_root, top_k=4, enable_semantic=False)
            semantic_hit = RetrievalHit(
                chunk_id="chunk_mem_meeting",
                text="用户将在2026年6月20日与导师见面。",
                score=0.8,
                source_type="memory",
                source_id="mem_meeting",
                source_session_id="sess_time",
                source_message_id="msg_time",
                created_at="2026-06-12T17:09:37+08:00",
            )
            manager._retrieve_semantic = lambda **_: ([semantic_hit], [])

            context = manager.prepare_context("你记得我2026年6月20日要做什么吗？", session_id="sess_probe")

            self.assertEqual(context.hits[0].id, "mem_meeting")
            self.assertEqual(context.hits[0].valid_at, "2026-06-20T00:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
