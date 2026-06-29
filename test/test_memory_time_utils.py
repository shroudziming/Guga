from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.event_summary_store import EventSummaryStore
from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.memory.time_utils import extract_semantic_time


class SummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "Extract one long-term memory candidate" in prompt:
            return (
                '{"should_archive": true, "topic": "schedule", '
                '"summary": "用户在2026年6月20日要和导师见面", '
                '"importance": 0.8, "confidence": 0.9}'
            )
        if "用户画像候选提取器" in prompt:
            return ""
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

    def test_archival_memory_keeps_transaction_time_and_semantic_valid_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, model=SummaryModel(), enable_semantic=False)

            manager.record_user_message("sess_time", "真实测试：我在2026年6月20日要和导师见面，请你记住。")
            manager.record_assistant_message("sess_time", "记住了")
            manager.finalize_turn("sess_time")

            payload = json.loads((memory_root / "archival_memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(payload["created_at"].endswith("+08:00"))
            self.assertEqual(payload["valid_at"], "2026-06-20T00:00:00+08:00")
            self.assertEqual(payload["semantic_day"], "2026-06-20")
            self.assertEqual(payload["time_source"], "semantic_explicit_date")

    def test_event_summary_preserves_conversation_day_and_semantic_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EventSummaryStore(Path(tmp) / "event_summaries.jsonl")
            payload = store.refresh_daily_summary(
                session_id="sess_time",
                day="2026-06-12",
                dialogue="user: 我在2026年6月20日要和导师见面",
                source_message_ids=["msg_time"],
                summarizer=MemoryBankSummarizer(model=SummaryModel()),
            )

            self.assertEqual(payload["day"], "2026-06-12")
            self.assertEqual(payload["valid_at"], "2026-06-20T00:00:00+08:00")
            self.assertEqual(payload["semantic_day"], "2026-06-20")

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
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)

            context = manager.prepare_context("你记得我2026年6月20日要做什么吗？", session_id="sess_probe")

            self.assertEqual(context.hits[0].id, "mem_meeting")
            self.assertEqual(context.hits[0].valid_at, "2026-06-20T00:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
