from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.manager import MemoryManager


class TimelineFactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_root = Path(self.tmp.name)
        self.manager = MemoryManager(memory_root=self.memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def test_finalize_turn_writes_timeline_fact_for_time_bound_plan(self) -> None:
        session_id = "sess_fact_write"
        self.manager.record_user_message(session_id, "我在2026年7月3日要提交项目报告，请你记住。")
        self.manager.record_assistant_message(session_id, "我记住了。")

        self.manager.finalize_turn(session_id)

        fact_file = self.memory_root / "timeline_facts.jsonl"
        self.assertTrue(fact_file.exists())
        rows = [json.loads(line) for line in fact_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        fact = rows[0]
        self.assertEqual(fact["type"], "timeline_fact")
        self.assertEqual(fact["subject"], "user")
        self.assertEqual(fact["predicate"], "has_time_bound_plan")
        self.assertEqual(fact["semantic_day"], "2026-07-03")
        self.assertEqual(fact["valid_from"], "2026-07-03T00:00:00+08:00")
        self.assertEqual(fact["valid_at"], "2026-07-03T00:00:00+08:00")
        self.assertTrue(fact["created_at"].endswith("+08:00"))
        self.assertEqual(fact["source_session_id"], session_id)
        self.assertTrue(fact["source_message_ids"])
        self.assertIn("提交项目报告", fact["semantic_text"])

    def test_finalize_turn_does_not_write_fact_for_casual_today_chat(self) -> None:
        session_id = "sess_no_fact"
        self.manager.record_user_message(session_id, "你好，今天随便聊聊。")
        self.manager.record_assistant_message(session_id, "可以。")

        self.manager.finalize_turn(session_id)

        self.assertFalse((self.memory_root / "timeline_facts.jsonl").exists())

    def test_date_query_retrieves_timeline_fact_and_dedupes_same_source_summary(self) -> None:
        source_message_id = "msg_plan"
        self._append_jsonl(
            self.memory_root / "timeline_facts.jsonl",
            {
                "fact_id": "fact_submit_report",
                "id": "fact_submit_report",
                "type": "timeline_fact",
                "subject": "user",
                "predicate": "has_time_bound_plan",
                "object": "提交项目报告",
                "summary": "用户在2026-07-03有时间相关安排：提交项目报告。",
                "semantic_text": "用户在2026-07-03有时间相关安排：提交项目报告。",
                "raw_excerpt": "我在2026年7月3日要提交项目报告",
                "created_at": "2026-06-27T12:00:00+08:00",
                "updated_at": "2026-06-27T12:00:00+08:00",
                "valid_from": "2026-07-03T00:00:00+08:00",
                "valid_to": "",
                "valid_at": "2026-07-03T00:00:00+08:00",
                "invalid_at": "",
                "semantic_day": "2026-07-03",
                "day": "2026-07-03",
                "time_source": "semantic_explicit_date",
                "time_granularity": "day",
                "source_session_id": "sess_history",
                "source_message_ids": [source_message_id],
                "confidence": 0.9,
                "importance": 0.8,
                "memory_strength": 1,
                "retention": 1.0,
                "status": "active",
                "extraction_version": "test",
            },
        )
        self._append_jsonl(
            self.memory_root / "event_summaries.jsonl",
            {
                "id": "evt_daily_20260703",
                "type": "event_summary",
                "summary": "2026-07-03 的对话摘要：用户提到要提交项目报告。",
                "raw_excerpt": "用户提到要提交项目报告。",
                "created_at": "2026-06-27T12:01:00+08:00",
                "updated_at": "2026-06-27T12:01:00+08:00",
                "day": "2026-07-03",
                "valid_at": "2026-07-03T00:00:00+08:00",
                "semantic_day": "2026-07-03",
                "time_source": "semantic_explicit_date",
                "source_session_id": "sess_history",
                "source_message_ids": [source_message_id],
                "confidence": 0.8,
                "importance": 0.7,
                "memory_strength": 1,
                "retention": 1.0,
                "status": "active",
            },
        )

        context = self.manager.prepare_context("2026年7月3日我要做什么？", session_id="sess_now")

        self.assertGreaterEqual(len(context.hits), 1)
        self.assertEqual(context.hits[0].memory_type, "timeline_fact")
        self.assertEqual(context.hits[0].id, "fact_submit_report")
        self.assertEqual(context.event_summaries, [])
        prompt = self.manager.compose_system_prompt("你是一个助手", context)
        self.assertIn("fact_submit_report", prompt)
        self.assertNotIn("evt_daily_20260703", prompt)


if __name__ == "__main__":
    unittest.main()
