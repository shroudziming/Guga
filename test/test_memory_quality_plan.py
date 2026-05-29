from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.rag.schemas import RetrievalHit


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MemoryQualityPlanTest(unittest.TestCase):
    def test_current_turn_is_weakened_and_not_reinforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_advisor",
                    "type": "episodic",
                    "summary": "用户下周三要和导师见面。",
                    "raw_excerpt": "下周三和导师见面",
                    "created_at": "2099-01-01T00:00:00+00:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "importance": 0.8,
                    "confidence": 0.9,
                    "status": "active",
                },
            )

            session_id = "sess_current"
            manager.record_user_message(session_id, "你记得我下周三要去做什么吗")
            context = manager.prepare_context("你记得我下周三要去做什么吗", session_id=session_id)

            self.assertEqual(context.hits[0].id, "mem_advisor")
            current_hits = [hit for hit in context.hits if hit.is_current_turn]
            self.assertEqual(len(current_hits), 1)
            self.assertLess(current_hits[0].score, context.hits[0].score)

            session_memory = json.loads((memory_root / "session_memories.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(session_memory["memory_strength"], 1)

    def test_low_score_memories_are_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            manager.memory_min_score = 0.3
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_weak",
                    "type": "episodic",
                    "summary": "alpha detail",
                    "raw_excerpt": "alpha detail",
                    "created_at": "2099-01-01T00:00:00+00:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "importance": 0.0,
                    "confidence": 0.0,
                    "status": "active",
                },
            )

            context = manager.prepare_context("alpha beta gamma delta epsilon", session_id="sess_probe")

            self.assertEqual(context.hits, [])

    def test_semantic_and_lexical_hits_are_fused_by_max_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=2, recency_weight=0.0, enable_semantic=False)
            record = {
                "id": "mem_mix",
                "type": "episodic",
                "summary": "lexical beta",
                "raw_excerpt": "lexical beta",
                "created_at": "2099-01-01T00:00:00+00:00",
                "last_recalled_at": "2099-01-01T00:00:00+00:00",
                "memory_strength": 1,
                "retention": 1.0,
                "source_session_id": "sess_old",
                "source_message_ids": ["msg_old"],
                "importance": 0.0,
                "confidence": 0.0,
                "status": "active",
            }
            lexical_hit = manager._to_hit(record, 0.8)
            semantic_hit = RetrievalHit(
                chunk_id="chunk_mem_mix",
                text="semantic beta",
                score=0.2,
                source_type="episodic",
                source_id="mem_mix",
                source_session_id="sess_old",
                source_message_id="msg_old",
                created_at="2099-01-01T00:00:00+00:00",
            )

            hits = manager._merge_memory_hits(
                [semantic_hit],
                [lexical_hit],
                [record],
                current_turn_ids=set(),
                time_hints={},
                session_id="sess_probe",
            )

            self.assertEqual(hits[0].id, "mem_mix")
            self.assertEqual(hits[0].score, 0.8)
            self.assertEqual(hits[0].semantic_score, 0.2)
            self.assertEqual(hits[0].lexical_score, 0.8)
            self.assertEqual(hits[0].score_source, "lexical")

    def test_explicit_date_query_prioritizes_daily_event_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260509",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-09",
                    "summary": "当天讨论了导师安排。",
                    "raw_excerpt": "导师安排",
                    "created_at": "2026-05-09T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_may9",
                    "source_message_ids": ["msg_may9"],
                    "importance": 0.75,
                    "confidence": 0.8,
                    "status": "active",
                },
            )
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_other_day",
                    "type": "episodic",
                    "summary": "那天的对话比较长。",
                    "raw_excerpt": "那天的对话",
                    "created_at": "2026-05-10T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_may10",
                    "source_message_ids": ["msg_may10"],
                    "status": "active",
                },
            )

            context = manager.prepare_context("2026-05-09那天的对话", session_id="sess_probe")

            self.assertEqual(context.hits[0].id, "evt_daily_20260509")

    def test_recent_and_last_session_time_references_are_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260508",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-08",
                    "summary": "旧会话内容。",
                    "created_at": "2026-05-08T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "status": "active",
                },
            )
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260510",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-10",
                    "summary": "最近一次历史会话内容。",
                    "created_at": "2026-05-10T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "source_session_id": "sess_latest",
                    "source_message_ids": ["msg_latest"],
                    "status": "active",
                },
            )
            last_context = manager.prepare_context("上次我们聊了什么", session_id="sess_now")
            self.assertEqual(last_context.hits[0].id, "evt_daily_20260510")

            manager.record_user_message("sess_now", "我们先聊蝴蝶刀")
            manager.record_assistant_message("sess_now", "刚才你说什么：蝴蝶刀练习安排")
            manager.record_user_message("sess_now", "刚才你说什么")
            recent_context = manager.prepare_context("刚才你说什么", session_id="sess_now")

            self.assertTrue(recent_context.hits[0].summary.startswith("assistant:"))

    def test_portrait_summary_ignores_one_off_bug_feedback(self) -> None:
        summarizer = MemoryBankSummarizer()

        daily = summarizer.summarize_daily_personality("user: 你刚才没有输出，有 bug")
        global_portrait = summarizer.summarize_global_portrait(
            [
                daily,
                "- temporary_state: 用户近期可能存在压力或情绪波动。",
                "- stable_interest: 用户想练蝴蝶刀。",
            ]
        )

        self.assertEqual(daily, "")
        self.assertNotIn("bug", global_portrait.lower())
        self.assertNotIn("temporary", global_portrait.lower())
        self.assertIn("蝴蝶刀", global_portrait)


if __name__ == "__main__":
    unittest.main()
