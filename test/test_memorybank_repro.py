from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from guga.memory.forgetting import retention_score
from guga.memory.manager import MemoryManager


class MemoryBankReproTest(unittest.TestCase):
    def test_finalize_turn_uses_llm_archival_extraction_when_model_available(self) -> None:
        class FakeModel:
            def generate_reply(self, messages, gen):
                prompt = messages[-1]["content"]
                if "Extract one long-term memory candidate" in prompt:
                    return json.dumps(
                        {
                            "should_archive": True,
                            "topic": "work",
                            "summary": "The user works as a backend engineer in Hangzhou.",
                            "importance": 0.9,
                            "confidence": 0.8,
                        }
                    )
                return "- LLM generated summary"

        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, model=FakeModel(), enable_semantic=False)
            manager.record_user_message("sess_llm", "I work as a backend engineer in Hangzhou.")
            manager.record_assistant_message("sess_llm", "I will remember that.")
            manager.finalize_turn("sess_llm")

            archival_payload = json.loads((memory_root / "archival_memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(archival_payload["topic"], "work")
            self.assertEqual(archival_payload["summary"], "The user works as a backend engineer in Hangzhou.")
            self.assertEqual(archival_payload["importance"], 0.9)
            self.assertEqual(archival_payload["confidence"], 0.8)

    def test_retention_uses_ebbinghaus_curve(self) -> None:
        record = {
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_recalled_at": "2026-01-01T00:00:00+00:00",
            "memory_strength": 2,
        }

        score = retention_score(record, now=datetime(2026, 1, 3, tzinfo=timezone.utc))

        self.assertAlmostEqual(score, 0.367879, places=5)

    def test_prepare_context_reinforces_recalled_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            archival = memory_root / "archival_memory.jsonl"
            archival.write_text(
                json.dumps(
                    {
                        "id": "mem_work",
                        "type": "episodic",
                        "summary": "用户提到：我在杭州做后端开发",
                        "raw_excerpt": "我在杭州做后端开发",
                        "created_at": "2099-01-01T00:00:00+00:00",
                        "last_recalled_at": "2099-01-01T00:00:00+00:00",
                        "memory_strength": 1,
                        "retention": 1.0,
                        "source_session_id": "sess_seed",
                        "source_message_ids": ["msg_seed"],
                        "status": "active",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False)

            context = manager.prepare_context("你记得我在杭州做什么吗", session_id="sess_probe")

            self.assertEqual(context.hits[0].id, "mem_work")
            updated = json.loads(archival.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(updated["memory_strength"], 2)
            self.assertEqual(updated["retention"], 1.0)
            self.assertTrue(updated["last_recalled_at"])

    def test_non_recall_query_does_not_reinforce_noisy_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            archival = memory_root / "archival_memory.jsonl"
            archival.write_text(
                json.dumps(
                    {
                        "id": "mem_work",
                        "type": "episodic",
                        "summary": "用户提到：我在杭州做后端开发",
                        "raw_excerpt": "我在杭州做后端开发",
                        "created_at": "2099-01-01T00:00:00+00:00",
                        "last_recalled_at": "2099-01-01T00:00:00+00:00",
                        "memory_strength": 1,
                        "source_session_id": "sess_seed",
                        "source_message_ids": ["msg_seed"],
                        "status": "active",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False)

            manager.prepare_context("杭州后端开发有什么建议", session_id="sess_probe")

            updated = json.loads(archival.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(updated["memory_strength"], 1)

    def test_decay_policy_disabled_by_default_keeps_old_memory_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            archival = memory_root / "archival_memory.jsonl"
            archival.write_text(
                json.dumps(
                    {
                        "id": "mem_old",
                        "type": "episodic",
                        "summary": "用户提到：我在北京工作",
                        "raw_excerpt": "我在北京工作",
                        "created_at": "2020-01-01T00:00:00+00:00",
                        "last_recalled_at": "2020-01-01T00:00:00+00:00",
                        "memory_strength": 1,
                        "source_session_id": "sess_old",
                        "source_message_ids": ["msg_old"],
                        "status": "active",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False)

            manager.prepare_context("你记得我在哪工作吗", session_id="sess_probe")

            updated = json.loads(archival.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(updated["status"], "active")
            self.assertNotIn("decayed_at", updated)

    def test_decay_policy_enabled_requires_one_year_and_writes_decayed_at(self) -> None:
        original_enabled = os.environ.get("Guga_MEMORY_DECAY_ENABLED")
        original_min_age = os.environ.get("Guga_MEMORY_DECAY_MIN_AGE_DAYS")
        os.environ["Guga_MEMORY_DECAY_ENABLED"] = "1"
        os.environ.pop("Guga_MEMORY_DECAY_MIN_AGE_DAYS", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                memory_root = Path(tmp)
                archival = memory_root / "archival_memory.jsonl"
                rows = [
                    {
                        "id": "mem_under_year",
                        "type": "episodic",
                        "summary": "用户提到：我在上海工作",
                        "raw_excerpt": "我在上海工作",
                        "created_at": "2026-01-01T00:00:00+08:00",
                        "last_recalled_at": "2026-01-01T00:00:00+08:00",
                        "memory_strength": 1,
                        "source_session_id": "sess_recent",
                        "source_message_ids": ["msg_recent"],
                        "status": "active",
                    },
                    {
                        "id": "mem_over_year",
                        "type": "episodic",
                        "summary": "用户提到：我在北京工作",
                        "raw_excerpt": "我在北京工作",
                        "created_at": "2024-01-01T00:00:00+08:00",
                        "last_recalled_at": "2024-01-01T00:00:00+08:00",
                        "memory_strength": 1,
                        "source_session_id": "sess_old",
                        "source_message_ids": ["msg_old"],
                        "status": "active",
                    },
                ]
                archival.write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8",
                )
                manager = MemoryManager(memory_root=memory_root, enable_semantic=False)

                manager.prepare_context("你记得我在哪工作吗", session_id="sess_probe")

                updated_rows = [json.loads(line) for line in archival.read_text(encoding="utf-8").splitlines()]
                by_id = {row["id"]: row for row in updated_rows}
                self.assertEqual(by_id["mem_under_year"]["status"], "active")
                self.assertNotIn("decayed_at", by_id["mem_under_year"])
                self.assertEqual(by_id["mem_over_year"]["status"], "decayed")
                self.assertTrue(by_id["mem_over_year"]["decayed_at"].endswith("+08:00"))
        finally:
            if original_enabled is None:
                os.environ.pop("Guga_MEMORY_DECAY_ENABLED", None)
            else:
                os.environ["Guga_MEMORY_DECAY_ENABLED"] = original_enabled
            if original_min_age is None:
                os.environ.pop("Guga_MEMORY_DECAY_MIN_AGE_DAYS", None)
            else:
                os.environ["Guga_MEMORY_DECAY_MIN_AGE_DAYS"] = original_min_age

    def test_finalize_turn_writes_memorybank_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False, top_k=4)
            manager.record_user_message("sess_layers", "我叫小明，我在深圳工作，也不喜欢说教式安慰")
            manager.record_assistant_message("sess_layers", "记住了")
            manager.finalize_turn("sess_layers")

            archival_payload = json.loads((memory_root / "archival_memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(archival_payload["memory_strength"], 1)
            self.assertEqual(archival_payload["retention"], 1.0)
            self.assertEqual(archival_payload["type"], "episodic")

            session_memory_payload = json.loads((memory_root / "session_memories.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(session_memory_payload["type"], "conversation_turn")
            self.assertEqual(session_memory_payload["memory_strength"], 1)

            event_rows = [
                json.loads(line)
                for line in (memory_root / "event_summaries.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            daily_event = next(row for row in event_rows if row["scope"] == "daily")
            global_event = next(row for row in event_rows if row["scope"] == "global")
            self.assertEqual(daily_event["type"], "event_summary")
            self.assertIn("深圳工作", daily_event["summary"])
            self.assertIn("深圳工作", global_event["summary"])

            insight_rows = [
                json.loads(line)
                for line in (memory_root / "personality_insights.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(insight_rows[0]["scope"], "daily")
            self.assertIn("工作或职业背景", insight_rows[0]["summary"])

            profile = json.loads((memory_root / "profile.json").read_text(encoding="utf-8"))
            self.assertIn("portrait_summary", profile)
            self.assertIn("工作或职业背景", profile["portrait_summary"])
            self.assertIn("负向偏好", profile["portrait_summary"])
            self.assertEqual(profile["daily_personality_count"], 1)

            context = manager.prepare_context("你记得我在深圳工作的事吗", session_id="sess_probe")
            prompt = manager.compose_system_prompt("你是助手", context)
            self.assertIn("[User Portrait]", prompt)
            self.assertIn("[Relevant Event Summaries]", prompt)
            self.assertIn("[Relevant Conversation Memories]", prompt)

    def test_daily_summary_aggregates_sessions_from_same_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False, top_k=4)
            manager.record_user_message("sess_a", "我是叔本明")
            manager.record_assistant_message("sess_a", "记住了")
            manager.finalize_turn("sess_a")

            manager.record_user_message("sess_b", "我是谁")
            manager.record_assistant_message("sess_b", "你是叔本明")
            manager.finalize_turn("sess_b")

            event_rows = [
                json.loads(line)
                for line in (memory_root / "event_summaries.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            daily_event = next(row for row in event_rows if row["scope"] == "daily")
            self.assertIn("我是叔本明", daily_event["summary"])
            self.assertIn("我是谁", daily_event["summary"])

            profile = json.loads((memory_root / "profile.json").read_text(encoding="utf-8"))
            self.assertIn("叔本明", profile["portrait_summary"])


if __name__ == "__main__":
    unittest.main()
