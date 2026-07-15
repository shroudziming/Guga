from __future__ import annotations

import json
import re
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from guga.memory.forgetting import retention_score
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.rag.schemas import RetrievalHit


class SummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "Low-level memory consolidation" in prompt:
            summary = "用户提到深圳工作和不喜欢说教式安慰。"
            if "叔本明" in prompt:
                summary = "用户自称叔本明，并询问我是谁。"
            return json.dumps(
                {
                    "semantic_event_operations": [
                        {"operation": "create", "event_kind": "state_change", "subject": "user", "entity": "user context", "description": summary, "time_expression": "", "end_unknown": True, "source_message_ids": [], "guga_reflection": {"appraisal": "这项用户背景值得记住。", "felt_response": "我会认真留意。"}}
                    ],
                    "event_summaries": [
                        {
                            "summary": summary,
                            "source_message_ids": [],
                            "confidence": 0.9,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "High-level memory consolidation" in prompt:
            event_match = re.search(r'"id"\s*:\s*"(evt_[^"]+)"', prompt)
            event_ids = [event_match.group(1)] if event_match else []
            if "叔本明" in prompt:
                return json.dumps(
                    {
                        "decision": "update_high_level_memory",
                        "archival_operations": [
                            {"topic": "identity", "summary": "用户自称叔本明。", "importance": 0.8, "confidence": 0.9, "source_event_ids": event_ids}
                        ],
                        "user_model_operations": [],
                        "reason": "identity",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "decision": "update_high_level_memory",
                    "archival_operations": [
                        {
                            "topic": "profile",
                            "summary": "用户提到深圳工作和不喜欢说教式安慰",
                            "importance": 0.8,
                            "confidence": 0.9,
                            "source_event_ids": event_ids,
                        }
                    ],
                    "user_model_operations": [],
                    "reason": "profile",
                },
                ensure_ascii=False,
            )
        if "Memory route classifier" in prompt:
            if "叔本明" in prompt:
                return json.dumps(
                    [
                        {"target": "archival_memory", "label": "stable_identity", "content": "用户自称叔本明。", "topic": "identity"},
                        {"target": "personality_insight", "label": "stable_identity", "content": "用户自称叔本明。"},
                    ],
                    ensure_ascii=False,
                )
            return json.dumps(
                [
                    {
                        "target": "archival_memory",
                        "label": "stable_context",
                        "content": "用户提到深圳工作和不喜欢说教式安慰",
                        "topic": "profile",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                    {"target": "personality_insight", "label": "stable_context", "content": "用户谈到了工作或职业背景。"},
                    {"target": "personality_insight", "label": "stable_preference", "content": "用户表达了明确的负向偏好或互动边界。"},
                ],
                ensure_ascii=False,
            )
        if "用户画像整理器" in prompt:
            if "叔本明" in prompt:
                return "- 用户自称叔本明。"
            return "\n".join(
                [
                    "- 用户谈到了工作或职业背景。",
                    "- 用户表达了明确的负向偏好或互动边界。",
                ]
            )
        if "Summarize the events" in prompt:
            lines = []
            if "深圳工作" in prompt or "深圳" in prompt:
                lines.append("- 用户提到深圳工作。")
            if "不喜欢说教式安慰" in prompt:
                lines.append("- 用户不喜欢说教式安慰。")
            if "我是叔本明" in prompt:
                lines.append("- 用户说我是叔本明。")
            if "我是谁" in prompt:
                lines.append("- 用户询问我是谁。")
            return "\n".join(lines) or "- LLM generated summary"
        if "global event summary" in prompt:
            return prompt.split("summary.\n\n", 1)[-1].strip() or "- LLM generated summary"
        return "- LLM generated summary"


class MemoryBankReproTest(unittest.TestCase):
    def test_finalize_turn_uses_llm_archival_extraction_when_model_available(self) -> None:
        class FakeModel:
            def generate_reply(self, messages, gen):
                prompt = messages[-1]["content"]
                if "Low-level memory consolidation" in prompt:
                    return json.dumps(
                        {
                            "semantic_event_operations": [
                                {"operation": "create", "event_kind": "state_change", "subject": "user", "entity": "work", "description": "The user works as a backend engineer in Hangzhou.", "time_expression": "", "end_unknown": True, "source_message_ids": [], "guga_reflection": {"appraisal": "This work context is worth remembering.", "felt_response": "I will keep it in mind."}}
                            ],
                            "event_summaries": [
                                {
                                    "summary": "The user works as a backend engineer in Hangzhou.",
                                    "confidence": 0.9,
                                }
                            ],
                        }
                    )
                if "High-level memory consolidation" in prompt:
                    event_match = re.search(r'"id"\s*:\s*"(evt_[^"]+)"', prompt)
                    return json.dumps(
                        {
                            "decision": "update_high_level_memory",
                            "archival_operations": [
                                {
                                    "topic": "work",
                                    "summary": "The user works as a backend engineer in Hangzhou.",
                                    "importance": 0.9,
                                    "confidence": 0.8,
                                    "source_event_ids": [event_match.group(1)] if event_match else [],
                                }
                            ],
                            "user_model_operations": [],
                            "reason": "stable work context",
                        }
                    )
                return "- LLM generated summary"

        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(
                memory_root=memory_root,
                model=FakeModel(),
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
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
            manager._retrieve_semantic = lambda **_: (
                [
                    RetrievalHit(
                        chunk_id="memory:mem_work:c0",
                        text="用户提到：我在杭州做后端开发",
                        score=0.8,
                        source_type="memory",
                        source_id="mem_work",
                        source_session_id="sess_seed",
                        source_message_id="msg_seed",
                        created_at="2099-01-01T00:00:00+00:00",
                    )
                ],
                [],
            )

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
            manager = MemoryManager(
                memory_root=memory_root,
                model=SummaryModel(),
                enable_semantic=False,
                top_k=4,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
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
            batch_event = next(row for row in event_rows if row["source_of_truth"] is False)
            self.assertEqual(batch_event["type"], "event_summary")
            self.assertIn("深圳工作", batch_event["summary"])

            records = manager._load_archival_records()
            manager._retrieve_semantic = lambda **_: (
                [
                    RetrievalHit(
                        chunk_id=f"memory:{record['id']}:c0",
                        text=str(record.get("summary") or record.get("raw_excerpt") or ""),
                        score=0.8,
                        source_type="memory",
                        source_id=str(record["id"]),
                        source_session_id=str(record.get("source_session_id", "")),
                        source_message_id=str((record.get("source_message_ids") or [""])[0]),
                        created_at=str(record.get("created_at", "")),
                    )
                    for record in records
                ],
                [],
            )

            context = manager.prepare_context("你记得我在深圳工作的事吗", session_id="sess_probe")
            prompt = manager.compose_system_prompt("你是助手", context)
            self.assertIn("[Semantic Events]", prompt)
            self.assertIn("[Archival Memory]", prompt)
            self.assertIn("[Derived Event Summaries]", prompt)
            self.assertIn("[Raw Evidence]", prompt)
            self.assertNotIn("[Guga User Model]", prompt)

    def test_daily_summary_aggregates_sessions_from_same_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(
                memory_root=memory_root,
                model=SummaryModel(),
                enable_semantic=False,
                top_k=4,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
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
            summaries = "\n".join(row["summary"] for row in event_rows if row["source_of_truth"] is False)
            self.assertIn("叔本明", summaries)



if __name__ == "__main__":
    unittest.main()
