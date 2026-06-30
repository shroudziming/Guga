from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.manager import MemoryManager


class SummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "Memory route classifier" in prompt:
            return json.dumps(
                [
                    {
                        "target": "archival_memory",
                        "label": "stable_context",
                        "content": "用户叫小明，在深圳工作",
                        "topic": "profile",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                    {"target": "personality_insight", "label": "stable_context", "content": "用户在深圳工作。"},
                ],
                ensure_ascii=False,
            )
        if "用户画像整理器" in prompt:
            return "- 用户在深圳工作。"
        return "- 用户叫小明，在深圳工作。"


class MemoryManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_root = Path(self.tmp.name)
        self.manager = MemoryManager(
            memory_root=self.memory_root,
            model=SummaryModel(),
            top_k=2,
            recency_weight=0.2,
            enable_semantic=False,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_prepare_context_returns_provenance_hits(self) -> None:
        archival = self.memory_root / "archival_memory.jsonl"
        rows = [
            {
                "id": "mem_old",
                "summary": "用户提到：我在北京工作",
                "raw_excerpt": "我在北京工作",
                "created_at": "2025-01-01T00:00:00+00:00",
                "source_session_id": "sess_old",
                "source_message_ids": ["msg_old"],
                "importance": 0.6,
                "confidence": 0.6,
                "status": "active",
            },
            {
                "id": "mem_new",
                "summary": "用户提到：我现在在上海工作",
                "raw_excerpt": "我现在在上海工作",
                "created_at": "2099-01-01T00:00:00+00:00",
                "source_session_id": "sess_new",
                "source_message_ids": ["msg_new"],
                "importance": 0.7,
                "confidence": 0.8,
                "status": "active",
            },
            {
                "id": "mem_irrelevant",
                "summary": "用户提到：我喜欢猫",
                "raw_excerpt": "我喜欢猫",
                "created_at": "2099-01-01T00:00:00+00:00",
                "source_session_id": "sess_other",
                "source_message_ids": ["msg_other"],
                "status": "active",
            },
        ]
        archival.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8")

        context = self.manager.prepare_context("你记得我在哪工作吗", session_id="sess_test")

        self.assertGreaterEqual(len(context.hits), 1)
        self.assertEqual(context.hits[0].id, "mem_new")
        self.assertTrue(context.hits[0].source_session_id)
        self.assertTrue(context.hits[0].source_message_ids)
        self.assertIn("用户提到", context.archival_memories[0])

        prompt = self.manager.compose_system_prompt("你是一个助手", context)
        self.assertIn("[Relevant Memory]", prompt)
        self.assertIn("[Relevant Documents]", prompt)
        self.assertIn("mem_new", prompt)
        self.assertIn("src=sess_new/msg_new", prompt)

    def test_compose_prompt_explicitly_handles_no_hit(self) -> None:
        context = self.manager.prepare_context("完全不相关问题", session_id="sess_none")
        prompt = self.manager.compose_system_prompt("你是一个助手", context)
        self.assertIn("当前未检索到可靠历史记忆", prompt)
        self.assertIn("当前未检索到相关文档片段", prompt)
        self.assertIn("不要编造", prompt)

    def test_finalize_turn_writes_archival_and_session_schema(self) -> None:
        session_id = "sess_schema"
        self.manager.record_user_message(session_id, "我叫小明，我在深圳工作")
        self.manager.record_assistant_message(session_id, "记住了")
        self.manager.finalize_turn(session_id)

        session_file = self.memory_root / "sessions" / f"{session_id}.jsonl"
        self.assertTrue(session_file.exists())
        session_rows = [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(session_rows), 2)
        self.assertIn("source", session_rows[0])
        self.assertIn("metadata", session_rows[0])

        archival_file = self.memory_root / "archival_memory.jsonl"
        self.assertTrue(archival_file.exists())
        payload = json.loads(archival_file.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(payload["type"], "episodic")
        self.assertEqual(payload["status"], "active")
        self.assertTrue(payload["source_message_ids"])


if __name__ == "__main__":
    unittest.main()
