from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.rag.faiss_store import IncompatibleIndexError
from guga.rag.schemas import RetrievalHit
from guga.types import DocumentHit, MemoryContext, MemoryHit


class SummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "Low-level memory consolidation" in prompt:
            return json.dumps(
                {
                    "semantic_event_operations": [
                        {
                            "operation": "create",
                            "event_kind": "state_change",
                            "subject": "user",
                            "entity": "work location",
                            "description": "用户叫小明，在深圳工作。",
                            "time_expression": "",
                            "end_unknown": True,
                            "source_message_ids": [],
                        }
                    ],
                    "event_summaries": [
                        {
                            "summary": "用户叫小明，在深圳工作。",
                            "source_message_ids": [],
                            "confidence": 0.9,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "High-level memory consolidation" in prompt:
            event_match = re.search(r'"id"\s*:\s*"(evt_[^"]+)"', prompt)
            return json.dumps(
                {
                    "decision": "update_high_level_memory",
                    "archival_operations": [
                        {
                            "topic": "profile",
                            "summary": "用户叫小明，在深圳工作",
                            "importance": 0.8,
                            "confidence": 0.9,
                            "source_event_ids": [event_match.group(1)] if event_match else [],
                        }
                    ],
                    "user_model_operations": [
                        {
                            "operation": "upsert",
                            "statement": "Guga 了解到用户在深圳工作。",
                            "kind": "work_context",
                            "confidence": 0.9,
                            "stability": "explicit",
                            "source_event_ids": [event_match.group(1)] if event_match else [],
                        }
                    ],
                    "reason": "stable profile",
                },
                ensure_ascii=False,
            )
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
            enable_semantic=False,
            consolidation_config=MemoryConsolidationConfig(batch_turns=1),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_incompatible_persisted_index_is_rebuilt(self) -> None:
        class StoreStub:
            def has_persisted_index(self) -> bool:
                return True

        class RagStub:
            def __init__(self) -> None:
                self.store = StoreStub()
                self.rebuild_calls = 0

            def ensure_loaded(self) -> None:
                raise IncompatibleIndexError("old embedding model")

            def rebuild_indexes(self, memory_root):
                self.rebuild_calls += 1
                return {"memory_chunks": 1, "document_chunks": 0, "total_chunks": 1}

        rag = RagStub()
        self.manager.rag_pipeline = rag

        self.manager._ensure_semantic_index("sess_rebuild")

        self.assertEqual(rag.rebuild_calls, 1)
        self.assertTrue(self.manager._semantic_ready)

    def test_compatible_persisted_index_prunes_invalid_memory_sources(self) -> None:
        class StoreStub:
            def has_persisted_index(self) -> bool:
                return True

        class RagStub:
            def __init__(self) -> None:
                self.store = StoreStub()
                self.prune_calls = 0

            def ensure_loaded(self) -> None:
                return None

            def prune_invalid_memory_records(self, memory_root) -> int:
                _ = memory_root
                self.prune_calls += 1
                return 2

        rag = RagStub()
        self.manager.rag_pipeline = rag

        self.manager._ensure_semantic_index("sess_prune")

        self.assertEqual(rag.prune_calls, 1)
        self.assertTrue(self.manager._semantic_ready)

    def test_semantic_retrieval_failure_is_not_silently_ignored(self) -> None:
        class RagStub:
            def retrieve(self, **kwargs):
                _ = kwargs
                raise RuntimeError("BGE-M3 unavailable")

        self.manager.rag_pipeline = RagStub()
        self.manager._semantic_ready = True

        with self.assertRaisesRegex(RuntimeError, "BGE-M3 unavailable"):
            self.manager._retrieve_semantic("query", "sess_failure")

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
        self.manager._retrieve_semantic = lambda **_: (
            [
                RetrievalHit(
                    chunk_id="memory:mem_new:c0",
                    text="用户提到：我现在在上海工作",
                    score=0.8,
                    source_type="memory",
                    source_id="mem_new",
                    source_session_id="sess_new",
                    source_message_id="msg_new",
                    created_at="2099-01-01T00:00:00+00:00",
                )
            ],
            [],
        )

        context = self.manager.prepare_context("你记得我在哪工作吗", session_id="sess_test")

        self.assertGreaterEqual(len(context.hits), 1)
        self.assertEqual(context.hits[0].id, "mem_new")
        self.assertTrue(context.hits[0].source_session_id)
        self.assertTrue(context.hits[0].source_message_ids)
        self.assertIn("用户提到", context.archival_memories[0])

        prompt = self.manager.compose_system_prompt("你是一个助手", context)
        self.assertIn("[Archival Memory]", prompt)
        self.assertNotIn("[Relevant Documents]", prompt)
        self.assertIn("mem_new", prompt)
        self.assertIn("src=sess_new/msg_new", prompt)

    def test_prepare_context_excludes_user_model_backed_by_inactive_event(self) -> None:
        (self.memory_root / "semantic_events.jsonl").write_text(
            json.dumps(
                {
                    "id": "evt_cancelled",
                    "type": "semantic_event",
                    "description": "用户周日看牙。",
                    "status": "inactive",
                    "inactive_reason": "cancelled",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.memory_root / "guga_user_model.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-13T00:00:00+08:00",
                    "insights": [
                        {
                            "id": "gum_cancelled",
                            "statement": "用户正在安排看牙。",
                            "source_event_ids": ["evt_cancelled"],
                            "status": "active",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        context = self.manager.prepare_context("最近有什么安排", "sess_current")

        self.assertEqual(context.user_portrait, "")

    def test_compose_prompt_explicitly_handles_no_hit(self) -> None:
        context = self.manager.prepare_context("完全不相关问题", session_id="sess_none")
        prompt = self.manager.compose_system_prompt("你是一个助手", context)
        self.assertNotIn("当前未检索到可靠历史记忆", prompt)
        self.assertNotIn("当前未检索到相关文档片段", prompt)
        self.assertNotIn("[Current Rule]", prompt)

    def test_general_prompt_does_not_render_event_doc_or_current_turn_noise(self) -> None:
        self.manager.record_user_message("sess_social", "你撒个娇看看")

        context = self.manager.prepare_context("你撒个娇看看", session_id="sess_social")
        prompt = self.manager.compose_system_prompt("你是一个助手", context)

        self.assertIn("[Base Persona]", prompt)
        self.assertNotIn("[Relevant Event Summaries]", prompt)
        self.assertNotIn("[Historical Conversation Context]", prompt)
        self.assertNotIn("[Relevant Documents]", prompt)
        self.assertNotIn("你撒个娇看看", prompt)

    def test_history_prompt_renders_summary_and_source_messages(self) -> None:
        session_id = "sess_history"
        user_id = self.manager.record_user_message(
            session_id,
            "我之前问你推荐过一部悬疑网剧。",
            created_at="2026-07-12T09:00:00+08:00",
        )
        assistant_id = self.manager.record_assistant_message(
            session_id,
            "我推荐过《隐秘的角落》。",
            created_at="2026-07-12T09:01:00+08:00",
        )
        event = {
            "id": "evt_daily_20260628",
            "type": "event_summary",
            "scope": "daily",
            "day": "2026-06-28",
            "summary": "用户询问并获得悬疑网剧推荐。",
            "raw_excerpt": "用户问悬疑网剧，助手推荐《隐秘的角落》。",
            "created_at": "2026-06-28T01:35:00+08:00",
            "source_session_id": session_id,
            "source_message_ids": [user_id, assistant_id],
            "memory_strength": 1,
            "importance": 0.8,
            "confidence": 0.9,
            "status": "active",
        }
        (self.memory_root / "event_summaries.jsonl").write_text(
            json.dumps(event, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.manager._retrieve_semantic = lambda **_: (
            [
                RetrievalHit(
                    chunk_id="memory:evt_daily_20260628:c0",
                    text=event["summary"],
                    score=0.8,
                    source_type="memory",
                    source_id=event["id"],
                    source_session_id=session_id,
                    source_message_id=user_id,
                    created_at=event["created_at"],
                ),
                RetrievalHit(
                    chunk_id=f"memory:turn_{user_id}:c0",
                    text="我之前问你推荐过一部悬疑网剧。",
                    score=0.7,
                    source_type="memory",
                    source_id=f"turn_{user_id}",
                    source_session_id=session_id,
                    source_message_id=user_id,
                    created_at="2026-07-12T09:00:00+08:00",
                ),
            ],
            [],
        )

        context = self.manager.prepare_context("上次我们聊了什么", session_id="sess_now")
        prompt = self.manager.compose_system_prompt("你是一个助手", context)

        self.assertIn("[Derived Event Summaries]", prompt)
        self.assertIn("[Raw Evidence]", prompt)
        self.assertIn("[Raw Evidence]", prompt)
        self.assertIn("at=2026-07-12", prompt)
        self.assertIn("用户询问并获得悬疑网剧推荐。", prompt)
        self.assertIn("我之前问你推荐过一部悬疑网剧。", prompt)
        self.assertNotIn(f"Assistant({assistant_id})", prompt)
        self.assertNotIn("[Historical Conversation Context]", prompt)

    def test_profile_prompt_renders_portrait_without_event_summary(self) -> None:
        (self.memory_root / "semantic_events.jsonl").write_text(
            json.dumps(
                {
                    "id": "evt_identity",
                    "type": "semantic_event",
                    "event_kind": "state_change",
                    "subject": "user",
                    "entity": "identity",
                    "description": "用户自称叔本明。",
                    "status": "active",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.memory_root / "guga_user_model.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-06-28T01:35:00+08:00",
                    "insights": [
                        {
                            "id": "gum_name",
                            "statement": "用户自称叔本明。",
                            "kind": "identity",
                            "confidence": 0.9,
                            "stability": "explicit",
                            "source_event_ids": ["evt_identity"],
                            "status": "active",
                            "updated_at": "2026-06-28T01:35:00+08:00",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        event = {
            "id": "evt_daily_20260628",
            "type": "event_summary",
            "scope": "daily",
            "day": "2026-06-28",
            "summary": "一次普通闲聊摘要。",
            "created_at": "2026-06-28T01:35:00+08:00",
            "source_session_id": "sess_old",
            "source_message_ids": [],
            "status": "active",
        }
        (self.memory_root / "event_summaries.jsonl").write_text(
            json.dumps(event, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        context = self.manager.prepare_context("你觉得我是谁？", session_id="sess_now")
        prompt = self.manager.compose_system_prompt("你是一个助手", context)

        self.assertIn("[Guga User Model]", prompt)
        self.assertIn("叔本明", prompt)
        self.assertNotIn("一次普通闲聊摘要", prompt)
        self.assertNotIn("[Derived Event Summaries]", prompt)

    def test_document_section_only_renders_when_document_hits_exist(self) -> None:
        context = MemoryContext(
            document_hits=[
                DocumentHit(
                    chunk_id="doc_1",
                    text="文档片段内容",
                    score=0.77,
                    source_id="source_doc",
                    source_path="notes.md",
                )
            ],
            query_route="hybrid",
            query_reason="default_hybrid",
        )

        prompt = self.manager.compose_system_prompt("你是一个助手", context)

        self.assertIn("[Relevant Documents]", prompt)
        self.assertIn("doc_1", prompt)
        self.assertIn("文档片段内容", prompt)

    def test_hybrid_prompt_renders_memory_layers_in_priority_order(self) -> None:
        context = MemoryContext(
            hits=[
                MemoryHit(
                    id="evt_current",
                    summary="用户当前房贷预批额度为 $400,000。",
                    memory_type="semantic_event",
                    source_session_id="mortgage_update",
                    source_message_ids=["msg_400"],
                    created_at="2023-11-30T00:36:00+08:00",
                ),
                MemoryHit(
                    id="mem_background",
                    summary="用户正在购房。",
                    memory_type="episodic",
                    source_session_id="mortgage_old",
                    source_message_ids=["msg_background"],
                    created_at="2023-08-11T00:01:00+08:00",
                ),
                MemoryHit(
                    id="summary_mortgage",
                    summary="房贷相关对话摘要。",
                    memory_type="event_summary",
                    source_session_id="mortgage_update",
                    source_message_ids=["msg_400"],
                    created_at="2023-11-30T00:36:00+08:00",
                ),
                MemoryHit(
                    id="turn_400",
                    summary="remember when I got pre-approved for $400,000 from Wells Fargo?",
                    raw_excerpt="VERY LONG ORIGINAL MESSAGE THAT MUST NOT BE RENDERED",
                    memory_type="conversation_turn",
                    source_session_id="mortgage_update",
                    source_message_ids=["msg_400"],
                    created_at="2023-11-30T00:36:00+08:00",
                ),
            ],
            user_portrait="用户倾向在重要财务决策前确认细节。",
            query_route="hybrid",
        )

        prompt = self.manager.compose_system_prompt("你是一个助手", context)

        self.assertLess(prompt.index("[Semantic Events]"), prompt.index("[Archival Memory]"))
        self.assertLess(prompt.index("[Archival Memory]"), prompt.index("[Derived Event Summaries]"))
        self.assertLess(prompt.index("[Derived Event Summaries]"), prompt.index("[Raw Evidence]"))
        self.assertLess(prompt.index("[Raw Evidence]"), prompt.index("[Guga User Model]"))
        self.assertIn("2023-11-30 00:36", prompt)
        self.assertNotIn("VERY LONG ORIGINAL MESSAGE", prompt)

    def test_semantic_raw_chunk_does_not_restore_full_message(self) -> None:
        record = {
            "id": "turn_long",
            "type": "conversation_turn",
            "summary": "FULL ORIGINAL MESSAGE " * 100,
            "raw_excerpt": "FULL ORIGINAL MESSAGE " * 100,
            "created_at": "2023-11-30T00:36:00+08:00",
            "source_session_id": "mortgage_update",
            "source_message_ids": ["msg_400"],
            "status": "active",
        }
        semantic_hit = RetrievalHit(
            chunk_id="memory:turn_long:c4",
            text="remember when I got pre-approved for $400,000 from Wells Fargo?",
            score=0.8,
            source_type="memory",
            source_id="turn_long",
            source_session_id="mortgage_update",
            source_message_id="msg_400",
            created_at="2023-11-30T00:36:00+08:00",
        )

        hits = self.manager._merge_memory_hits(
            [semantic_hit],
            [record],
            current_turn_ids=set(),
            time_hints={},
            session_id="sess_now",
        )

        self.assertEqual(hits[0].summary, semantic_hit.text)
        self.assertEqual(hits[0].raw_excerpt, semantic_hit.text)
        self.assertEqual(hits[0].chunk_id, "memory:turn_long:c4")

    def test_context_selection_reserves_relevant_slots_for_memory_layers(self) -> None:
        self.manager.top_k = 4
        hits = [
            MemoryHit(id="raw_1", summary="raw one", score=0.99, memory_type="conversation_turn"),
            MemoryHit(id="raw_2", summary="raw two", score=0.98, memory_type="conversation_turn"),
            MemoryHit(id="archive", summary="background", score=0.70, memory_type="episodic"),
            MemoryHit(id="summary", summary="derived", score=0.60, memory_type="event_summary"),
            MemoryHit(id="event", summary="current fact", score=0.50, memory_type="semantic_event"),
        ]

        selected = self.manager._filter_memory_hits(hits)

        self.assertEqual(
            {hit.memory_type for hit in selected},
            {"semantic_event", "episodic", "event_summary", "conversation_turn"},
        )

    def test_context_selection_keeps_three_hits_per_memory_layer(self) -> None:
        self.manager.top_k = 3
        memory_types = ("semantic_event", "episodic", "event_summary", "conversation_turn")
        hits = [
            MemoryHit(
                id=f"{memory_type}_{index}",
                summary=f"{memory_type} {index}",
                score=1.0 - (index * 0.01),
                memory_type=memory_type,
            )
            for memory_type in memory_types
            for index in range(4)
        ]

        selected = self.manager._filter_memory_hits(hits)

        self.assertEqual(len(selected), 12)
        for memory_type in memory_types:
            layer_hits = [hit for hit in selected if hit.memory_type == memory_type]
            self.assertEqual(len(layer_hits), 3)
            self.assertEqual([hit.id for hit in layer_hits], [f"{memory_type}_{index}" for index in range(3)])

    def test_finalize_turn_writes_archival_and_session_schema(self) -> None:
        session_id = "sess_schema"
        self.manager.record_user_message(session_id, "我叫小明，我在深圳工作")
        self.manager.record_assistant_message(session_id, "记住了")
        self.manager.finalize_turn(session_id)
        self.manager.flush_session_memory(session_id)

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
        self.assertTrue(payload["source_event_ids"])


if __name__ == "__main__":
    unittest.main()
