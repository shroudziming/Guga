from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.types import GenerationConfig


class ConsolidationModel:
    def __init__(self, high_decision: str = "update_high_level_memory") -> None:
        self.prompts: list[str] = []
        self.high_packets: list[dict] = []
        self.high_decision = high_decision

    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Low-level memory consolidation" in prompt:
            include_reflection = "include_guga_reflection: true" in prompt
            reflection = {
                "appraisal": "Guga thinks this is important.",
                "felt_response": "Guga feels attentive.",
                "relational_intent": "Guga should remember the plan gently.",
                "interpretation_confidence": 0.8,
            }
            if not include_reflection:
                reflection = {}
            return json.dumps(
                {
                    "semantic_event_operations": [
                        {
                            "operation": "create",
                            "event_kind": "task",
                            "subject": "user",
                            "entity": "project report",
                            "description": "The user needs to submit the project report.",
                            "time_expression": "2026-07-03",
                            "end_unknown": False,
                            "confidence": 0.91,
                            "source_message_ids": [],
                            "guga_reflection": reflection if include_reflection else {},
                        }
                    ],
                    "event_summaries": [
                        {
                            "action": "upsert",
                            "scope": "batch",
                            "summary": "The user discussed a project report deadline.",
                            "source_message_ids": [],
                            "confidence": 0.84,
                            **reflection,
                        }
                    ],
                }
            )
        if "High-level memory consolidation" in prompt:
            packet_text = prompt.split("Input packet:\n", 1)[1]
            packet, _ = json.JSONDecoder().raw_decode(packet_text)
            self.high_packets.append(packet)
            source_event_ids = [str(packet["semantic_events"][-1]["id"])] if packet.get("semantic_events") else []
            if self.high_decision == "no_high_level_update":
                return json.dumps(
                    {
                        "decision": "no_high_level_update",
                        "archival_operations": [],
                        "user_model_operations": [],
                        "reason": "No stable long-term memory found.",
                    }
                )
            return json.dumps(
                {
                    "decision": "update_high_level_memory",
                    "archival_operations": [
                        {
                            "topic": "deadline",
                            "summary": "The user has a project report deadline.",
                            "importance": 0.8,
                            "confidence": 0.88,
                            "source_event_ids": source_event_ids,
                        }
                    ],
                    "user_model_operations": [
                        {
                            "operation": "upsert",
                            "statement": "Guga should be careful about deadline reminders.",
                            "kind": "reminder_pattern",
                            "confidence": 0.8,
                            "stability": "recurring",
                            "source_event_ids": source_event_ids,
                        }
                    ],
                    "reason": "The low-level memories contain a stable reminder preference.",
                }
            )
        return "chat answer"


class BadHighLevelModel(ConsolidationModel):
    def generate_reply(self, messages, gen):
        prompt = messages[-1]["content"]
        if "High-level memory consolidation" in prompt:
            self.prompts.append(prompt)
            return "{not valid json"
        return super().generate_reply(messages, gen)


class RetryHighLevelModel(ConsolidationModel):
    def __init__(self) -> None:
        super().__init__()
        self.high_attempts = 0

    def generate_reply(self, messages, gen):
        prompt = messages[-1]["content"]
        if "High-level memory consolidation" in prompt:
            self.high_attempts += 1
            if self.high_attempts == 1:
                self.prompts.append(prompt)
                return "{not valid json"
        return super().generate_reply(messages, gen)


class AlwaysBadLowModel(ConsolidationModel):
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Low-level memory consolidation" in prompt:
            return "{not valid json"
        if "High-level memory consolidation" in prompt:
            raise AssertionError("Stage 2 must not run after Stage 1 failure")
        return "chat answer"


class MemoryConsolidationTest(unittest.TestCase):
    def _record_turns(self, manager: MemoryManager, session_id: str, count: int) -> None:
        for index in range(count):
            manager.record_user_message(session_id, f"turn {index}: submit project report on 2026-07-03")
            manager.record_assistant_message(session_id, "noted")
            manager.finalize_turn(session_id)

    def test_pending_turns_do_not_call_llm_until_batch_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ConsolidationModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=10),
            )

            self._record_turns(manager, "sess_batch", 9)

            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["sessions"]["sess_batch"]["pending_turns"]), 9)
            self.assertEqual(model.prompts, [])
            self.assertTrue((Path(tmp) / "sessions" / "sess_batch.jsonl").exists())
            self.assertTrue((Path(tmp) / "session_memories.jsonl").exists())

            self._record_turns(manager, "sess_batch", 1)

            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["sessions"]["sess_batch"]["batch_seq"], 1)
            self.assertEqual(state["sessions"]["sess_batch"]["pending_turns"], [])
            self.assertTrue(any("Low-level memory consolidation" in prompt for prompt in model.prompts))
            self.assertTrue(any("High-level memory consolidation" in prompt for prompt in model.prompts))

    def test_low_level_consolidation_can_omit_guga_reflection_for_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ConsolidationModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(
                    batch_turns=1,
                    include_guga_reflection=False,
                    enable_user_model_updates=False,
                ),
            )

            self._record_turns(manager, "sess_benchmark", 1)

            fact = json.loads((Path(tmp) / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
            event = json.loads((Path(tmp) / "event_summaries.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("guga_reflection", fact)
            self.assertFalse(event.get("guga_assessment"))
            self.assertFalse(event.get("guga_thought"))
            self.assertFalse((Path(tmp) / "profile.json").exists())
            self.assertFalse((Path(tmp) / "personality_insights.jsonl").exists())

    def test_high_level_consolidation_receives_persisted_low_level_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ConsolidationModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )

            self._record_turns(manager, "sess_stage_order", 1)

            self.assertEqual(len(model.high_packets), 1)
            packet = model.high_packets[0]
            self.assertNotIn("pending_low_level_updates", packet)
            persisted = json.loads((Path(tmp) / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(packet["semantic_events"][-1]["id"], persisted["id"])

    def test_high_level_packet_excludes_inactive_events_and_stale_derived_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MemoryManager(memory_root=root, enable_semantic=False)
            active_id = manager.semantic_event_store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "state_change",
                        "subject": "user",
                        "entity": "mortgage preapproval",
                        "description": "用户当前的房贷预批额度为四十万美元。",
                        "end_unknown": True,
                    }
                ],
                session_id="sess_current",
                include_guga_reflection=False,
            ).created_event_ids[0]
            cancelled_id = manager.semantic_event_store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "appointment",
                        "subject": "user",
                        "entity": "dentist appointment",
                        "description": "用户周日看牙。",
                        "end_unknown": False,
                    }
                ],
                session_id="sess_cancelled",
                include_guga_reflection=False,
            ).created_event_ids[0]
            manager.semantic_event_store.apply_operations(
                operations=[{"operation": "cancel", "target_event_id": cancelled_id}],
                session_id="sess_cancelled",
                include_guga_reflection=False,
            )
            (root / "archival_memory.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "mem_current", "summary": "当前额度", "source_event_ids": [active_id], "status": "active"}),
                        json.dumps({"id": "mem_cancelled", "summary": "周日看牙", "source_event_ids": [cancelled_id], "status": "active"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "event_summaries.jsonl").write_text(
                json.dumps({"id": "summary_cancelled", "type": "event_summary", "summary": "周日看牙", "covered_event_ids": [cancelled_id], "deactivated_event_ids": [cancelled_id], "status": "active"})
                + "\n",
                encoding="utf-8",
            )

            packet = manager._build_high_level_packet()

            self.assertEqual([event["id"] for event in packet["semantic_events"]], [active_id])
            self.assertEqual([memory["id"] for memory in packet["archival_memory"]], ["mem_current"])
            self.assertEqual(packet["event_summaries"], [])

    def test_low_level_packet_uses_rag_event_ids_without_keyword_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            store = manager.semantic_event_store
            relevant = store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "state_change",
                        "subject": "user",
                        "entity": "房贷预批额度",
                        "description": "用户的房贷预批额度为三十五万美元。",
                        "time_expression": "",
                        "end_unknown": True,
                        "reference_created_at": "2026-07-01T09:00:00+08:00",
                    }
                ],
                session_id="sess_old",
                include_guga_reflection=False,
            ).created_event_ids[0]
            rag_selected = ""
            for index in range(21):
                created = store.apply_operations(
                    operations=[
                        {
                            "operation": "create",
                            "event_kind": "task",
                            "subject": "user",
                            "entity": f"无关任务{index}",
                            "description": f"用户需要完成无关任务{index}。",
                            "time_expression": "",
                            "end_unknown": True,
                            "reference_created_at": "2026-07-02T09:00:00+08:00",
                        }
                    ],
                    session_id="sess_recent",
                    include_guga_reflection=False,
                )
                rag_selected = created.created_event_ids[0]

            class RagStub:
                def retrieve(self, query, memory_top_k, document_top_k):
                    _ = query, memory_top_k, document_top_k
                    return [
                        type(
                            "Hit",
                            (),
                            {
                                "source_id": rag_selected,
                                "source_type": "memory",
                                "text": "RAG selected memory",
                                "source_session_id": "sess_recent",
                                "source_message_id": "",
                                "score": 0.9,
                            },
                        )()
                    ], []

            manager.rag_pipeline = RagStub()

            user_id = manager.record_user_message("sess_new", "房贷预批额度改成四十万美元了")
            assistant_id = manager.record_assistant_message("sess_new", "我记下了。")
            packet = manager._build_low_level_packet(
                session_id="sess_new",
                pending_turns=[{"user_message_id": user_id, "assistant_message_id": assistant_id}],
            )

            self.assertEqual(len(packet["recent_active_events"]), 5)
            relevant_ids = {event["id"] for event in packet["relevant_active_events"]}
            self.assertIn(rag_selected, relevant_ids)
            self.assertNotIn(relevant, relevant_ids)

    def test_consolidation_rag_context_rejects_inactive_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            event_id = manager.semantic_event_store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "appointment",
                        "subject": "user",
                        "entity": "dentist appointment",
                        "description": "用户周日看牙。",
                        "end_unknown": False,
                    }
                ],
                session_id="sess_old",
                include_guga_reflection=False,
            ).created_event_ids[0]
            manager.semantic_event_store.apply_operations(
                operations=[{"operation": "cancel", "target_event_id": event_id}],
                session_id="sess_old",
                include_guga_reflection=False,
            )

            class RagStub:
                def retrieve(self, query, memory_top_k, document_top_k):
                    _ = query, memory_top_k, document_top_k
                    return [
                        type(
                            "Hit",
                            (),
                            {
                                "source_id": event_id,
                                "source_type": "memory",
                                "text": "用户周日看牙。",
                                "source_session_id": "sess_old",
                                "source_message_id": "",
                                "score": 0.9,
                            },
                        )()
                    ], []

            manager.rag_pipeline = RagStub()

            self.assertEqual(manager._consolidation_retrieved_context("看牙安排"), [])

    def test_high_level_noop_leaves_high_level_files_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ConsolidationModel(high_decision="no_high_level_update")
            logs: list[str] = []
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                debug=True,
                debug_sink=logs.append,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )

            self._record_turns(manager, "sess_noop", 1)

            self.assertTrue((Path(tmp) / "semantic_events.jsonl").exists())
            self.assertTrue((Path(tmp) / "event_summaries.jsonl").exists())
            self.assertFalse((Path(tmp) / "archival_memory.jsonl").exists())
            self.assertFalse((Path(tmp) / "profile.json").exists())
            self.assertFalse((Path(tmp) / "personality_insights.jsonl").exists())
            self.assertIn("high_level_noop", "\n".join(logs))

    def test_high_level_failure_keeps_only_stage_two_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = BadHighLevelModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )

            self._record_turns(manager, "sess_bad_high", 1)

            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            session_state = state["sessions"]["sess_bad_high"]
            self.assertEqual(session_state["pending_turns"], [])
            self.assertEqual(session_state["pending_high_level"]["batch_seq"], 1)
            self.assertTrue((Path(tmp) / "semantic_events.jsonl").exists())
            self.assertTrue((Path(tmp) / "event_summaries.jsonl").exists())
            self.assertFalse((Path(tmp) / "archival_memory.jsonl").exists())

    def test_low_failure_persists_active_batch_and_queues_new_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = AlwaysBadLowModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
            manager.summarizer.retry_delays = ()

            self._record_turns(manager, "sess_low_pending", 1)
            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            active = state["sessions"]["sess_low_pending"]["active_batch"]
            self.assertEqual(active["stage"], "low")
            self.assertEqual(active["status"], "pending_retry")
            self.assertEqual(active["attempt_count"], 3)

            manager.record_user_message("sess_low_pending", "a later queued turn")
            manager.record_assistant_message("sess_low_pending", "queued")
            manager.finalize_turn("sess_low_pending")
            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            session_state = state["sessions"]["sess_low_pending"]
            self.assertEqual(len(session_state["queued_turns"]), 1)
            self.assertEqual(session_state["active_batch"]["batch_seq"], 1)

    def test_restart_resumes_high_stage_without_repeating_low_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = MemoryManager(
                memory_root=root,
                model=BadHighLevelModel(),
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
            first.summarizer.retry_delays = ()
            self._record_turns(first, "sess_restart_high", 1)
            event_count = len((root / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines())

            recovery_model = ConsolidationModel(high_decision="no_high_level_update")
            recovered = MemoryManager(
                memory_root=root,
                model=recovery_model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
            recovered.summarizer.retry_delays = ()
            recovered.flush_session_memory("sess_restart_high")

            state = json.loads((root / "consolidation_state.json").read_text(encoding="utf-8"))
            session_state = state["sessions"]["sess_restart_high"]
            low_calls = [prompt for prompt in recovery_model.prompts if "Low-level memory consolidation" in prompt]
            self.assertEqual(low_calls, [])
            self.assertIsNone(session_state["active_batch"])
            self.assertEqual(len((root / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()), event_count)

    def test_pending_high_level_retry_does_not_repeat_stage_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = RetryHighLevelModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )

            self._record_turns(manager, "sess_retry_high", 1)
            manager.flush_session_memory("sess_retry_high")

            low_prompts = [prompt for prompt in model.prompts if "Low-level memory consolidation" in prompt]
            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            events = (Path(tmp) / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(low_prompts), 1)
            self.assertEqual(len(events), 1)
            self.assertIsNone(state["sessions"]["sess_retry_high"]["pending_high_level"])
            self.assertEqual(state["sessions"]["sess_retry_high"]["consolidation_batches"], 1)

    def test_chat_session_flush_consolidates_incomplete_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ConsolidationModel()
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=model,
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=10),
            )
            session = ChatSession(
                model=model,
                system_prompt="You are Guga.",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_flush",
            )

            session.reply("submit project report on 2026-07-03")
            manager.wait_for_background_tasks(timeout=3)
            self.assertFalse((Path(tmp) / "event_summaries.jsonl").exists())

            session.flush_memory()

            self.assertTrue((Path(tmp) / "event_summaries.jsonl").exists())

    def test_summarizer_uses_json_mode_when_model_supports_it(self) -> None:
        class JsonModeModel:
            def __init__(self) -> None:
                self.json_calls = 0
                self.chat_calls = 0
                self.prompts: list[str] = []

            def generate_json_reply(self, messages, gen):
                _ = gen
                self.prompts.append(messages[-1]["content"])
                self.json_calls += 1
                return json.dumps({"semantic_event_operations": [], "event_summaries": []})

            def generate_reply(self, messages, gen):
                _ = messages, gen
                self.chat_calls += 1
                return "not json"

        model = JsonModeModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True)

        result = summarizer.consolidate_low_level_memory({"new_turns": []}, include_guga_reflection=False)

        self.assertEqual(result, {"semantic_event_operations": [], "event_summaries": []})
        self.assertEqual(model.json_calls, 1)
        self.assertEqual(model.chat_calls, 0)
        self.assertIn("semantic_event_operations", model.prompts[0])
        self.assertIn("At most 1 event_summary", model.prompts[0])
        self.assertIn("Do not create events for generic questions", model.prompts[0])

    def test_summarizer_retries_invalid_json_once(self) -> None:
        class RetryModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply(self, messages, gen):
                _ = messages, gen
                self.calls += 1
                if self.calls == 1:
                    return "I cannot provide that."
                return json.dumps({"semantic_event_operations": [], "event_summaries": []})

        model = RetryModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True)

        result = summarizer.consolidate_low_level_memory({"new_turns": []}, include_guga_reflection=False)

        self.assertEqual(result, {"semantic_event_operations": [], "event_summaries": []})
        self.assertEqual(model.calls, 2)

    def test_summarizer_retries_invalid_json_mode_response(self) -> None:
        class InvalidJsonModeModel:
            def __init__(self) -> None:
                self.json_calls = 0
                self.chat_calls = 0

            def generate_json_reply(self, messages, gen):
                _ = messages, gen
                self.json_calls += 1
                return "{truncated"

            def generate_reply(self, messages, gen):
                _ = messages, gen
                self.chat_calls += 1
                return json.dumps({"semantic_event_operations": [], "event_summaries": []})

        model = InvalidJsonModeModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())

        result = summarizer.consolidate_low_level_memory({"new_turns": []}, include_guga_reflection=False)

        self.assertEqual(result, {"semantic_event_operations": [], "event_summaries": []})
        self.assertEqual(model.json_calls, 1)
        self.assertEqual(model.chat_calls, 1)

    def test_summarizer_retries_schema_invalid_json(self) -> None:
        class SchemaRetryModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply(self, messages, gen):
                _ = messages, gen
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"semantic_event_operations": {}, "event_summaries": []})
                return json.dumps({"semantic_event_operations": [], "event_summaries": []})

        model = SchemaRetryModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())

        result = summarizer.consolidate_low_level_memory({"new_turns": []}, include_guga_reflection=False)

        self.assertEqual(result, {"semantic_event_operations": [], "event_summaries": []})
        self.assertEqual(model.calls, 2)
        self.assertEqual(summarizer.last_structured_attempts[0]["error_type"], "schema")


if __name__ == "__main__":
    unittest.main()
