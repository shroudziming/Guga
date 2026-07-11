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
            if self.high_decision == "no_high_level_update":
                return json.dumps(
                    {
                        "decision": "no_high_level_update",
                        "archival_updates": [],
                        "profile_updates": [],
                        "personality_insight_updates": [],
                        "reason": "No stable long-term memory found.",
                    }
                )
            return json.dumps(
                {
                    "decision": "update_high_level_memory",
                    "archival_updates": [
                        {
                            "topic": "deadline",
                            "summary": "The user has a project report deadline.",
                            "importance": 0.8,
                            "confidence": 0.88,
                            "source_message_ids": [],
                        }
                    ],
                    "profile_updates": [
                        {"summary": "The user cares about tracking deadlines."}
                    ],
                    "personality_insight_updates": [
                        {"summary": "Guga should be careful about deadline reminders."}
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
                    enable_profile_updates=False,
                    enable_personality_updates=False,
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

    def test_high_level_failure_does_not_write_partial_low_level_updates(self) -> None:
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
            self.assertEqual(len(state["sessions"]["sess_bad_high"]["pending_turns"]), 1)
            self.assertFalse((Path(tmp) / "semantic_events.jsonl").exists())
            self.assertFalse((Path(tmp) / "event_summaries.jsonl").exists())
            self.assertFalse((Path(tmp) / "archival_memory.jsonl").exists())

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


if __name__ == "__main__":
    unittest.main()
