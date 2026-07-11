from __future__ import annotations

import json
import tempfile
from time import perf_counter, sleep
import unittest
from collections.abc import Callable
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.types import GenerationConfig


class FakeChatModel:
    def __init__(self, capture_prompt: Callable[[str], None] | None = None) -> None:
        self.capture_prompt = capture_prompt

    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
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
                            "entity": "work context",
                            "description": "用户在杭州工作，做后端开发。",
                            "time_expression": "",
                            "end_unknown": True,
                            "source_message_ids": [],
                        }
                    ],
                    "event_summaries": [
                        {
                            "summary": "用户在杭州工作，做后端开发。",
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
                    "archival_operations": [
                        {
                            "topic": "work",
                            "summary": "用户在杭州工作，做后端开发",
                            "importance": 0.8,
                            "confidence": 0.9,
                            "source_event_ids": ["evt_work_context"],
                        }
                    ],
                    "user_model_operations": [],
                    "reason": "work context",
                },
                ensure_ascii=False,
            )
        if "Memory route classifier" in prompt:
            return json.dumps(
                [
                    {
                        "target": "archival_memory",
                        "label": "stable_context",
                        "content": "用户在杭州工作，做后端开发",
                        "topic": "work",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                    {
                        "target": "personality_insight",
                        "label": "stable_context",
                        "content": "用户在杭州工作，做后端开发。",
                        "confidence": 0.9,
                    },
                ],
                ensure_ascii=False,
            )
        if "用户画像整理器" in prompt:
            return "- 用户在杭州工作，做后端开发。"
        if "Summarize" in prompt or "summary" in prompt:
            return "- 用户在杭州工作，做后端开发。"
        system_prompt = messages[0]["content"]
        if self.capture_prompt is not None:
            self.capture_prompt(system_prompt)
        return "这是测试回复"


class ChatSessionRagFlowTest(unittest.TestCase):
    def test_reply_injects_memory_and_emits_debug_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_root = Path(tmp_dir)
            logs: list[str] = []
            prompts: list[str] = []

            model = FakeChatModel(capture_prompt=prompts.append)
            manager = MemoryManager(
                memory_root=memory_root,
                model=model,
                debug=True,
                debug_sink=logs.append,
                top_k=2,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
            manager.record_user_message("seed_session", "我在杭州工作，做后端开发")
            manager.record_assistant_message("seed_session", "收到")
            manager.finalize_turn("seed_session")

            session = ChatSession(
                model=model,
                system_prompt="你是陪伴助手",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_main",
                debug=True,
                debug_sink=logs.append,
            )

            answer = session.reply("你记得我在杭州做什么吗")

            self.assertEqual(answer, "这是测试回复")
            self.assertTrue(prompts)
            self.assertIn("[Relevant Conversation Memories]", prompts[0])
            self.assertNotIn("[Relevant Documents]", prompts[0])
            self.assertIn("src=", prompts[0])

            joined = "\n".join(logs)
            self.assertIn("reply_start", joined)
            self.assertIn("prepare_context_done", joined)
            self.assertIn("model_generate_start", joined)
            self.assertIn("model_generate_done", joined)
            self.assertIn("finalize_queued", joined)
            self.assertIn("retrieve_done", joined)
            self.assertIn("query=", joined)
            self.assertIn("hit_ids=", joined)
            self.assertIn("latency_ms=", joined)
            manager.wait_for_background_tasks(timeout=3)

    def test_reply_queues_memory_finalization_without_blocking_user_response(self) -> None:
        class SlowMemoryModel(FakeChatModel):
            def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
                prompt = messages[-1]["content"]
                if "Low-level memory consolidation" in prompt or "High-level memory consolidation" in prompt:
                    sleep(1.0)
                if "Low-level memory consolidation" in prompt:
                    return json.dumps(
                        {
                            "semantic_event_operations": [
                                {
                                    "operation": "create",
                                    "event_kind": "state_change",
                                    "subject": "user",
                                    "entity": "work context",
                                    "description": "The user works on backend systems.",
                                    "time_expression": "",
                                    "end_unknown": True,
                                    "source_message_ids": [],
                                }
                            ],
                            "event_summaries": [
                                {
                                    "summary": "The user works on backend systems.",
                                    "source_message_ids": [],
                                    "confidence": 0.9,
                                }
                            ],
                        }
                    )
                if "High-level memory consolidation" in prompt:
                    return json.dumps(
                        {
                            "decision": "update_high_level_memory",
                            "archival_operations": [
                                {
                                    "topic": "work",
                                    "summary": "The user works on backend systems.",
                                    "importance": 0.8,
                                    "confidence": 0.9,
                                    "source_event_ids": ["evt_work_context"],
                                }
                            ],
                            "user_model_operations": [],
                            "reason": "work context",
                        }
                    )
                return super().generate_reply(messages, gen)

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_root = Path(tmp_dir)
            manager = MemoryManager(
                memory_root=memory_root,
                model=SlowMemoryModel(),
                enable_semantic=False,
                consolidation_config=MemoryConsolidationConfig(batch_turns=1),
            )
            session = ChatSession(
                model=SlowMemoryModel(),
                system_prompt="You are a companion assistant.",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_async",
            )

            started = perf_counter()
            answer = session.reply("I work on backend systems and care about reliability.")
            elapsed = perf_counter() - started

            self.assertEqual(answer, "这是测试回复")
            self.assertLess(elapsed, 0.5)

            manager.wait_for_background_tasks(timeout=3)
            archival = memory_root / "archival_memory.jsonl"
            self.assertTrue(archival.exists())
            self.assertIn("backend systems", archival.read_text(encoding="utf-8"))

    def test_stream_empty_content_retries_non_streaming_before_persisting(self) -> None:
        class EmptyStreamModel:
            def generate_reply_stream(self, messages: list[dict[str, str]], gen: GenerationConfig, cancel_event=None):
                _ = messages, gen, cancel_event
                if False:
                    yield ""

            def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
                _ = messages, gen
                return "retry answer"

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_root = Path(tmp_dir)
            manager = MemoryManager(memory_root=memory_root, enable_semantic=False)
            session = ChatSession(
                model=EmptyStreamModel(),
                system_prompt="You are a companion assistant.",
                generation=GenerationConfig(max_new_tokens=64),
                memory_manager=manager,
                session_id="sess_empty_stream",
            )

            chunks = list(session.reply_stream("hello"))

            self.assertEqual(chunks, ["retry answer"])
            rows = (memory_root / "sessions" / "sess_empty_stream.jsonl").read_text(encoding="utf-8")
            self.assertIn('"role": "assistant"', rows)
            self.assertIn("retry answer", rows)
            manager.wait_for_background_tasks(timeout=3)


if __name__ == "__main__":
    unittest.main()
