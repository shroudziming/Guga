from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.manager import MemoryManager
from guga.types import GenerationConfig


class FakeChatModel:
    def __init__(self, capture_prompt: Callable[[str], None] | None = None) -> None:
        self.capture_prompt = capture_prompt

    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
        _ = gen
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

            manager = MemoryManager(memory_root=memory_root, debug=True, debug_sink=logs.append, top_k=2)
            manager.record_user_message("seed_session", "我在杭州工作，做后端开发")
            manager.record_assistant_message("seed_session", "收到")
            manager.finalize_turn("seed_session")

            model = FakeChatModel(capture_prompt=prompts.append)
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
            self.assertIn("[Relevant Memory]", prompts[0])
            self.assertIn("[Relevant Documents]", prompts[0])
            self.assertIn("src=", prompts[0])

            joined = "\n".join(logs)
            self.assertIn("reply_start", joined)
            self.assertIn("prepare_context_done", joined)
            self.assertIn("model_generate_start", joined)
            self.assertIn("model_generate_done", joined)
            self.assertIn("finalize_done", joined)
            self.assertIn("retrieve_done", joined)
            self.assertIn("query=", joined)
            self.assertIn("hit_ids=", joined)
            self.assertIn("latency_ms=", joined)


if __name__ == "__main__":
    unittest.main()
