from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.manager import MemoryManager
from guga.tools import ToolCall, ToolModelResponse, ToolRegistry, ToolSpec
from guga.types import GenerationConfig


class ToolCallingTest(unittest.TestCase):
    def test_reply_executes_tool_and_continues_generation(self) -> None:
        class FakeToolModel:
            def __init__(self) -> None:
                self.calls = 0
                self.seen_tool_result = False

            def generate_reply(self, messages, gen):
                return "fallback"

            def generate_reply_with_tools(self, messages, gen, tools):
                self.calls += 1
                self.seen_tool_result = any(message.get("role") == "tool" for message in messages)
                if not self.seen_tool_result:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="guga_test_tool", arguments={"query": "上周"})],
                    )
                tool_payload = next(message["content"] for message in messages if message.get("role") == "tool")
                data = json.loads(tool_payload)
                return ToolModelResponse(content=f"我想起来了：{data['result']}", tool_calls=[])

        def handler(args):
            return {"result": f"handled {args['query']}"}

        registry = ToolRegistry(
            [
                ToolSpec(
                    name="guga_test_tool",
                    description="Test tool",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    handler=handler,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            logs: list[str] = []
            memory_manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False, debug=True, debug_sink=logs.append)
            session = ChatSession(
                model=FakeToolModel(),
                system_prompt="base",
                generation=GenerationConfig(),
                memory_manager=memory_manager,
                debug=True,
                debug_sink=logs.append,
                tool_registry=registry,
            )

            answer = session.reply("测试工具调用")

        self.assertEqual(answer, "我想起来了：handled 上周")
        self.assertTrue(any("tool_call round=1 name=guga_test_tool ok=True" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
