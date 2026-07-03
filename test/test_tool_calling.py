from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.manager import MemoryManager
from guga.tools import ToolCall, ToolModelResponse, ToolRegistry, ToolSpec, ToolStreamText, ToolStreamToolCalls
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

    def test_reply_stream_with_tools_yields_transition_then_final_answer(self) -> None:
        class StreamingToolModel:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[dict]] = []

            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "fallback"

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                return ToolModelResponse(content="fallback", tool_calls=[])

            def generate_reply_with_tools_stream(self, messages, gen, tools, cancel_event=None):
                _ = gen, tools, cancel_event
                self.calls += 1
                self.messages_seen.append([dict(message) for message in messages])
                if self.calls == 1:
                    yield ToolStreamText("我查一下。")
                    yield ToolStreamToolCalls(
                        [ToolCall(id="call_1", name="guga_test_tool", arguments={"query": "上周"})]
                    )
                    return
                yield ToolStreamText("查到了：handled 上周")

        model = StreamingToolModel()
        session = _tool_session(model)

        chunks = list(session.reply_stream("测试工具调用"))

        self.assertEqual(chunks, ["我查一下。", "查到了：handled 上周"])
        self.assertFalse(any('"result"' in chunk for chunk in chunks))
        second_round_messages = model.messages_seen[1]
        assistant_message = next(message for message in second_round_messages if message.get("role") == "assistant")
        self.assertEqual(assistant_message["content"], "我查一下。")
        self.assertEqual(assistant_message["tool_calls"][0]["function"]["name"], "guga_test_tool")
        tool_message = next(message for message in second_round_messages if message.get("role") == "tool")
        self.assertIn("handled 上周", tool_message["content"])

    def test_reply_stream_with_tools_no_content_before_tool_has_no_early_chunk(self) -> None:
        class StreamingToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "fallback"

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                return ToolModelResponse(content="fallback", tool_calls=[])

            def generate_reply_with_tools_stream(self, messages, gen, tools, cancel_event=None):
                _ = messages, gen, tools, cancel_event
                self.calls += 1
                if self.calls == 1:
                    yield ToolStreamToolCalls(
                        [ToolCall(id="call_1", name="guga_test_tool", arguments={"query": "上周"})]
                    )
                    return
                yield ToolStreamText("查到了：handled 上周")

        chunks = list(_tool_session(StreamingToolModel()).reply_stream("测试工具调用"))

        self.assertEqual(chunks, ["查到了：handled 上周"])

    def test_reply_stream_with_tools_falls_back_when_stream_method_missing(self) -> None:
        class NonStreamingToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "fallback"

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="guga_test_tool", arguments={"query": "上周"})],
                    )
                return ToolModelResponse(content="我想起来了：handled 上周", tool_calls=[])

        chunks = list(_tool_session(NonStreamingToolModel()).reply_stream("测试工具调用"))

        self.assertEqual(chunks, ["我想起来了：handled 上周"])


def _tool_session(model) -> ChatSession:
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

    memory_root = Path(tempfile.mkdtemp())
    memory_manager = MemoryManager(memory_root=memory_root, enable_semantic=False)
    return ChatSession(
        model=model,
        system_prompt="base",
        generation=GenerationConfig(),
        memory_manager=memory_manager,
        tool_registry=registry,
    )


if __name__ == "__main__":
    unittest.main()
