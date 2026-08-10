from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.chat.session import ChatSession
from guga.memory.manager import MemoryManager
from guga.tools import (
    ToolCall,
    ToolModelResponse,
    ToolRegistry,
    ToolSpec,
    ToolStreamText,
    ToolStreamToolCalls,
    conversation_tool_registry,
    default_tool_registry,
)
from guga.types import GenerationConfig
from guga.workspace import WorkspaceContext


class ToolCallingTest(unittest.TestCase):
    def test_registry_filters_schemas_without_hiding_registered_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = default_tool_registry(Path(tmp))

            schemas = registry.openai_tools(names={"guga_read_file"})

        self.assertIn("guga_run_command", registry.names())
        self.assertIn("guga_workspace", registry.names())
        self.assertEqual([item["function"]["name"] for item in schemas], ["guga_read_file"])

    def test_operational_tools_follow_confirmed_session_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_root = root / "default"
            other_root = root / "other"
            default_root.mkdir()
            other_root.mkdir()
            (default_root / "note.txt").write_text("default", encoding="utf-8")
            (other_root / "note.txt").write_text("other", encoding="utf-8")
            workspace = WorkspaceContext(default_root)
            registry = default_tool_registry(workspace=workspace)

            denied = registry.execute(
                ToolCall(id="read-before-inspect", name="guga_read_file", arguments={"path": "note.txt"})
            )
            inspected = registry.execute(
                ToolCall(id="inspect-default", name="guga_workspace", arguments={"action": "inspect"})
            )
            first_read = registry.execute(
                ToolCall(id="read-default", name="guga_read_file", arguments={"path": "note.txt"})
            )
            switched = registry.execute(
                ToolCall(
                    id="switch",
                    name="guga_workspace",
                    arguments={"action": "set", "path": str(other_root)},
                )
            )
            denied_after_switch = registry.execute(
                ToolCall(id="read-after-switch", name="guga_read_file", arguments={"path": "note.txt"})
            )
            registry.execute(
                ToolCall(id="inspect-other", name="guga_workspace", arguments={"action": "inspect"})
            )
            second_read = registry.execute(
                ToolCall(id="read-other", name="guga_read_file", arguments={"path": "note.txt"})
            )

        self.assertFalse(denied["ok"])
        self.assertIn("inspect", denied["error"])
        self.assertTrue(inspected["ok"])
        self.assertEqual(first_read["content"], "default")
        self.assertFalse(switched["confirmed"])
        self.assertFalse(denied_after_switch["ok"])
        self.assertIn("inspect", denied_after_switch["error"])
        self.assertEqual(second_read["content"], "other")

    def test_conversation_registry_excludes_operational_tools(self) -> None:
        registry = conversation_tool_registry()

        self.assertEqual(registry.names(), {"guga_parse_time"})

    def test_chat_session_defaults_to_conversation_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            session = ChatSession(
                model=object(),
                system_prompt="base",
                generation=GenerationConfig(),
                memory_manager=memory_manager,
            )

        self.assertEqual({"guga_parse_time"}, session.tool_registry.names())

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
            memory_manager.wait_for_background_tasks(timeout=3)

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
