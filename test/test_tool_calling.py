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
    def test_empty_tool_stream_retry_uses_agent_loop_and_records_trace(self) -> None:
        class EmptyThenToolModel:
            def __init__(self) -> None:
                self.tool_calls = 0

            def generate_reply(self, messages, gen):
                _ = messages, gen
                raise AssertionError("raw retry bypassed the tool agent loop")

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.tool_calls += 1
                if self.tool_calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_retry", name="guga_test_tool", arguments={"query": "重试"})],
                    )
                return ToolModelResponse(content="重试工具完成。", tool_calls=[])

            def generate_reply_with_tools_stream(self, messages, gen, tools, cancel_event=None):
                _ = messages, gen, tools, cancel_event
                if False:
                    yield ToolStreamText("")

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            registry = ToolRegistry(
                [
                    ToolSpec(
                        name="guga_test_tool",
                        description="test",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=lambda args: {"result": args["query"]},
                    )
                ]
            )
            session = ChatSession(
                model=EmptyThenToolModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_tool_retry",
                tool_registry=registry,
            )

            chunks = list(session.reply_stream("用工具重试"))

            self.assertEqual(chunks, ["重试工具完成。"])
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "sessions" / "sess_tool_retry.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [item["tool"] for item in rows[0]["metadata"]["turn"]["tool_interactions"]],
                ["guga_test_tool"],
            )

    def test_non_time_tool_result_with_scalar_queries_records_empty_memory_ids(self) -> None:
        class ScalarQueriesModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_scalar", name="guga_test_tool", arguments={"query": "项目"})],
                    )
                return ToolModelResponse(content="标量结果已处理。", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), enable_semantic=False)
            registry = ToolRegistry(
                [
                    ToolSpec(
                        name="guga_test_tool",
                        description="test",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=lambda args: {"queries": 42, "result": args["query"]},
                    )
                ]
            )
            session = ChatSession(
                model=ScalarQueriesModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_scalar_queries",
                tool_registry=registry,
            )

            self.assertEqual(session.reply("读取项目", finalize_memory=False), "标量结果已处理。")

            user = json.loads(
                (Path(tmp) / "sessions" / "sess_scalar_queries.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                user["metadata"]["turn"]["tool_interactions"][0]["returned_memory_ids"],
                [],
            )

    def test_time_tool_and_other_tool_share_agent_loop_and_persist_turn_trace(self) -> None:
        class MixedToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_time",
                                name="memory_time_search",
                                arguments={
                                    "time_queries": [
                                        {
                                            "original_expression": "8月20号",
                                            "start_at": "2026-08-20T00:00:00+08:00",
                                            "end_at": "2026-08-21T00:00:00+08:00",
                                            "relation": "overlap",
                                            "granularity": "day",
                                        }
                                    ]
                                },
                            ),
                            ToolCall(id="call_other", name="guga_test_tool", arguments={"query": "面试"}),
                        ],
                    )
                return ToolModelResponse(content="8月20号有面试安排。", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            manager.semantic_event_file.write_text(
                json.dumps(
                    {
                        "id": "evt_interview",
                        "type": "semantic_event",
                        "description": "参加面试",
                        "start_at": "2026-08-20T14:00:00+08:00",
                        "end_at": "2026-08-20T15:00:00+08:00",
                        "status": "active",
                        "updated_at": "2026-08-19T09:00:00+08:00",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            registry = ToolRegistry(
                [
                    ToolSpec(
                        name="guga_test_tool",
                        description="test",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=lambda args: {"result": f"handled {args['query']}"},
                    )
                ]
            )
            session = ChatSession(
                model=MixedToolModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_trace",
                tool_registry=registry,
            )

            answer = session.reply(
                "8月20号有什么面试安排",
                finalize_memory=False,
                created_at="2026-08-19T10:00:00+08:00",
            )

            self.assertEqual(answer, "8月20号有面试安排。")
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "sessions" / "sess_trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            user = next(row for row in rows if row["role"] == "user")
            turn = user["metadata"]["turn"]
            self.assertEqual(turn["assistant_message_id"], next(row["id"] for row in rows if row["role"] == "assistant"))
            self.assertEqual(turn["temporal_context"]["reference_time"], "2026-08-19T10:00:00+08:00")
            self.assertEqual(
                turn["temporal_context"]["time_queries"][0]["start_at"],
                "2026-08-20T00:00:00+08:00",
            )
            self.assertEqual(
                [item["tool"] for item in turn["tool_interactions"]],
                ["memory_time_search", "guga_test_tool"],
            )
            self.assertNotIn("result", turn["tool_interactions"][1])
            self.assertEqual(manager._turn_state["sess_trace"]["turn_id"], f"turn_{user['id']}")
            self.assertEqual(
                manager._turn_state["sess_trace"]["temporal_context"]["reference_time"],
                "2026-08-19T10:00:00+08:00",
            )
            self.assertEqual(len(manager._turn_state["sess_trace"]["tool_interactions"]), 2)

    def test_non_time_tool_continues_loop_without_temporal_context(self) -> None:
        class OtherToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_other", name="guga_test_tool", arguments={"query": "项目"})],
                    )
                return ToolModelResponse(content="工具处理完成。", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            model = OtherToolModel()
            registry = ToolRegistry(
                [
                    ToolSpec(
                        name="guga_test_tool",
                        description="test",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=lambda args: {"result": args["query"]},
                    )
                ]
            )
            session = ChatSession(
                model=model,
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=manager,
                session_id="sess_other",
                tool_registry=registry,
            )

            self.assertEqual(session.reply("读取项目", finalize_memory=False), "工具处理完成。")

            rows = (Path(tmp) / "sessions" / "sess_other.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            user = json.loads(rows[0])
            self.assertEqual(model.calls, 2)
            self.assertNotIn("temporal_context", user["metadata"]["turn"])

    def test_reused_registry_keeps_time_tool_bound_to_each_session(self) -> None:
        class TimeToolModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                self.calls += 1
                if self.calls == 1:
                    return ToolModelResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_time",
                                name="memory_time_search",
                                arguments={
                                    "time_queries": [
                                        {
                                            "original_expression": "8月20号",
                                            "start_at": "2026-08-20T00:00:00+08:00",
                                            "end_at": "2026-08-21T00:00:00+08:00",
                                            "relation": "overlap",
                                            "granularity": "day",
                                        }
                                    ]
                                },
                            )
                        ],
                    )
                return ToolModelResponse(content="完成", tool_calls=[])

        def make_manager(root: Path, event_id: str) -> MemoryManager:
            manager = MemoryManager(memory_root=root, model=None, enable_semantic=False)
            manager.semantic_event_file.write_text(
                json.dumps(
                    {
                        "id": event_id,
                        "type": "semantic_event",
                        "description": event_id,
                        "start_at": "2026-08-20T14:00:00+08:00",
                        "end_at": "2026-08-20T15:00:00+08:00",
                        "status": "active",
                        "updated_at": "2026-08-19T09:00:00+08:00",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return manager

        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            shared_registry = ToolRegistry()
            first = ChatSession(
                model=TimeToolModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=make_manager(Path(first_tmp), "evt_first"),
                session_id="sess_first",
                tool_registry=shared_registry,
            )
            ChatSession(
                model=TimeToolModel(),
                system_prompt="test",
                generation=GenerationConfig(),
                memory_manager=make_manager(Path(second_tmp), "evt_second"),
                session_id="sess_second",
                tool_registry=shared_registry,
            )

            self.assertEqual(first.reply("8月20号", finalize_memory=False), "完成")

            user = json.loads(
                (Path(first_tmp) / "sessions" / "sess_first.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                user["metadata"]["turn"]["tool_interactions"][0]["returned_memory_ids"],
                ["evt_first"],
            )

    def test_generation_exception_clears_active_time_tool_context(self) -> None:
        class FailingModel:
            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                raise RuntimeError("generation failed")

        session = _tool_session(FailingModel())

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            session.reply("测试异常", finalize_memory=False)

        self.assertEqual(session._active_turn_user_text, "")
        self.assertEqual(session._active_turn_reference_time, "")

    def test_closing_stream_clears_active_time_tool_context(self) -> None:
        class StreamingModel:
            def generate_reply_with_tools(self, messages, gen, tools):
                _ = messages, gen, tools
                return ToolModelResponse(content="fallback", tool_calls=[])

            def generate_reply_with_tools_stream(self, messages, gen, tools, cancel_event=None):
                _ = messages, gen, tools, cancel_event
                yield ToolStreamText("部分回复")
                yield ToolStreamText("不应消费")

        session = _tool_session(StreamingModel())
        stream = session.reply_stream("测试取消")

        self.assertEqual(next(stream), "部分回复")
        stream.close()

        self.assertEqual(session._active_turn_user_text, "")
        self.assertEqual(session._active_turn_reference_time, "")

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
