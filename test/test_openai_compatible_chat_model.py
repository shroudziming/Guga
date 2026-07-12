from __future__ import annotations

import json
from threading import Event
import unittest
from unittest.mock import patch

from guga.models.openai_compatible_chat_model import ApiConfig, OpenAICompatibleChatModel
from guga.tools import ToolCall, ToolStreamText, ToolStreamToolCalls
from guga.types import GenerationConfig


class OpenAICompatibleChatModelTest(unittest.TestCase):
    def test_structured_reply_preserves_finish_reason_and_usage(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )
        response = {
            "choices": [
                {
                    "message": {"content": '{"ok": true}'},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with patch.object(model, "_post_chat_completions", return_value=response):
            reply = model.generate_structured_reply(
                [{"role": "user", "content": "return json"}],
                GenerationConfig(max_new_tokens=128),
            )

        self.assertEqual(reply.content, '{"ok": true}')
        self.assertEqual(reply.finish_reason, "length")
        self.assertEqual(reply.response_mode, "json_object")
        self.assertEqual(reply.output_chars, 12)
        self.assertEqual(reply.usage["total_tokens"], 30)

    def test_extract_text_content_ignores_none_stream_chunks(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )

        self.assertEqual(model._extract_text_content(None), "")

    def test_streams_tool_text_chunks_immediately(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )

        with patch("guga.models.openai_compatible_chat_model.request.urlopen") as urlopen:
            urlopen.return_value = _FakeSseResponse(
                [
                    _sse({"choices": [{"delta": {"content": "我查"}, "finish_reason": None}]}),
                    _sse({"choices": [{"delta": {"content": "一下。"}, "finish_reason": "stop"}]}),
                    "data: [DONE]\n\n",
                ]
            )

            events = list(
                model.generate_reply_with_tools_stream(
                    [{"role": "user", "content": "hi"}],
                    GenerationConfig(),
                    tools=[],
                )
            )

        self.assertEqual(events, [ToolStreamText("我查"), ToolStreamText("一下。")])

    def test_yields_content_before_accumulated_tool_calls(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )

        with patch("guga.models.openai_compatible_chat_model.request.urlopen") as urlopen:
            urlopen.return_value = _FakeSseResponse(
                [
                    _sse({"choices": [{"delta": {"content": "我查一下。"}, "finish_reason": None}]}),
                    _sse(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {"name": "guga_list_dir", "arguments": '{"path":'},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    ),
                    _sse(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": '"."}'},
                                            }
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        }
                    ),
                    "data: [DONE]\n\n",
                ]
            )

            stream = model.generate_reply_with_tools_stream(
                [{"role": "user", "content": "hi"}],
                GenerationConfig(),
                tools=[],
            )
            first_event = next(stream)
            remaining_events = list(stream)

        self.assertEqual(first_event, ToolStreamText("我查一下。"))
        self.assertEqual(
            remaining_events,
            [ToolStreamToolCalls([ToolCall(id="call_1", name="guga_list_dir", arguments={"path": "."})])],
        )

    def test_cancel_stops_streaming_tool_events(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )
        cancel_event = Event()
        cancel_event.set()

        with patch("guga.models.openai_compatible_chat_model.request.urlopen") as urlopen:
            urlopen.return_value = _FakeSseResponse(
                [
                    _sse({"choices": [{"delta": {"content": "不会出现"}, "finish_reason": None}]}),
                    "data: [DONE]\n\n",
                ]
            )

            events = list(
                model.generate_reply_with_tools_stream(
                    [{"role": "user", "content": "hi"}],
                    GenerationConfig(),
                    tools=[],
                    cancel_event=cancel_event,
                )
            )

        self.assertEqual(events, [])


class _FakeSseResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]
        self.headers = {"Content-Type": "text/event-stream"}

    def __enter__(self) -> "_FakeSseResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb

    def __iter__(self):
        return iter(self._lines)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    unittest.main()
