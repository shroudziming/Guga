from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_tool_stream_order.py"
    spec = importlib.util.spec_from_file_location("check_tool_stream_order", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolStreamOrderScriptTest(unittest.TestCase):
    def test_records_content_before_tool_call_argument_chunks(self) -> None:
        module = _load_script_module()
        recorder = module.StreamOrderRecorder()

        recorder.record_payload(
            json.dumps(
                {"choices": [{"delta": {"content": "我查一下。"}, "finish_reason": None}]},
                ensure_ascii=False,
            )
        )
        recorder.record_payload(
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "guga_probe_tool",
                                            "arguments": '{"query":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        recorder.record_payload(
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"order_probe"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        recorder.record_done()

        self.assertEqual(
            recorder.lines,
            [
                "001 content: 我查一下。",
                "002 tool_call[0].id: call_1",
                "003 tool_call[0].name: guga_probe_tool",
                '004 tool_call[0].arguments += {"query":',
                '005 tool_call[0].arguments += "order_probe"}',
                "006 finish_reason: tool_calls",
                "007 done",
            ],
        )
        self.assertEqual(
            recorder.summary(),
            {
                "first_content_event": 1,
                "first_tool_call_event": 2,
                "content_before_tool_call": True,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "name": "guga_probe_tool",
                        "arguments": {"query": "order_probe"},
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
