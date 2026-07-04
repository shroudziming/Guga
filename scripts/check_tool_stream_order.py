from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.config import DEFAULT_MODEL_ID


DEFAULT_PROMPT = (
    "请先输出一句话：我查一下。然后调用工具 guga_probe_tool，参数 query 填 order_probe。"
    "不要直接编造工具结果。"
)


class StreamOrderRecorder:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._sequence = 0
        self._first_content_event: int | None = None
        self._first_tool_call_event: int | None = None
        self._tool_call_parts: dict[int, dict[str, str]] = {}

    def record_payload(self, payload_text: str) -> None:
        if payload_text == "[DONE]":
            self.record_done()
            return

        try:
            event = json.loads(payload_text)
        except json.JSONDecodeError:
            self._append(f"invalid_json: {payload_text[:200]}")
            return

        choices = event.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}

        content = _extract_text_content(delta.get("content", ""))
        if content:
            self._append(f"content: {content}")
            if self._first_content_event is None:
                self._first_content_event = self._sequence

        for raw_call in delta.get("tool_calls", []) or []:
            if not isinstance(raw_call, dict):
                continue
            index = _tool_call_index(raw_call, self._tool_call_parts)
            part = self._tool_call_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})

            call_id = raw_call.get("id")
            if call_id:
                value = str(call_id)
                part["id"] += value
                self._append_tool_event(f"tool_call[{index}].id: {value}")

            function = raw_call.get("function", {})
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if name:
                value = str(name)
                part["name"] += value
                self._append_tool_event(f"tool_call[{index}].name: {value}")

            arguments = function.get("arguments")
            if arguments:
                value = str(arguments)
                part["arguments"] += value
                self._append_tool_event(f"tool_call[{index}].arguments += {value}")

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self._append(f"finish_reason: {finish_reason}")

    def record_done(self) -> None:
        self._append("done")

    def summary(self) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self._tool_call_parts):
            part = self._tool_call_parts[index]
            raw_arguments = part.get("arguments", "")
            try:
                arguments: Any = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = raw_arguments
            tool_calls.append(
                {
                    "index": index,
                    "id": part.get("id", ""),
                    "name": part.get("name", ""),
                    "arguments": arguments,
                }
            )

        return {
            "first_content_event": self._first_content_event,
            "first_tool_call_event": self._first_tool_call_event,
            "content_before_tool_call": (
                self._first_content_event is not None
                and self._first_tool_call_event is not None
                and self._first_content_event < self._first_tool_call_event
            ),
            "tool_calls": tool_calls,
        }

    def _append(self, message: str) -> None:
        self._sequence += 1
        self.lines.append(f"{self._sequence:03d} {message}")

    def _append_tool_event(self, message: str) -> None:
        self._append(message)
        if self._first_tool_call_event is None:
            self._first_tool_call_event = self._sequence


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extract_text_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _tool_call_index(raw_call: dict[str, Any], parts: dict[int, dict[str, str]]) -> int:
    try:
        return int(raw_call.get("index", len(parts)))
    except (TypeError, ValueError):
        return len(parts)


def _resolve_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _probe_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "guga_probe_tool",
            "description": "A harmless diagnostic tool used only to inspect streaming tool-call order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Diagnostic query text.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def _iter_sse_payloads(response) -> Any:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        yield line[5:].strip()


def _build_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是工具流顺序诊断模型。按用户要求输出文本并调用工具。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "tools": [_probe_tool_schema()],
        "tool_choice": "auto",
        "stream": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检测 OpenAI-compatible 模型 stream=True + tools 的返回顺序。")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="发给模型的诊断提示词。")
    parser.add_argument("--model", default=None, help="覆盖 Guga_MODEL_ID。")
    parser.add_argument("--base-url", default=None, help="覆盖 Guga_API_BASE_URL/OPENAI_BASE_URL。")
    parser.add_argument("--timeout", type=int, default=None, help="HTTP 超时时间，单位秒。")
    parser.add_argument("--max-tokens", type=int, default=256, help="本次诊断请求的 max_tokens。")
    parser.add_argument("--show-raw", action="store_true", help="同时打印原始 SSE JSON。")
    args = parser.parse_args(argv)

    _load_env_file()

    model = args.model or os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID)
    base_url = (
        args.base_url
        or os.environ.get("Guga_API_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )
    api_key = os.environ.get("Guga_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    timeout = args.timeout or _env_int("Guga_API_TIMEOUT", 90, minimum=5)

    if not base_url:
        print("缺少 Guga_API_BASE_URL 或 OPENAI_BASE_URL。", file=sys.stderr)
        return 2
    if not api_key:
        print("缺少 Guga_API_KEY 或 OPENAI_API_KEY。", file=sys.stderr)
        return 2

    endpoint = _resolve_endpoint(base_url)
    payload = _build_payload(model=model, prompt=args.prompt, max_tokens=max(1, args.max_tokens))
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    print(f"model={model}")
    print(f"endpoint={endpoint}")
    print(f"prompt={args.prompt}")
    print("tool=guga_probe_tool")
    print("")

    recorder = StreamOrderRecorder()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" not in content_type:
                text = resp.read().decode("utf-8", errors="ignore")
                print(f"非 SSE 响应: content_type={content_type}")
                print(text[:1000])
                return 1

            for payload_text in _iter_sse_payloads(resp):
                if args.show_raw:
                    print(f"raw: {payload_text}")
                before = len(recorder.lines)
                recorder.record_payload(payload_text)
                for line in recorder.lines[before:]:
                    print(line)
                if payload_text == "[DONE]":
                    break
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"API 请求失败: HTTP {exc.code}", file=sys.stderr)
        print(detail[:1000], file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"API 连接失败: {exc}", file=sys.stderr)
        return 1

    print("")
    print("summary:")
    print(json.dumps(recorder.summary(), ensure_ascii=False, indent=2))
    return 0


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


if __name__ == "__main__":
    raise SystemExit(main())
