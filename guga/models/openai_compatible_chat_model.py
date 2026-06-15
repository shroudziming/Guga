from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event
from urllib import error, request

from guga.types import GenerationConfig
from guga.tools import ToolCall, ToolModelResponse, parse_tool_arguments


@dataclass
class ApiConfig:
    base_url: str
    api_key: str
    timeout_seconds: int = 90


class OpenAICompatibleChatModel:
    """Use OpenAI-compatible chat-completions APIs with the local chat interface."""

    def __init__(self, model_id: str, api_config: ApiConfig) -> None:
        self.model_id = model_id
        self.api_config = api_config

    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_new_tokens,
        }
        response = self._post_chat_completions(payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"API 返回异常，缺少 choices: {response}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        return self._extract_text_content(content).strip()

    def generate_reply_with_tools(
        self,
        messages: list[dict],
        gen: GenerationConfig,
        tools: list[dict],
    ) -> ToolModelResponse:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_new_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        response = self._post_chat_completions(payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"API 返回异常，缺少 choices: {response}")

        message = choices[0].get("message", {})
        content = self._extract_text_content(message.get("content", "")).strip()
        tool_calls = []
        for index, raw_call in enumerate(message.get("tool_calls", []) or []):
            function = raw_call.get("function", {}) if isinstance(raw_call, dict) else {}
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            call_id = str(raw_call.get("id") or f"tool_call_{index}")
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=parse_tool_arguments(function.get("arguments", "")),
                )
            )
        return ToolModelResponse(content=content, tool_calls=tool_calls)

    def generate_reply_stream(
        self,
        messages: list[dict[str, str]],
        gen: GenerationConfig,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_new_tokens,
            "stream": True,
        }

        endpoint = self._resolve_endpoint(self.api_config.base_url)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_config.api_key}")

        try:
            with request.urlopen(req, timeout=self.api_config.timeout_seconds) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/event-stream" not in content_type:
                    text = resp.read().decode("utf-8", errors="ignore")
                    data = json.loads(text)
                    choices = data.get("choices", [])
                    if not choices:
                        return
                    message = choices[0].get("message", {})
                    content = self._extract_text_content(message.get("content", ""))
                    if content:
                        yield content
                    return

                for raw_line in resp:
                    if cancel_event is not None and cancel_event.is_set():
                        return

                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    payload_text = line[5:].strip()
                    if payload_text == "[DONE]":
                        return

                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue

                    choices = event.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    piece = self._extract_text_content(delta.get("content", ""))
                    if piece:
                        yield piece
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API 请求失败: HTTP {exc.code}, detail={detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"API 连接失败: {exc}") from exc

    def _extract_text_content(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "".join(text_parts)
        return str(content)

    def _post_chat_completions(self, payload: dict) -> dict:
        endpoint = self._resolve_endpoint(self.api_config.base_url)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_config.api_key}")

        try:
            with request.urlopen(req, timeout=self.api_config.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API 请求失败: HTTP {exc.code}, detail={detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"API 连接失败: {exc}") from exc

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API 返回非 JSON: {text[:300]}") from exc

    def _resolve_endpoint(self, base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"
