from __future__ import annotations

from collections.abc import MutableMapping


def configure_voice_tool_mode(env: MutableMapping[str, str]) -> bool:
    """Configure ChatSession tool use for latency-sensitive voice chat.

    Voice chat defaults to tools off for the lowest first-audio latency. When
    explicitly enabled, streaming depends on the chat model's tool-stream API
    support and on whether the upstream model emits text before tool calls.
    """

    if _env_bool(env.get("GUGA_VOICE_WITH_TOOLS", "")):
        return True

    env["Guga_MAX_TOOL_ROUNDS"] = "0"
    return False


def _env_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}
