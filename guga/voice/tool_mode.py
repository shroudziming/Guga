from __future__ import annotations

from collections.abc import MutableMapping


def configure_voice_tool_mode(env: MutableMapping[str, str]) -> bool:
    """Configure ChatSession tool use for latency-sensitive voice chat.

    Voice chat defaults to true text streaming. Guga's current tool-capable path
    is non-streaming, so it is only kept when explicitly requested.
    """

    if _env_bool(env.get("GUGA_VOICE_WITH_TOOLS", "")):
        return True

    env["Guga_MAX_TOOL_ROUNDS"] = "0"
    return False


def _env_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}
