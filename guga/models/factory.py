from __future__ import annotations

import os

from guga.models.openai_compatible_chat_model import ApiConfig, OpenAICompatibleChatModel
from guga.models.qwen_vl_chat_model import QwenVLChatModel


def create_chat_model(model_id: str, cache_dir: str | None = None):
    provider = os.environ.get("Guga_MODEL_PROVIDER", "local").strip().lower()
    if provider == "api":
        base_url = os.environ.get("Guga_API_BASE_URL", "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip()
        api_key = os.environ.get("Guga_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not base_url:
            raise ValueError("Guga_MODEL_PROVIDER=api 时必须设置 Guga_API_BASE_URL 或 OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("Guga_MODEL_PROVIDER=api 时必须设置 Guga_API_KEY 或 OPENAI_API_KEY")

        timeout_value = os.environ.get("Guga_API_TIMEOUT", "90").strip()
        try:
            timeout_seconds = max(5, int(timeout_value))
        except ValueError:
            timeout_seconds = 90

        return OpenAICompatibleChatModel(
            model_id=model_id,
            api_config=ApiConfig(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            ),
        )

    return QwenVLChatModel(model_id=model_id, cache_dir=cache_dir)

