from __future__ import annotations

from typing import Protocol

from guga.types import GenerationConfig


class ChatModel(Protocol):
    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:  # pragma: no cover
        ...
