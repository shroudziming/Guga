from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    split_reason: str


class TextSentenceBuffer:
    """Collect streaming text chunks and emit complete short sentences."""

    def __init__(
        self,
        *,
        boundaries: str = "。！？；.!?;\n",
        max_chars: int = 80,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.boundaries = set(boundaries)
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        return [segment.text for segment in self.feed_segments(chunk)]

    def feed_segments(self, chunk: str) -> list[TextSegment]:
        if not chunk:
            return []

        self._buffer += chunk
        segments: list[TextSegment] = []

        while self._buffer:
            boundary_index = self._first_boundary_index()
            if boundary_index >= 0:
                end = boundary_index + 1
                boundary = self._buffer[boundary_index]
                sentence = self._buffer[:end].strip()
                self._buffer = self._buffer[end:]
                if sentence:
                    segments.append(TextSegment(text=sentence, split_reason=f"boundary:{boundary}"))
                continue

            if len(self._buffer) >= self.max_chars:
                sentence = self._buffer[: self.max_chars].strip()
                self._buffer = self._buffer[self.max_chars :]
                if sentence:
                    segments.append(TextSegment(text=sentence, split_reason=f"max_chars:{self.max_chars}"))
                continue

            break

        return segments

    def flush(self) -> list[str]:
        return [segment.text for segment in self.flush_segments()]

    def flush_segments(self) -> list[TextSegment]:
        sentence = self._buffer.strip()
        self._buffer = ""
        return [TextSegment(text=sentence, split_reason="flush")] if sentence else []

    def _first_boundary_index(self) -> int:
        indexes = [self._buffer.find(char) for char in self.boundaries]
        indexes = [index for index in indexes if index >= 0]
        return min(indexes) if indexes else -1


def sentence_buffer_from_env(env: Mapping[str, str]) -> TextSentenceBuffer:
    return TextSentenceBuffer(max_chars=_env_int(env.get("GUGA_TTS_SENTENCE_MAX_CHARS", ""), 16))


def _env_int(raw: str, default: int) -> int:
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(200, max(8, value))
