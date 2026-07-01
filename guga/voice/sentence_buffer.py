from __future__ import annotations


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
        if not chunk:
            return []

        self._buffer += chunk
        sentences: list[str] = []

        while self._buffer:
            boundary_index = self._first_boundary_index()
            if boundary_index >= 0:
                end = boundary_index + 1
                sentence = self._buffer[:end].strip()
                self._buffer = self._buffer[end:]
                if sentence:
                    sentences.append(sentence)
                continue

            if len(self._buffer) >= self.max_chars:
                sentence = self._buffer[: self.max_chars].strip()
                self._buffer = self._buffer[self.max_chars :]
                if sentence:
                    sentences.append(sentence)
                continue

            break

        return sentences

    def flush(self) -> list[str]:
        sentence = self._buffer.strip()
        self._buffer = ""
        return [sentence] if sentence else []

    def _first_boundary_index(self) -> int:
        indexes = [self._buffer.find(char) for char in self.boundaries]
        indexes = [index for index in indexes if index >= 0]
        return min(indexes) if indexes else -1
