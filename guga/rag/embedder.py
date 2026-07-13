from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from math import sqrt
from typing import Protocol


class BaseEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class HashingEmbedder:
    dim: int = 128

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._encode_one(text))
        return vectors

    def _encode_one(self, text: str) -> list[float]:
        values = [0.0 for _ in range(self.dim)]
        compact = text.strip()
        if not compact:
            return values

        grams = self._char_ngrams(compact)
        for token in grams:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dim
            values[index] += 1.0

        norm = sqrt(sum(item * item for item in values))
        if norm <= 0:
            return values
        return [item / norm for item in values]

    def _char_ngrams(self, text: str) -> list[str]:
        if len(text) < 2:
            return [text]
        tokens: list[str] = []
        for size in (2, 3):
            for idx in range(0, max(0, len(text) - size + 1)):
                tokens.append(text[idx : idx + size])
        return tokens[:256]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            model = self._load_model()
            vectors = model.encode(texts, normalize_embeddings=True)
        except Exception as exc:
            raise RuntimeError(f"failed to load or run embedding model {self.model_name}: {exc}") from exc
        return [list(map(float, row)) for row in vectors]

    def _load_model(self):
        if self._model is not None:
            return self._model

        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        return self._model


def build_embedder(model_name: str) -> BaseEmbedder:
    return SentenceTransformerEmbedder(model_name=model_name)
