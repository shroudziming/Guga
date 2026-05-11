from __future__ import annotations

import json
from pathlib import Path

from guga.rag.schemas import DocumentChunk


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chunks: list[DocumentChunk] = []
        self.vectors: list[list[float]] = []
        self.dim = 0
        self._faiss = None
        self._index = None
        self._typed_indexes: dict[str, tuple[object, list[int]]] = {}
        self._load_faiss()

    def _load_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            self._faiss = None

    def has_persisted_index(self) -> bool:
        return (self.index_dir / "chunks.jsonl").exists() and (self.index_dir / "vectors.json").exists()

    def rebuild(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.dim = len(vectors[0]) if vectors else 0
        self._rebuild_index()

    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if self.dim == 0 and vectors:
            self.dim = len(vectors[0])
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)
        self._rebuild_index()

    def replace_by_source_id(self, source_id: str, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        kept_chunks: list[DocumentChunk] = []
        kept_vectors: list[list[float]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if chunk.source_id == source_id:
                continue
            kept_chunks.append(chunk)
            kept_vectors.append(vector)
        self.chunks = kept_chunks
        self.vectors = kept_vectors
        if self.dim == 0 and vectors:
            self.dim = len(vectors[0])
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = None
        self._typed_indexes = {}
        if not self._faiss or not self.vectors:
            return

        import numpy as np

        index = self._faiss.IndexFlatIP(self.dim)
        matrix = np.array(self.vectors, dtype="float32")
        index.add(matrix)
        self._index = index

        rows_by_type: dict[str, list[int]] = {}
        for idx, chunk in enumerate(self.chunks):
            rows_by_type.setdefault(chunk.source_type, []).append(idx)

        for source_type, row_ids in rows_by_type.items():
            if not source_type or not row_ids:
                continue
            typed_index = self._faiss.IndexFlatIP(self.dim)
            typed_matrix = np.array([self.vectors[idx] for idx in row_ids], dtype="float32")
            typed_index.add(typed_matrix)
            self._typed_indexes[source_type] = (typed_index, row_ids)

    def search(self, query_vec: list[float], top_k: int, source_type: str = "") -> list[tuple[int, float]]:
        if top_k <= 0 or not query_vec or not self.chunks:
            return []

        if self._faiss is not None:
            import numpy as np

            if source_type and source_type in self._typed_indexes:
                typed_index, row_ids = self._typed_indexes[source_type]
                q = np.array([query_vec], dtype="float32")
                scores, indices = typed_index.search(q, min(top_k, len(row_ids)))
                results: list[tuple[int, float]] = []
                for idx, score in zip(indices[0], scores[0]):
                    if int(idx) < 0:
                        continue
                    results.append((row_ids[int(idx)], float(score)))
                return results

            if self._index is not None and not source_type:
                q = np.array([query_vec], dtype="float32")
                scores, indices = self._index.search(q, min(top_k, len(self.chunks)))
                results: list[tuple[int, float]] = []
                for idx, score in zip(indices[0], scores[0]):
                    if int(idx) < 0:
                        continue
                    results.append((int(idx), float(score)))
                return results

        candidates: list[tuple[int, float]] = []
        for idx, (chunk, vec) in enumerate(zip(self.chunks, self.vectors)):
            if source_type and chunk.source_type != source_type:
                continue
            candidates.append((idx, _dot(query_vec, vec)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[: min(top_k, len(candidates))]

    def load(self) -> None:
        chunks_file = self.index_dir / "chunks.jsonl"
        vectors_file = self.index_dir / "vectors.json"
        if not chunks_file.exists() or not vectors_file.exists():
            self.chunks = []
            self.vectors = []
            self.dim = 0
            self._index = None
            return

        chunks: list[DocumentChunk] = []
        for line in chunks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(DocumentChunk.from_dict(payload))

        vectors = json.loads(vectors_file.read_text(encoding="utf-8"))
        self.rebuild(chunks=chunks, vectors=[[float(item) for item in row] for row in vectors])

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = self.index_dir / "chunks.jsonl"
        vectors_file = self.index_dir / "vectors.json"

        chunks_file.write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in self.chunks) + ("\n" if self.chunks else ""),
            encoding="utf-8",
        )
        vectors_file.write_text(json.dumps(self.vectors, ensure_ascii=False), encoding="utf-8")
