from __future__ import annotations

import json
from pathlib import Path

from guga.rag.schemas import DocumentChunk


class IncompatibleIndexError(RuntimeError):
    """Raised when persisted vectors were built by a different embedder."""


class VectorStore:
    def __init__(self, index_dir: Path, embedding_model: str = "") -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
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
        except Exception as exc:
            raise RuntimeError("FAISS is required for semantic retrieval") from exc

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

    def prune_memory_sources(self, valid_source_ids: set[str]) -> int:
        kept_chunks: list[DocumentChunk] = []
        kept_vectors: list[list[float]] = []
        removed = 0
        for chunk, vector in zip(self.chunks, self.vectors):
            if chunk.source_type == "memory" and chunk.source_id not in valid_source_ids:
                removed += 1
                continue
            kept_chunks.append(chunk)
            kept_vectors.append(vector)
        if removed:
            self.chunks = kept_chunks
            self.vectors = kept_vectors
            self.dim = len(kept_vectors[0]) if kept_vectors else 0
            self._rebuild_index()
        return removed

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
        if len(query_vec) != self.dim:
            raise IncompatibleIndexError(
                f"query dimension {len(query_vec)} does not match persisted index dimension {self.dim}"
            )

        import numpy as np

        if source_type:
            if source_type not in self._typed_indexes:
                return []
            typed_index, row_ids = self._typed_indexes[source_type]
            q = np.array([query_vec], dtype="float32")
            scores, indices = typed_index.search(q, min(top_k, len(row_ids)))
            results: list[tuple[int, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if int(idx) < 0:
                    continue
                results.append((row_ids[int(idx)], float(score)))
            return results

        if self._index is None:
            return []
        q = np.array([query_vec], dtype="float32")
        scores, indices = self._index.search(q, min(top_k, len(self.chunks)))
        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if int(idx) < 0:
                continue
            results.append((int(idx), float(score)))
        return results

    def load(self) -> None:
        chunks_file = self.index_dir / "chunks.jsonl"
        vectors_file = self.index_dir / "vectors.json"
        metadata_file = self.index_dir / "index_meta.json"
        if not chunks_file.exists() or not vectors_file.exists():
            self.chunks = []
            self.vectors = []
            self.dim = 0
            self._index = None
            return
        if not metadata_file.exists():
            raise IncompatibleIndexError("persisted index has no embedding model metadata")

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        persisted_model = str(metadata.get("embedding_model", ""))
        if self.embedding_model and persisted_model != self.embedding_model:
            raise IncompatibleIndexError(
                f"persisted index model {persisted_model!r} does not match configured model {self.embedding_model!r}"
            )

        chunks: list[DocumentChunk] = []
        for line in chunks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(DocumentChunk.from_dict(payload))

        vectors = [[float(item) for item in row] for row in json.loads(vectors_file.read_text(encoding="utf-8"))]
        dimension = len(vectors[0]) if vectors else 0
        if int(metadata.get("dimension", -1)) != dimension:
            raise IncompatibleIndexError("persisted index metadata dimension does not match vectors")
        self.rebuild(chunks=chunks, vectors=vectors)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = self.index_dir / "chunks.jsonl"
        vectors_file = self.index_dir / "vectors.json"
        metadata_file = self.index_dir / "index_meta.json"

        chunks_file.write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in self.chunks) + ("\n" if self.chunks else ""),
            encoding="utf-8",
        )
        vectors_file.write_text(json.dumps(self.vectors, ensure_ascii=False), encoding="utf-8")
        metadata_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "embedding_model": self.embedding_model,
                    "dimension": self.dim,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
