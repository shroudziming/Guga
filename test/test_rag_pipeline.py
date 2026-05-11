from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.rag.chunker import chunk_text
from guga.rag.embedder import HashingEmbedder
from guga.rag.faiss_store import VectorStore
from guga.rag.pipeline import RagPipeline
from guga.rag.schemas import DocumentChunk


class FakeFaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.rows = 0
        self.search_calls = 0

    def add(self, matrix) -> None:
        self.rows = len(matrix)

    def search(self, query, top_k: int):
        self.search_calls += 1
        count = min(top_k, self.rows)
        return [[1.0 - (idx * 0.1) for idx in range(count)]], [[idx for idx in range(count)]]


class FakeFaiss:
    def __init__(self) -> None:
        self.indexes: list[FakeFaissIndex] = []

    def IndexFlatIP(self, dim: int) -> FakeFaissIndex:
        index = FakeFaissIndex(dim)
        self.indexes.append(index)
        return index


class RagPipelineTest(unittest.TestCase):
    def test_chunk_text_with_overlap(self) -> None:
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)
        self.assertEqual(chunks, ["abcd", "cdef", "efgh", "ghij", "ij"])

    def test_rebuild_and_retrieve_dual_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            docs_dir = root / "documents"
            index_dir = root / "index"
            memory_root.mkdir(parents=True, exist_ok=True)
            docs_dir.mkdir(parents=True, exist_ok=True)

            (memory_root / "archival_memory.jsonl").write_text(
                '{"id":"mem1","summary":"用户提到：我在杭州做后端","source_session_id":"sess1","source_message_ids":["msg1"],"created_at":"2026-01-01T00:00:00+00:00","status":"active"}\n',
                encoding="utf-8",
            )
            (docs_dir / "guide.md").write_text("杭州后端开发实践指南", encoding="utf-8")

            pipeline = RagPipeline(
                index_dir=index_dir,
                documents_dir=docs_dir,
                embedding_model="fake",
                chunk_size=32,
                chunk_overlap=8,
                embedder=HashingEmbedder(dim=64),
            )
            result = pipeline.rebuild_indexes(memory_root=memory_root)

            self.assertGreaterEqual(result["memory_chunks"], 1)
            self.assertGreaterEqual(result["document_chunks"], 1)

            memory_hits, doc_hits = pipeline.retrieve("你记得我在杭州做什么", memory_top_k=3, document_top_k=3)
            self.assertTrue(memory_hits)
            self.assertTrue(doc_hits)
            self.assertEqual(memory_hits[0].source_type, "memory")
            self.assertEqual(doc_hits[0].source_type, "document")

    def test_vector_store_uses_typed_faiss_index_for_source_type_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(Path(tmp))
            fake_faiss = FakeFaiss()
            store._faiss = fake_faiss
            chunks = [
                DocumentChunk(id="memory:m1:c0", text="memory one", source_type="memory", source_id="m1"),
                DocumentChunk(id="document:d1:c0", text="document one", source_type="document", source_id="d1"),
                DocumentChunk(id="memory:m2:c0", text="memory two", source_type="memory", source_id="m2"),
            ]
            vectors = [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]

            store.rebuild(chunks=chunks, vectors=vectors)
            rows = store.search(query_vec=[1.0, 0.0], top_k=2, source_type="memory")

            self.assertEqual([idx for idx, _ in rows], [0, 2])
            typed_search_calls = sum(index.search_calls for index in fake_faiss.indexes if index.rows == 2)
            self.assertEqual(typed_search_calls, 1)


if __name__ == "__main__":
    unittest.main()
