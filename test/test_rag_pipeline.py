from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guga.config import DEFAULT_RAG_EMBEDDING_MODEL
from guga.rag.chunker import chunk_text
from guga.rag.embedder import HashingEmbedder, SentenceTransformerEmbedder, build_embedder
from guga.rag.faiss_store import IncompatibleIndexError, VectorStore
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
    def test_production_embedding_model_is_bge_m3(self) -> None:
        self.assertEqual(DEFAULT_RAG_EMBEDDING_MODEL, "BAAI/bge-m3")

    def test_sentence_transformer_failure_does_not_fall_back_to_hashing(self) -> None:
        embedder = build_embedder("BAAI/bge-m3")
        self.assertIsInstance(embedder, SentenceTransformerEmbedder)

        with patch.object(embedder, "_load_model", side_effect=OSError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "BAAI/bge-m3"):
                embedder.encode(["query"])

    def test_vector_store_requires_faiss(self) -> None:
        original_import = builtins.__import__

        def rejecting_import(name, *args, **kwargs):
            if name == "faiss":
                raise ImportError("faiss unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=rejecting_import):
            with self.assertRaisesRegex(RuntimeError, "FAISS"):
                VectorStore(Path("unused"), embedding_model="BAAI/bge-m3")

    def test_persisted_index_rejects_different_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            (index_dir / "chunks.jsonl").write_text(
                json.dumps(DocumentChunk(id="memory:m1:c0", text="x", source_type="memory", source_id="m1").to_dict())
                + "\n",
                encoding="utf-8",
            )
            (index_dir / "vectors.json").write_text("[[1.0, 0.0]]", encoding="utf-8")
            (index_dir / "index_meta.json").write_text(
                json.dumps({"schema_version": 1, "embedding_model": "old/model", "dimension": 2}),
                encoding="utf-8",
            )

            store = VectorStore(index_dir, embedding_model="BAAI/bge-m3")
            with self.assertRaisesRegex(IncompatibleIndexError, "old/model"):
                store.load()

    def test_chunk_text_with_overlap(self) -> None:
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)
        self.assertEqual(chunks, ["abcd", "cdef", "efgh", "ghij", "ij"])

    def test_rebuild_excludes_records_derived_from_inactive_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            memory_root.mkdir()
            (memory_root / "semantic_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "evt_active", "type": "semantic_event", "description": "当前额度为四十万美元", "status": "active"}),
                        json.dumps({"id": "evt_old", "type": "semantic_event", "description": "旧额度为三十五万美元", "status": "inactive", "inactive_reason": "replaced"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (memory_root / "event_summaries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "summary_current", "type": "event_summary", "summary": "当前额度", "covered_event_ids": ["evt_active"], "deactivated_event_ids": [], "status": "active"}),
                        json.dumps({"id": "summary_old", "type": "event_summary", "summary": "旧额度", "covered_event_ids": ["evt_old"], "deactivated_event_ids": ["evt_old"], "status": "active"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (memory_root / "archival_memory.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "mem_current", "summary": "当前额度", "source_event_ids": ["evt_active"], "status": "active"}),
                        json.dumps({"id": "mem_old", "summary": "旧额度", "source_event_ids": ["evt_old"], "status": "active"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pipeline = RagPipeline(
                index_dir=root / "index",
                documents_dir=root / "documents",
                embedding_model="test-hashing",
                chunk_size=64,
                chunk_overlap=8,
                embedder=HashingEmbedder(dim=64),
            )

            pipeline.rebuild_indexes(memory_root)

            source_ids = {chunk.source_id for chunk in pipeline.store.chunks}
            self.assertTrue({"evt_active", "summary_current", "mem_current"}.issubset(source_ids))
            self.assertTrue({"evt_old", "summary_old", "mem_old"}.isdisjoint(source_ids))

    def test_adding_inactive_payload_removes_existing_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = RagPipeline(
                index_dir=root / "index",
                documents_dir=root / "documents",
                embedding_model="test-hashing",
                chunk_size=64,
                chunk_overlap=8,
                embedder=HashingEmbedder(dim=64),
            )
            active = {
                "id": "evt_cancelled",
                "type": "semantic_event",
                "description": "用户周日看牙",
                "status": "active",
            }
            pipeline.add_memory_record(active)
            self.assertIn("evt_cancelled", {chunk.source_id for chunk in pipeline.store.chunks})

            pipeline.add_memory_record({**active, "status": "inactive", "inactive_reason": "cancelled"})

            self.assertNotIn("evt_cancelled", {chunk.source_id for chunk in pipeline.store.chunks})

    def test_prune_removes_existing_derived_vectors_after_event_deactivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            memory_root.mkdir()
            event_path = memory_root / "semantic_events.jsonl"
            event_path.write_text(
                json.dumps({"id": "evt_old", "type": "semantic_event", "description": "旧安排", "status": "active"}) + "\n",
                encoding="utf-8",
            )
            (memory_root / "archival_memory.jsonl").write_text(
                json.dumps({"id": "mem_old", "summary": "旧安排", "source_event_ids": ["evt_old"], "status": "active"}) + "\n",
                encoding="utf-8",
            )
            pipeline = RagPipeline(
                index_dir=root / "index",
                documents_dir=root / "documents",
                embedding_model="test-hashing",
                chunk_size=64,
                chunk_overlap=8,
                embedder=HashingEmbedder(dim=64),
            )
            pipeline.rebuild_indexes(memory_root)
            self.assertTrue({"evt_old", "mem_old"}.issubset({chunk.source_id for chunk in pipeline.store.chunks}))
            event_path.write_text(
                json.dumps({"id": "evt_old", "type": "semantic_event", "description": "旧安排", "status": "inactive", "inactive_reason": "cancelled"}) + "\n",
                encoding="utf-8",
            )

            pipeline.prune_invalid_memory_records(memory_root)

            self.assertTrue({"evt_old", "mem_old"}.isdisjoint({chunk.source_id for chunk in pipeline.store.chunks}))

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
