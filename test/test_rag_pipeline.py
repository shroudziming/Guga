from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.rag.chunker import chunk_text
from guga.rag.embedder import HashingEmbedder
from guga.rag.pipeline import RagPipeline


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


if __name__ == "__main__":
    unittest.main()
