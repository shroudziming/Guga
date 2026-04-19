from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.rag.embedder import HashingEmbedder
from guga.rag.pipeline import RagPipeline


class RagSmokeTest(unittest.TestCase):
    def test_memory_manager_rebuild_and_context_contains_document_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            docs_dir = root / "documents"
            index_dir = memory_root / "rag" / "index"
            docs_dir.mkdir(parents=True, exist_ok=True)
            memory_root.mkdir(parents=True, exist_ok=True)

            (docs_dir / "onboarding.md").write_text("Guga 项目是本地部署聊天陪伴系统。", encoding="utf-8")
            (memory_root / "archival_memory.jsonl").write_text(
                '{"id":"mem1","summary":"用户提到：我最近换了工作","source_session_id":"sess1","source_message_ids":["msg1"],"created_at":"2026-01-01T00:00:00+00:00","status":"active"}\n',
                encoding="utf-8",
            )

            pipeline = RagPipeline(
                index_dir=index_dir,
                documents_dir=docs_dir,
                embedding_model="fake",
                chunk_size=64,
                chunk_overlap=10,
                embedder=HashingEmbedder(dim=64),
            )
            pipeline.rebuild_indexes(memory_root=memory_root)

            memory_hits, doc_hits = pipeline.retrieve("这个项目是什么", memory_top_k=4, document_top_k=4)

            self.assertTrue(memory_hits)
            self.assertTrue(doc_hits)
            self.assertEqual(memory_hits[0].source_type, "memory")
            self.assertEqual(doc_hits[0].source_type, "document")


if __name__ == "__main__":
    unittest.main()
