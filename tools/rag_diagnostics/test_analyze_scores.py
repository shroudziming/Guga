from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_scores import analyze_index, hashing_encode


class RagScoreDiagnosticsTest(unittest.TestCase):
    def test_hashing_encode_returns_normalized_vector(self) -> None:
        vector = hashing_encode("GDPR data protection", dim=128)

        self.assertEqual(len(vector), 128)
        self.assertAlmostEqual(sum(value * value for value in vector), 1.0, places=6)

    def test_analyze_index_reports_distribution_and_source_concentration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            chunks = [
                {"id": "c1", "text": "GDPR personal data", "source_id": "article"},
                {"id": "c2", "text": "GDPR privacy rights", "source_id": "article"},
                {"id": "c3", "text": "dentist appointment", "source_id": "chat"},
            ]
            vectors = [hashing_encode(row["text"], dim=128) for row in chunks]
            (index_dir / "chunks.jsonl").write_text(
                "\n".join(json.dumps(row) for row in chunks) + "\n",
                encoding="utf-8",
            )
            (index_dir / "vectors.json").write_text(json.dumps(vectors), encoding="utf-8")

            report = analyze_index(
                index_dir=index_dir,
                queries=["GDPR privacy"],
                top_k=3,
                focus_source_id="article",
            )

        query = report["queries"][0]
        self.assertEqual(report["index"]["vector_dim"], 128)
        self.assertEqual(query["top_chunks"][0]["source_id"], "article")
        self.assertEqual(query["top_source_counts"]["article"], 2)
        self.assertEqual(query["focus_source"]["chunk_count"], 2)
        self.assertIn("stddev", query["all_scores"])
        self.assertIn("top1_minus_topk", query["separation"])

    def test_analyze_index_rejects_non_hashing_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            (index_dir / "chunks.jsonl").write_text(
                json.dumps({"id": "c1", "text": "text", "source_id": "s1"}) + "\n",
                encoding="utf-8",
            )
            (index_dir / "vectors.json").write_text(json.dumps([[0.0] * 384]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "128-dimensional HashingEmbedder"):
                analyze_index(index_dir=index_dir, queries=["query"], top_k=1)


if __name__ == "__main__":
    unittest.main()
