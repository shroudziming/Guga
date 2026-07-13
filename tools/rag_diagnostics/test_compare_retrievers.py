from __future__ import annotations

import unittest

from compare_retrievers import bm25_scores, diversify_by_source, rank_of_text


class RetrieverComparisonTest(unittest.TestCase):
    def test_bm25_rewards_query_terms_instead_of_shared_character_shapes(self) -> None:
        texts = [
            "GDPR processing personal data regulation",
            "pre-approved for $400,000 from Wells Fargo",
            "Oxford masters programme requirements",
        ]

        scores = bm25_scores(texts, "mortgage pre-approved Wells Fargo")

        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_diversify_limits_chunks_from_one_long_message(self) -> None:
        rows = [
            {"source_id": "gdpr", "score": 0.9},
            {"source_id": "gdpr", "score": 0.8},
            {"source_id": "correct", "score": 0.7},
        ]

        selected = diversify_by_source(rows, max_per_source=1, limit=3)

        self.assertEqual([row["source_id"] for row in selected], ["gdpr", "correct"])

    def test_rank_of_text_finds_expected_evidence(self) -> None:
        rows = [{"text": "old $350,000"}, {"text": "new $400,000"}]

        self.assertEqual(rank_of_text(rows, "$400,000"), 2)
        self.assertIsNone(rank_of_text(rows, "$500,000"))


if __name__ == "__main__":
    unittest.main()
