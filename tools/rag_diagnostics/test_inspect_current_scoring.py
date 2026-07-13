from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inspect_current_scoring import inspect_scores


class CurrentScoringInspectionTest(unittest.TestCase):
    def test_reports_actual_truncated_query_tokens_and_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "id": "turn_long",
                "type": "conversation_turn",
                "summary": "what was the amount I was processing data under regulation",
                "raw_excerpt": "",
                "source_session_id": "s1",
                "source_message_ids": ["m1"],
                "created_at": "2023-01-01T00:00:00+00:00",
                "importance": 0.5,
                "confidence": 0.9,
                "retention": 1.0,
                "status": "active",
            }
            (root / "session_memories.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = inspect_scores(
                root,
                "What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?",
                ["processing data"],
            )

        self.assertEqual(len(report["query_tokens"]), 12)
        self.assertNotIn("Wells", report["query_tokens"])
        self.assertEqual(report["records"][0]["id"], "turn_long")
        self.assertIn("lexical_overlap", report["records"][0]["components"])


if __name__ == "__main__":
    unittest.main()
