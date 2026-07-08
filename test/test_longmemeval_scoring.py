from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.benchmark.scoring import normalize_answer, score_results_file


class LongMemEvalScoringTest(unittest.TestCase):
    def test_normalize_answer_ignores_case_whitespace_and_punctuation(self) -> None:
        self.assertEqual(normalize_answer(" Blue notebooks! "), normalize_answer("blue notebooks"))
        self.assertEqual(normalize_answer("2026-01-02"), normalize_answer("2026/01/02"))

    def test_score_results_file_writes_metrics_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_file = root / "results.jsonl"
            metrics_file = root / "metrics.json"
            failures_file = root / "failures.jsonl"
            rows = [
                {
                    "case_id": "q1",
                    "question_type": "information_extraction",
                    "question": "What color?",
                    "answer": "Blue notebooks!",
                    "prediction": "blue notebooks",
                },
                {
                    "case_id": "q2",
                    "question_type": "temporal_reasoning",
                    "question": "When?",
                    "answer": "2026-01-02",
                    "prediction": "2026/01/02",
                },
                {
                    "case_id": "q3",
                    "question_type": "temporal_reasoning",
                    "question": "Where?",
                    "answer": "Paris",
                    "prediction": "London",
                },
            ]
            results_file.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            metrics = score_results_file(
                results_file=results_file,
                metrics_file=metrics_file,
                failures_file=failures_file,
            )

            self.assertEqual(metrics["total"], 3)
            self.assertEqual(metrics["correct"], 2)
            self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
            self.assertEqual(metrics["by_question_type"]["information_extraction"]["accuracy"], 1.0)
            self.assertEqual(metrics["by_question_type"]["temporal_reasoning"]["total"], 2)
            self.assertEqual(metrics["by_question_type"]["temporal_reasoning"]["correct"], 1)

            saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_metrics, metrics)
            failures = [json.loads(line) for line in failures_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["case_id"], "q3")
            self.assertEqual(failures[0]["normalized_answer"], "paris")
            self.assertEqual(failures[0]["normalized_prediction"], "london")

    def test_score_results_accepts_answer_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_file = root / "results.jsonl"
            results_file.write_text(
                json.dumps(
                    {
                        "case_id": "q1",
                        "question_type": "multi_session_reasoning",
                        "answer": ["blue", "green"],
                        "prediction": " Green! ",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = score_results_file(results_file=results_file)

            self.assertEqual(metrics["total"], 1)
            self.assertEqual(metrics["correct"], 1)


if __name__ == "__main__":
    unittest.main()
