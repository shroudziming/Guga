from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import unicodedata


def normalize_answer(value: Any) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    pieces: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            continue
        if char.isspace():
            continue
        pieces.append(char)
    return "".join(pieces)


def score_results_file(
    results_file: Path,
    metrics_file: Path | None = None,
    failures_file: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(results_file)
    scored_rows = [_score_row(row) for row in rows]
    metrics = _build_metrics(scored_rows)

    if metrics_file is not None:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures_file is not None:
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        failures = [row for row in scored_rows if not row["correct"]]
        failures_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + ("\n" if failures else ""),
            encoding="utf-8",
        )

    return metrics


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("prediction", "")
    normalized_prediction = normalize_answer(prediction)
    normalized_answers = _normalized_answers(row.get("answer", ""))
    correct = normalized_prediction in normalized_answers if normalized_answers else False
    scored = dict(row)
    scored["correct"] = correct
    scored["normalized_answer"] = normalized_answers[0] if normalized_answers else ""
    scored["normalized_answers"] = normalized_answers
    scored["normalized_prediction"] = normalized_prediction
    return scored


def _normalized_answers(answer: Any) -> list[str]:
    values = answer if isinstance(answer, list) else [answer]
    normalized: list[str] = []
    for value in values:
        item = normalize_answer(value)
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _build_metrics(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_rows)
    correct = sum(1 for row in scored_rows if row["correct"])
    by_type_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        question_type = str(row.get("question_type") or "unknown")
        by_type_rows[question_type].append(row)

    by_question_type = {}
    for question_type in sorted(by_type_rows):
        rows = by_type_rows[question_type]
        type_total = len(rows)
        type_correct = sum(1 for row in rows if row["correct"])
        by_question_type[question_type] = {
            "total": type_total,
            "correct": type_correct,
            "accuracy": _accuracy(type_correct, type_total),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": _accuracy(correct, total),
        "by_question_type": by_question_type,
    }


def _accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
