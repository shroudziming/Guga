from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    if k <= 0 or not expected_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for item in top if item in expected_ids)
    return hits / len(expected_ids)
