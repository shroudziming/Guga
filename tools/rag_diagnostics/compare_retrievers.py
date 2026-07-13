from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_scores import hashing_encode


_TOKEN = re.compile(r"[A-Za-z0-9$,.]+")


def tokenize(text: str) -> list[str]:
    return [token.lower().strip(".,") for token in _TOKEN.findall(text) if token.strip(".,")]


def bm25_scores(texts: list[str], query: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    documents = [tokenize(text) for text in texts]
    query_terms = tokenize(query)
    if not documents:
        return []
    average_length = sum(len(document) for document in documents) / len(documents) or 1.0
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * len(document) / average_length)
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def diversify_by_source(rows: list[dict[str, Any]], *, max_per_source: int, limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if counts[source_id] >= max_per_source:
            continue
        selected.append(row)
        counts[source_id] += 1
        if len(selected) >= limit:
            break
    return selected


def rank_of_text(rows: list[dict[str, Any]], pattern: str) -> int | None:
    needle = pattern.casefold()
    for rank, row in enumerate(rows, start=1):
        if needle in str(row.get("text", "")).casefold():
            return rank
    return None


def compare(index_dir: Path, query: str, *, patterns: list[str], top_k: int, max_per_source: int) -> dict[str, Any]:
    chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = json.loads((index_dir / "vectors.json").read_text(encoding="utf-8"))
    if len(chunks) != len(vectors):
        raise ValueError("chunk/vector count mismatch")
    if vectors and len(vectors[0]) != 128:
        raise ValueError("persisted hashing comparison requires a 128-dimensional index")
    query_vector = hashing_encode(query)
    hashing = [
        {**chunk, "score": sum(left * right for left, right in zip(query_vector, vector))}
        for chunk, vector in zip(chunks, vectors)
    ]
    hashing.sort(key=lambda row: float(row["score"]), reverse=True)
    lexical_scores = bm25_scores([str(chunk.get("text", "")) for chunk in chunks], query)
    bm25 = [{**chunk, "score": score} for chunk, score in zip(chunks, lexical_scores)]
    bm25.sort(key=lambda row: float(row["score"]), reverse=True)
    variants = {
        "persisted_hashing": hashing,
        "hashing_source_capped": diversify_by_source(hashing, max_per_source=max_per_source, limit=len(hashing)),
        "bm25": bm25,
        "bm25_source_capped": diversify_by_source(bm25, max_per_source=max_per_source, limit=len(bm25)),
    }
    return {
        "query": query,
        "index_chunk_count": len(chunks),
        "patterns": {
            name: {
                pattern: rank_of_text(rows, pattern)
                for pattern in patterns
            }
            for name, rows in variants.items()
        },
        "top": {
            name: [
                {
                    "rank": rank,
                    "score": round(float(row["score"]), 6),
                    "chunk_id": row.get("id", ""),
                    "source_id": row.get("source_id", ""),
                    "created_at": row.get("created_at", ""),
                    "text": str(row.get("text", "")).replace("\n", " ")[:220],
                }
                for rank, row in enumerate(rows[:top_k], start=1)
            ]
            for name, rows in variants.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare persisted hashing retrieval with isolated lexical/source-diversity controls.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--expect", action="append", default=[], help="Evidence substring whose rank should be reported.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = compare(
        args.index_dir,
        args.query,
        patterns=args.expect,
        top_k=max(1, args.top_k),
        max_per_source=max(1, args.max_per_source),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
