from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


HASHING_DIM = 128


def hashing_encode(text: str, *, dim: int = HASHING_DIM) -> list[float]:
    values = [0.0] * dim
    compact = text.strip()
    if not compact:
        return values
    grams: list[str] = []
    for size in (2, 3):
        for index in range(max(0, len(compact) - size + 1)):
            grams.append(compact[index : index + size])
    for token in grams[:256]:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        values[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def analyze_index(
    *,
    index_dir: Path,
    queries: list[str],
    top_k: int = 10,
    focus_source_id: str = "",
) -> dict[str, Any]:
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    vectors = _read_vectors(index_dir / "vectors.json")
    if len(chunks) != len(vectors):
        raise ValueError(f"chunk/vector count mismatch: {len(chunks)} != {len(vectors)}")
    vector_dim = len(vectors[0]) if vectors else 0
    if vector_dim != HASHING_DIM:
        raise ValueError(
            f"this standalone tool expects a {HASHING_DIM}-dimensional HashingEmbedder index; got {vector_dim}"
        )
    top_k = max(1, min(int(top_k), len(chunks))) if chunks else 0
    query_reports = [
        _analyze_query(
            query=query,
            chunks=chunks,
            vectors=vectors,
            top_k=top_k,
            focus_source_id=focus_source_id,
        )
        for query in queries
    ]
    return {
        "index": {
            "path": str(index_dir.resolve()),
            "chunk_count": len(chunks),
            "vector_count": len(vectors),
            "vector_dim": vector_dim,
            "embedder": "HashingEmbedder",
            "similarity": "normalized inner product (cosine)",
        },
        "queries": query_reports,
    }


def _analyze_query(
    *,
    query: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    top_k: int,
    focus_source_id: str,
) -> dict[str, Any]:
    query_vector = hashing_encode(query)
    scored = [
        (sum(left * right for left, right in zip(query_vector, vector)), chunk)
        for chunk, vector in zip(chunks, vectors)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    scores = [score for score, _ in scored]
    top = scored[:top_k]
    source_counts = Counter(str(chunk.get("source_id", "")) for _, chunk in top)
    focus_scores = [score for score, chunk in scored if str(chunk.get("source_id", "")) == focus_source_id]
    separation = {
        "top1_minus_top2": _gap(scores, 0, 1),
        "top1_minus_topk": _gap(scores, 0, top_k - 1),
        "topk_range": round(max(scores[:top_k]) - min(scores[:top_k]), 6) if top else 0.0,
        "scores_are_tightly_clustered": bool(top and max(scores[:top_k]) - min(scores[:top_k]) < 0.05),
    }
    return {
        "query": query,
        "all_scores": _distribution(scores),
        "separation": separation,
        "top_source_counts": dict(source_counts.most_common()),
        "top_chunks": [
            {
                "rank": rank,
                "score": round(score, 6),
                "chunk_id": str(chunk.get("id", "")),
                "source_id": str(chunk.get("source_id", "")),
                "source_session_id": str(chunk.get("source_session_id", "")),
                "text_excerpt": str(chunk.get("text", "")).replace("\n", " ")[:240],
            }
            for rank, (score, chunk) in enumerate(top, start=1)
        ],
        "focus_source": {
            "source_id": focus_source_id,
            "chunk_count": len(focus_scores),
            "scores": _distribution(focus_scores),
        }
        if focus_source_id
        else None,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "p10": round(_percentile(ordered, 0.10), 6),
        "median": round(_percentile(ordered, 0.50), 6),
        "p90": round(_percentile(ordered, 0.90), 6),
        "max": round(ordered[-1], 6),
        "mean": round(fmean(ordered), 6),
        "stddev": round(pstdev(ordered), 6),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _gap(scores: list[float], first: int, second: int) -> float:
    if first < 0 or second < 0 or first >= len(scores) or second >= len(scores):
        return 0.0
    return round(scores[first] - scores[second], 6)


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"chunks.jsonl line {line_number} is not an object")
        rows.append(payload)
    return rows


def _read_vectors(path: Path) -> list[list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("vectors.json must contain an array")
    return [[float(value) for value in row] for row in payload]


def _print_report(report: dict[str, Any]) -> None:
    index = report["index"]
    print(f"Index: {index['path']}")
    print(f"Embedder: {index['embedder']} dim={index['vector_dim']} chunks={index['chunk_count']}")
    for query in report["queries"]:
        stats = query["all_scores"]
        separation = query["separation"]
        print(f"\nQuery: {query['query']}")
        print(
            "Scores: "
            f"min={stats['min']:.4f} p10={stats['p10']:.4f} median={stats['median']:.4f} "
            f"p90={stats['p90']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} std={stats['stddev']:.4f}"
        )
        print(
            f"Separation: top1-top2={separation['top1_minus_top2']:.4f} "
            f"top1-topk={separation['top1_minus_topk']:.4f} tight={separation['scores_are_tightly_clustered']}"
        )
        for chunk in query["top_chunks"]:
            print(
                f"  {chunk['rank']:>2}. {chunk['score']:.4f} {chunk['chunk_id']} "
                f"source={chunk['source_id']} | {chunk['text_excerpt']}"
            )
        if query["focus_source"]:
            focus = query["focus_source"]
            print(f"Focus source {focus['source_id']}: {json.dumps(focus['scores'], ensure_ascii=False)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect score clustering in a persisted Guga RAG index without modifying it.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory containing chunks.jsonl and vectors.json.")
    parser.add_argument("--query", action="append", required=True, help="Query to analyze; repeat for related/unrelated controls.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--focus-source-id", default="", help="Optional source_id whose chunk-score distribution is reported separately.")
    parser.add_argument("--json-output", type=Path, help="Optional path for the complete JSON report.")
    args = parser.parse_args()
    report = analyze_index(
        index_dir=args.index_dir,
        queries=args.query,
        top_k=args.top_k,
        focus_source_id=args.focus_source_id,
    )
    _print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report: {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
