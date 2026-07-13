from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real BGE-M3 ranking probe against persisted chunks.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    chunks = [
        json.loads(line)
        for line in (args.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [str(chunk["text"]) for chunk in chunks]
    print(json.dumps({"phase": "start", "chunks": len(chunks), "torch": torch.__version__}), flush=True)

    started = perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", device="cuda", local_files_only=True)
    print(json.dumps({"phase": "loaded", "seconds": round(perf_counter() - started, 3)}), flush=True)

    probe = model.encode(texts[:3], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    print(json.dumps({"phase": "probe", "shape": list(probe.shape)}), flush=True)

    started = perf_counter()
    vectors = model.encode(
        texts,
        batch_size=max(1, args.batch_size),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    encode_seconds = perf_counter() - started
    print(json.dumps({"phase": "encoded", "seconds": round(encode_seconds, 3), "shape": list(vectors.shape)}), flush=True)

    model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    latencies: list[float] = []
    for _ in range(10):
        started = perf_counter()
        query_vector = model.encode(
            [args.query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        torch.cuda.synchronize()
        latencies.append((perf_counter() - started) * 1000)

    scores = vectors @ query_vector
    order = np.argsort(-scores)
    markers = ("$350,000", "$400,000", "The regulation does not purport")
    marker_ranks: dict[str, dict] = {}
    for marker in markers:
        for rank, index in enumerate(order, start=1):
            row_index = int(index)
            if marker.casefold() in texts[row_index].casefold():
                marker_ranks[marker] = {
                    "rank": rank,
                    "score": round(float(scores[row_index]), 6),
                    "chunk_id": chunks[row_index].get("id", ""),
                    "created_at": chunks[row_index].get("created_at", ""),
                }
                break

    result = {
        "phase": "result",
        "model": "BAAI/bge-m3",
        "device": torch.cuda.get_device_name(0),
        "model_dimension": int(vectors.shape[1]),
        "chunk_encode_seconds": round(encode_seconds, 3),
        "query_latency_ms": {
            "min": round(min(latencies), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "markers": marker_ranks,
        "top10": [
            {
                "rank": rank,
                "score": round(float(scores[int(index)]), 6),
                "chunk_id": chunks[int(index)].get("id", ""),
                "source_id": chunks[int(index)].get("source_id", ""),
                "text": texts[int(index)].replace("\n", " ")[:220],
            }
            for rank, index in enumerate(order[:10], start=1)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
