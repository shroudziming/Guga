from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.benchmark.longmemeval import run_longmemeval_benchmark
from guga.benchmark.workspace import benchmark_workspace
from guga.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, default_generation_config
from guga.models import create_chat_model


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Guga on LongMemEval with isolated benchmark state.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to LongMemEval JSON/JSONL data.")
    parser.add_argument("--run-id", default=None, help="Run id under data/benchmarks/longmemeval/runs/.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--debug", action="store_true", help="Write benchmark debug reports.")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic RAG indexes for a lightweight run.")
    parser.add_argument(
        "--ingest-mode",
        choices=("raw", "replay"),
        default="raw",
        help="raw imports history as retrievable memory; replay finalizes each historical turn like daily chat.",
    )
    args = parser.parse_args()

    _load_env_file()
    model_id = os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID)
    cache_dir = os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    model = create_chat_model(model_id=model_id, cache_dir=cache_dir)
    workspace = benchmark_workspace("longmemeval", run_id=args.run_id)
    results = run_longmemeval_benchmark(
        dataset_path=args.dataset,
        model=model,
        workspace=workspace,
        generation=default_generation_config(),
        limit=args.limit,
        debug=args.debug,
        enable_semantic=not args.no_semantic,
        ingest_mode=args.ingest_mode,
    )

    print(f"LongMemEval cases={len(results)}")
    print(f"run_root={workspace.root}")
    print(f"results={workspace.results_file}")


if __name__ == "__main__":
    main()
