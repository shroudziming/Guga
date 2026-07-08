from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.benchmark.scoring import score_results_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LongMemEval results produced by Guga.")
    parser.add_argument("--results", required=True, type=Path, help="Path to results.jsonl.")
    parser.add_argument("--metrics", type=Path, default=None, help="Output metrics.json path.")
    parser.add_argument("--failures", type=Path, default=None, help="Output failures.jsonl path.")
    args = parser.parse_args()

    results_file = args.results
    metrics_file = args.metrics or results_file.with_name("metrics.json")
    failures_file = args.failures or results_file.with_name("failures.jsonl")
    metrics = score_results_file(
        results_file=results_file,
        metrics_file=metrics_file,
        failures_file=failures_file,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"metrics={metrics_file}")
    print(f"failures={failures_file}")


if __name__ == "__main__":
    main()
