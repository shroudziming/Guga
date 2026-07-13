from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.memory.manager import MemoryManager


def inspect_scores(memory_root: Path, query: str, markers: list[str]) -> dict[str, Any]:
    manager = MemoryManager(memory_root=memory_root, enable_semantic=False)
    records: list[dict[str, Any]] = []
    for path in (
        memory_root / "session_memories.jsonl",
        memory_root / "archival_memory.jsonl",
        memory_root / "event_summaries.jsonl",
        memory_root / "semantic_events.jsonl",
    ):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
    selected: list[dict[str, Any]] = []
    for marker in markers:
        match = next(
            (
                record
                for record in records
                if marker.casefold()
                in f"{record.get('summary', '')} {record.get('raw_excerpt', '')} {record.get('description', '')}".casefold()
            ),
            None,
        )
        if match is None:
            selected.append({"marker": marker, "error": "not found"})
            continue
        score, components = manager._score_components(match, query)
        selected.append(
            {
                "marker": marker,
                "id": match.get("id", ""),
                "type": match.get("type", ""),
                "created_at": match.get("created_at", ""),
                "score": round(score, 6),
                "components": components,
                "excerpt": str(match.get("summary") or match.get("description") or "")[:240],
            }
        )
    return {"query": query, "query_tokens": manager._tokens(query), "records": selected}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the current MemoryManager lexical score without changing memory files.")
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--marker", action="append", required=True, help="Substring identifying a record to score.")
    args = parser.parse_args()
    print(json.dumps(inspect_scores(args.memory_root, args.query, args.marker), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
