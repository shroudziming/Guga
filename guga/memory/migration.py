from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from guga.memory.time_utils import apply_temporal_fields
from guga.utils.paths import memory_data_dir


MEMORY_JSONL_FILES = ("archival_memory.jsonl", "event_summaries.jsonl", "session_memories.jsonl")


def repair_memory_root(memory_root: Path | None = None, *, dry_run: bool = False, backup: bool = False) -> dict[str, int]:
    root = memory_root or memory_data_dir()
    stats = {"files": 0, "checked": 0, "updated": 0, "temporal_backfilled": 0, "noise_marked": 0}
    for name in MEMORY_JSONL_FILES:
        path = root / name
        if not path.exists():
            continue
        file_stats = repair_memory_file(path, dry_run=dry_run, backup=backup)
        stats["files"] += 1
        for key in ("checked", "updated", "temporal_backfilled", "noise_marked"):
            stats[key] += int(file_stats.get(key, 0))
    return stats


def repair_memory_file(path: Path, *, dry_run: bool = False, backup: bool = False) -> dict[str, int]:
    stats = {"checked": 0, "updated": 0, "temporal_backfilled": 0, "noise_marked": 0}
    if not path.exists():
        return stats

    output_lines: list[str] = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            output_lines.append(line)
            continue
        if not isinstance(payload, dict):
            output_lines.append(line)
            continue

        stats["checked"] += 1
        repaired, row_changed, row_stats = repair_memory_record(payload)
        changed = changed or row_changed
        if row_changed:
            stats["updated"] += 1
            stats["temporal_backfilled"] += int(row_stats.get("temporal_backfilled", 0))
            stats["noise_marked"] += int(row_stats.get("noise_marked", 0))
        output_lines.append(json.dumps(repaired, ensure_ascii=False))

    if changed and not dry_run:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return stats


def repair_memory_record(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, int]]:
    row = dict(payload)
    stats = {"temporal_backfilled": 0, "noise_marked": 0}
    changed = False

    if _needs_temporal_backfill(row):
        row = apply_temporal_fields(
            row,
            text=_record_text(row),
            reference_time=str(row.get("updated_at") or row.get("created_at") or ""),
        )
        stats["temporal_backfilled"] = 1
        changed = True

    noise_reason = _noise_reason(row)
    if noise_reason == "mojibake":
        if row.get("status") != "decayed":
            row["status"] = "decayed"
            changed = True
        if row.get("exclude_from_retrieval") is not True:
            row["exclude_from_retrieval"] = True
            changed = True
        if row.get("noise_reason") != noise_reason:
            row["noise_reason"] = noise_reason
            changed = True
    elif noise_reason == "system_feedback":
        if row.get("type") != "system_feedback":
            row.setdefault("original_type", str(row.get("type", "")))
            row["type"] = "system_feedback"
            changed = True
        if row.get("exclude_from_retrieval") is not True:
            row["exclude_from_retrieval"] = True
            changed = True
        if row.get("noise_reason") != noise_reason:
            row["noise_reason"] = noise_reason
            changed = True

    if noise_reason:
        stats["noise_marked"] = 1
    return row, changed, stats


def _needs_temporal_backfill(row: dict[str, Any]) -> bool:
    return not all(str(row.get(key, "")).strip() for key in ("valid_at", "semantic_day", "time_source", "time_granularity"))


def _record_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("summary", "")), str(row.get("raw_excerpt", ""))]
    return "\n".join(part for part in parts if part.strip())


def _noise_reason(row: dict[str, Any]) -> str:
    text = _record_text(row)
    if _looks_like_mojibake(text):
        return "mojibake"
    if str(row.get("type", "")) != "event_summary" and _looks_like_system_feedback(text):
        return "system_feedback"
    return ""


def _looks_like_mojibake(text: str) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    if not compact:
        return False
    replacement_count = compact.count("\ufffd")
    question_count = compact.count("?")
    return replacement_count > 0 or ("??" in compact and (question_count / len(compact)) >= 0.25)


def _looks_like_system_feedback(text: str) -> bool:
    lower = text.lower()
    feedback_tokens = ("bug", "没有输出", "没输出", "看不到你说", "输出不完整", "没生成", "卡住", "终止")
    assistant_tokens = ("助手", "你刚才", "上次我们聊天时你", "模型", "大模型", "llm")
    return any(token in lower for token in feedback_tokens) and any(token in lower for token in assistant_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Guga memory temporal fields and mark retrieval noise.")
    parser.add_argument("--memory-root", type=Path, default=memory_data_dir())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    stats = repair_memory_root(args.memory_root, dry_run=args.dry_run, backup=args.backup)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
