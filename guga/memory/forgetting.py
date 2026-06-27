from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from guga.memory.time_utils import format_beijing, now_beijing_iso


def now_iso() -> str:
    return now_beijing_iso()


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def elapsed_days(since: str, now: datetime | None = None) -> float:
    start = parse_iso(since)
    if start is None:
        return 0.0
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - start.astimezone(timezone.utc)).total_seconds() / 86400)


def retention_score(record: dict, now: datetime | None = None) -> float:
    strength = max(1, int(record.get("memory_strength", 1) or 1))
    anchor = str(record.get("last_recalled_at") or record.get("created_at") or "")
    days = elapsed_days(anchor, now=now)
    return round(math.exp(-days / strength), 6)


def normalize_memorybank_fields(record: dict, now: datetime | None = None) -> dict:
    normalized = dict(record)
    normalized["memory_strength"] = max(1, int(normalized.get("memory_strength", 1) or 1))
    normalized["last_recalled_at"] = str(
        normalized.get("last_recalled_at") or normalized.get("created_at") or now_iso()
    )
    normalized["retention"] = float(normalized.get("retention") or retention_score(normalized, now=now))
    normalized["status"] = str(normalized.get("status", "active"))
    return normalized


def reinforce_jsonl_records(path: Path, memory_ids: set[str], now: datetime | None = None) -> list[str]:
    if not path.exists() or not memory_ids:
        return []

    current = now or datetime.now(timezone.utc)
    recalled_at = format_beijing(current)
    rows: list[dict] = []
    changed_ids: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        if str(payload.get("id", "")) in memory_ids:
            payload = normalize_memorybank_fields(payload, now=current)
            payload["memory_strength"] = int(payload["memory_strength"]) + 1
            payload["last_recalled_at"] = recalled_at
            payload["retention"] = 1.0
            payload["status"] = "active"
            changed_ids.append(str(payload.get("id", "")))
        rows.append(payload)

    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return changed_ids


def refresh_jsonl_retention(
    path: Path,
    decay_threshold: float,
    now: datetime | None = None,
    min_age_days: float = 0.0,
) -> dict[str, int]:
    if not path.exists():
        return {"checked": 0, "decayed": 0}

    current = now or datetime.now(timezone.utc)
    rows: list[dict] = []
    checked = 0
    decayed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        payload = normalize_memorybank_fields(payload, now=current)
        if str(payload.get("status", "active")) == "active":
            checked += 1
            payload["retention"] = retention_score(payload, now=current)
            anchor = str(payload.get("last_recalled_at") or payload.get("created_at") or "")
            if elapsed_days(anchor, now=current) < min_age_days:
                rows.append(payload)
                continue
            if float(payload["retention"]) < decay_threshold:
                payload["status"] = "decayed"
                payload["decayed_at"] = format_beijing(current)
                decayed += 1
        rows.append(payload)

    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return {"checked": checked, "decayed": decayed}
