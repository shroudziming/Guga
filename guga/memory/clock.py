from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today_bucket() -> str:
    return datetime.now().astimezone().strftime("%Y-%m")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
