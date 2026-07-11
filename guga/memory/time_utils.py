from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone


BEIJING_TZ = timezone(timedelta(hours=8))
WEEKDAY_MAP = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


@dataclass(frozen=True)
class ResolvedEventTime:
    start_at: str | None
    end_at: str | None
    end_unknown: bool
    time_source: str
    time_granularity: str


def now_beijing() -> datetime:
    return datetime.now(BEIJING_TZ)


def now_beijing_iso() -> str:
    return format_beijing(now_beijing())


def format_beijing(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=BEIJING_TZ)
    return value.astimezone(BEIJING_TZ).isoformat(timespec="seconds")


def parse_datetime(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(BEIJING_TZ) if value.tzinfo else value.replace(tzinfo=BEIJING_TZ)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.astimezone(BEIJING_TZ) if parsed.tzinfo else parsed.replace(tzinfo=BEIJING_TZ)


def day_bucket(value: str | datetime | None) -> str:
    parsed = parse_datetime(value)
    return parsed.date().isoformat() if parsed else now_beijing().date().isoformat()


def resolve_event_time(
    time_expression: str,
    reference_created_at: str | datetime | None,
    end_unknown: bool,
) -> ResolvedEventTime:
    """Resolve an event's semantic time without trusting model-supplied dates."""
    reference = parse_datetime(reference_created_at) or now_beijing()
    expression = time_expression.strip()
    if not expression:
        return ResolvedEventTime(None, None, bool(end_unknown), "unknown", "unknown")

    explicit_dates = _extract_explicit_dates(expression, reference)
    if explicit_dates:
        start = explicit_dates[0]
        end = explicit_dates[1] if len(explicit_dates) > 1 else None
        return _resolved_date_range(start, end, end_unknown, "explicit_date")

    relative_date = _extract_relative_date(expression, reference)
    if relative_date is not None:
        return _resolved_date_range(relative_date, None, end_unknown, "relative_date")

    weekday = _extract_event_weekday(expression, reference)
    if weekday is not None:
        return _resolved_date_range(weekday, None, end_unknown, "relative_weekday")

    next_week = _extract_next_week_start(expression, reference)
    if next_week is not None:
        return _resolved_date_range(next_week, None, end_unknown, "relative_weekday")

    return ResolvedEventTime(None, None, bool(end_unknown), "unknown", "unknown")


def _resolved_date_range(
    start: datetime,
    end: datetime | None,
    end_unknown: bool,
    source: str,
) -> ResolvedEventTime:
    start_of_day = datetime.combine(start.date(), time.min, tzinfo=BEIJING_TZ)
    if end is not None:
        end_of_day = datetime.combine(end.date(), time.max.replace(microsecond=0), tzinfo=BEIJING_TZ)
        return ResolvedEventTime(
            format_beijing(start_of_day),
            format_beijing(end_of_day),
            False,
            source,
            "date",
        )
    if end_unknown:
        return ResolvedEventTime(format_beijing(start_of_day), None, True, source, "date")
    end_of_day = datetime.combine(start.date(), time.max.replace(microsecond=0), tzinfo=BEIJING_TZ)
    return ResolvedEventTime(format_beijing(start_of_day), format_beijing(end_of_day), False, source, "date")


def apply_temporal_fields(
    payload: dict,
    text: str = "",
    reference_time: str | datetime | None = None,
    update_created_at: bool = False,
) -> dict:
    """Attach Zep-style transaction time and semantic valid time fields.

    created_at/updated_at are transaction times. valid_at/invalid_at represent
    when the fact or summary is true in the conversation timeline.
    """
    row = dict(payload)
    write_time = parse_datetime(reference_time) or now_beijing()
    if update_created_at or not row.get("created_at"):
        row["created_at"] = format_beijing(write_time)
    else:
        row["created_at"] = format_beijing(parse_datetime(str(row.get("created_at"))) or write_time)
    row["updated_at"] = format_beijing(write_time)

    extracted = extract_semantic_time(text, reference_time=write_time)
    if extracted is None:
        valid_at = parse_datetime(str(row.get("valid_at", ""))) or write_time
        row["time_source"] = str(row.get("time_source") or "ingestion_time")
        row["time_granularity"] = str(row.get("time_granularity") or "second")
    else:
        valid_at, source, granularity = extracted
        row["time_source"] = source
        row["time_granularity"] = granularity
    row["valid_at"] = format_beijing(valid_at)
    row["invalid_at"] = str(row.get("invalid_at", ""))
    row["semantic_day"] = valid_at.date().isoformat()
    return row


def extract_semantic_time(text: str, reference_time: str | datetime | None = None) -> tuple[datetime, str, str] | None:
    ref = parse_datetime(reference_time) or now_beijing()
    content = text.strip()
    if not content:
        return None

    explicit = _extract_explicit_date(content, ref)
    if explicit is not None:
        return explicit, "semantic_explicit_date", "date"

    relative = _extract_relative_date(content, ref)
    if relative is not None:
        return relative, "semantic_relative_date", "date"

    weekday = _extract_relative_weekday(content, ref)
    if weekday is not None:
        return weekday, "semantic_relative_weekday", "date"

    return None


def _extract_explicit_date(text: str, reference_time: datetime) -> datetime | None:
    dates = _extract_explicit_dates(text, reference_time)
    return dates[0] if dates else None


def _extract_explicit_dates(text: str, reference_time: datetime) -> list[datetime]:
    patterns = (
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        r"(\d{4})年(\d{1,2})月(\d{1,2})(?:日|号)?",
    )
    dates: list[datetime] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            date = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if date is not None:
                dates.append(date)
        if dates:
            return dates

    for match in re.finditer(r"(?<!\d)(\d{1,2})月(\d{1,2})(?:日|号)?", text):
        date = _safe_date(reference_time.year, int(match.group(1)), int(match.group(2)))
        if date is not None:
            dates.append(date)
    return dates


def _extract_relative_date(text: str, reference_time: datetime) -> datetime | None:
    offsets = (
        ("前天", -2),
        ("昨天", -1),
        ("今天", 0),
        ("明天", 1),
        ("后天", 2),
    )
    for token, offset in offsets:
        if token in text:
            target = reference_time.date() + timedelta(days=offset)
            return datetime.combine(target, time.min, tzinfo=BEIJING_TZ)
    return None


def _extract_relative_weekday(text: str, reference_time: datetime) -> datetime | None:
    match = re.search(r"(下周|下星期|下礼拜)([一二三四五六日天])", text)
    if not match:
        return None
    target_weekday = WEEKDAY_MAP[match.group(2)]
    days_until_next_week = 7 - reference_time.weekday()
    monday_next_week = reference_time.date() + timedelta(days=days_until_next_week)
    target = monday_next_week + timedelta(days=target_weekday)
    return datetime.combine(target, time.min, tzinfo=BEIJING_TZ)


def _extract_event_weekday(text: str, reference_time: datetime) -> datetime | None:
    match = re.search(r"(?:(?P<prefix>[下这本])(?:周|星期|礼拜)|(?:周|星期|礼拜))(?P<weekday>[一二三四五六日天])", text)
    if not match:
        return None
    target_weekday = WEEKDAY_MAP[match.group("weekday")]
    prefix = match.group("prefix") or ""
    if prefix == "下":
        days_until_monday = 7 - reference_time.weekday()
        target = reference_time.date() + timedelta(days=days_until_monday + target_weekday)
    else:
        days = target_weekday - reference_time.weekday()
        if days < 0:
            days += 7
        target = reference_time.date() + timedelta(days=days)
    return datetime.combine(target, time.min, tzinfo=BEIJING_TZ)


def _extract_next_week_start(text: str, reference_time: datetime) -> datetime | None:
    if not re.search(r"下(?:周|星期|礼拜)", text):
        return None
    days_until_monday = 7 - reference_time.weekday()
    target = reference_time.date() + timedelta(days=days_until_monday)
    return datetime.combine(target, time.min, tzinfo=BEIJING_TZ)


def _safe_date(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day, tzinfo=BEIJING_TZ)
    except ValueError:
        return None
