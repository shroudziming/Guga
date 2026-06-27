from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from guga.memory.time_utils import extract_semantic_time, format_beijing, now_beijing, parse_datetime


_PLAN_KEYWORDS = (
    "要",
    "计划",
    "安排",
    "日程",
    "提交",
    "见面",
    "开会",
    "考试",
    "约",
    "提醒",
    "deadline",
    "meeting",
    "schedule",
    "plan",
)


class TimelineFactStore:
    """Store sparse time-bound facts separate from lossy summaries."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def append_from_turn(
        self,
        *,
        user_text: str,
        session_id: str,
        source_message_ids: list[str],
        created_at: str = "",
    ) -> dict:
        fact = self.extract_from_turn(
            user_text=user_text,
            session_id=session_id,
            source_message_ids=source_message_ids,
            created_at=created_at,
        )
        if not fact:
            return {}
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(fact, ensure_ascii=False) + "\n")
        return fact

    def extract_from_turn(
        self,
        *,
        user_text: str,
        session_id: str,
        source_message_ids: list[str],
        created_at: str = "",
    ) -> dict:
        text = user_text.strip()
        if not text or not self._looks_like_time_bound_plan(text):
            return {}

        write_time = parse_datetime(created_at) or now_beijing()
        extracted = extract_semantic_time(text, reference_time=write_time)
        if extracted is None:
            return {}
        valid_at, time_source, granularity = extracted
        valid_at_text = format_beijing(valid_at)
        created_at_text = format_beijing(write_time)
        semantic_day = valid_at.date().isoformat()
        obj = self._compact_object(text)
        fact_id = f"fact_{uuid4().hex[:10]}"
        semantic_text = f"用户在{semantic_day}有时间相关安排：{obj}"
        return {
            "fact_id": fact_id,
            "id": fact_id,
            "type": "timeline_fact",
            "subject": "user",
            "predicate": "has_time_bound_plan",
            "object": obj,
            "summary": semantic_text,
            "semantic_text": semantic_text,
            "raw_excerpt": text,
            "created_at": created_at_text,
            "updated_at": created_at_text,
            "valid_from": valid_at_text,
            "valid_to": "",
            "valid_at": valid_at_text,
            "invalid_at": "",
            "semantic_day": semantic_day,
            "day": semantic_day,
            "time_source": time_source,
            "time_granularity": granularity,
            "source_session_id": session_id,
            "source_message_ids": [item for item in source_message_ids if item],
            "confidence": 0.85,
            "importance": 0.75,
            "memory_strength": 1,
            "retention": 1.0,
            "status": "active",
            "extraction_version": "timeline_fact_v1",
        }

    def _looks_like_time_bound_plan(self, text: str) -> bool:
        if extract_semantic_time(text) is None:
            return False
        lower = text.lower()
        return any(keyword in lower for keyword in _PLAN_KEYWORDS)

    def _compact_object(self, text: str) -> str:
        compact = re.sub(r"请你记住[。.!！]*", "", text).strip(" ，。.!！")
        compact = re.sub(r"我在?\d{4}年\d{1,2}月\d{1,2}(?:日|号)?", "", compact)
        compact = re.sub(r"我在?\d{1,2}月\d{1,2}(?:日|号)?", "", compact)
        compact = re.sub(r"(今天|明天|后天|昨天|前天|下周[一二三四五六日天])", "", compact)
        compact = compact.strip(" ，。.!！")
        if compact.startswith("要"):
            compact = compact[1:].strip()
        return compact or text[:80]
