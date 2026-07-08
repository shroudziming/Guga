from __future__ import annotations

import json
import re
from hashlib import sha1
from pathlib import Path
from uuid import uuid4

from guga.memory.time_utils import extract_semantic_time, format_beijing, now_beijing, parse_datetime


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
        """Backward-compatible no-op; online timeline writes require LLM route items."""
        _ = user_text, session_id, source_message_ids, created_at
        return {}

    def append_from_route_items(
        self,
        *,
        user_text: str,
        route_items: list[dict],
        session_id: str,
        source_message_ids: list[str],
        created_at: str = "",
    ) -> dict:
        for route_item in route_items:
            fact = self.extract_from_route_item(
                user_text=user_text,
                route_item=route_item,
                session_id=session_id,
                source_message_ids=source_message_ids,
                created_at=created_at,
            )
            if not fact:
                continue
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fact, ensure_ascii=False) + "\n")
            return fact
        return {}

    def load_active(self) -> list[dict]:
        rows: list[dict] = []
        if not self.file_path.exists():
            return rows
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("status", "active") == "active":
                rows.append(payload)
        return rows

    def upsert_consolidated_fact(
        self,
        *,
        payload: dict,
        session_id: str,
        source_message_ids: list[str],
        include_guga_reflection: bool,
    ) -> dict:
        subject = str(payload.get("subject") or "user").strip() or "user"
        predicate = str(payload.get("predicate") or "related_to_event").strip() or "related_to_event"
        obj = str(payload.get("object") or payload.get("summary") or "").strip()
        summary = str(payload.get("summary") or obj).strip()
        if not summary:
            return {}
        semantic_day = str(payload.get("semantic_day") or "").strip()
        fact_id = str(payload.get("id") or payload.get("fact_id") or "").strip()
        if not fact_id:
            stable_key = "|".join([subject, predicate, obj, semantic_day])
            fact_id = f"fact_{sha1(stable_key.encode('utf-8')).hexdigest()[:12]}"
        now = format_beijing(now_beijing())
        valid_at = f"{semantic_day}T00:00:00+08:00" if semantic_day else ""
        fact = {
            "fact_id": fact_id,
            "id": fact_id,
            "type": "timeline_fact",
            "subject": subject,
            "predicate": predicate,
            "object": obj or summary[:120],
            "summary": summary,
            "semantic_text": summary,
            "raw_excerpt": summary,
            "created_at": now,
            "updated_at": now,
            "valid_from": valid_at,
            "valid_to": "",
            "valid_at": valid_at,
            "invalid_at": "",
            "semantic_day": semantic_day,
            "day": semantic_day,
            "time_source": "llm_consolidation",
            "time_granularity": "day" if semantic_day else "",
            "source_session_id": session_id,
            "source_message_ids": [item for item in (payload.get("source_message_ids") or source_message_ids) if item],
            "confidence": self._clamp_float(payload.get("confidence"), 0.85),
            "importance": self._clamp_float(payload.get("importance"), 0.75),
            "memory_strength": 1,
            "retention": 1.0,
            "status": "active",
            "extraction_version": "timeline_fact_consolidation_v1",
        }
        if include_guga_reflection:
            fact["guga_assessment"] = str(payload.get("guga_assessment", "")).strip()
            fact["guga_thought"] = str(payload.get("guga_thought", "")).strip()

        rows = self._read_rows()
        for index, row in enumerate(rows):
            if str(row.get("id", "")) == fact_id:
                fact["created_at"] = str(row.get("created_at") or now)
                rows[index] = fact
                self._write_rows(rows)
                return fact
        rows.append(fact)
        self._write_rows(rows)
        return fact

    def extract_from_route_item(
        self,
        *,
        user_text: str,
        route_item: dict,
        session_id: str,
        source_message_ids: list[str],
        created_at: str = "",
    ) -> dict:
        if route_item.get("target") != "timeline_fact":
            return {}
        return self.extract_from_turn(
            user_text=user_text,
            content=str(route_item.get("content", "")),
            confidence=float(route_item.get("confidence", 0.85) or 0.85),
            session_id=session_id,
            source_message_ids=source_message_ids,
            created_at=created_at,
        )

    def extract_from_turn(
        self,
        *,
        user_text: str,
        session_id: str,
        source_message_ids: list[str],
        created_at: str = "",
        content: str = "",
        confidence: float = 0.85,
    ) -> dict:
        text = user_text.strip()
        if not text:
            return {}

        write_time = parse_datetime(created_at) or now_beijing()
        semantic_source = f"{text}\n{content}".strip()
        extracted = extract_semantic_time(semantic_source, reference_time=write_time)
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
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "importance": 0.75,
            "memory_strength": 1,
            "retention": 1.0,
            "status": "active",
            "extraction_version": "timeline_fact_v1",
        }

    def _compact_object(self, text: str) -> str:
        compact = re.sub(r"请你记住[。.!！]*", "", text).strip(" ，。.!！")
        compact = re.sub(r"我在?\d{4}年\d{1,2}月\d{1,2}(?:日|号)?", "", compact)
        compact = re.sub(r"我在?\d{1,2}月\d{1,2}(?:日|号)?", "", compact)
        compact = re.sub(r"(今天|明天|后天|昨天|前天|下周[一二三四五六日天])", "", compact)
        compact = compact.strip(" ，。.!！")
        if compact.startswith("要"):
            compact = compact[1:].strip()
        return compact or text[:80]

    def _read_rows(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        rows: list[dict] = []
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _write_rows(self, rows: list[dict]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    def _clamp_float(self, value: object, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = fallback
        return max(0.0, min(number, 1.0))
