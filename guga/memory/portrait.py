from __future__ import annotations

import json
from pathlib import Path

from guga.memory.forgetting import now_iso
from guga.memory.summarizer import MemoryBankSummarizer
from guga.memory.time_utils import apply_temporal_fields


class UserPortraitStore:
    """Maintain MemoryBank daily personality insights and a global user portrait."""

    def __init__(self, file_path: Path, daily_file_path: Path | None = None) -> None:
        self.file_path = file_path
        self.daily_file_path = daily_file_path or file_path.with_name("personality_insights.jsonl")

    def load(self) -> dict:
        if not self.file_path.exists():
            return {}
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def refresh_daily_insight(
        self,
        day: str,
        dialogue: str,
        source_session_id: str,
        source_message_ids: list[str],
        summarizer: MemoryBankSummarizer,
    ) -> dict:
        summary = summarizer.summarize_daily_personality(dialogue).strip()
        if not summary:
            return {}

        rows = self._read_daily_rows()
        row_id = f"portrait_daily_{day.replace('-', '')}"
        existing = self._find(rows, row_id)
        created_at = str(existing.get("created_at") or now_iso()) if existing else now_iso()
        updated_at = now_iso()
        payload = apply_temporal_fields(
            {
                "id": row_id,
                "type": "user_portrait",
                "scope": "daily",
                "day": day,
                "summary": summary,
                "raw_excerpt": dialogue[-2000:],
                "source_session_id": source_session_id,
                "source_message_ids": source_message_ids,
                "created_at": created_at,
                "updated_at": updated_at,
                "status": "active",
            },
            text=f"{dialogue}\n{summary}",
            reference_time=updated_at,
        )
        self._upsert(rows, payload)
        self._write_daily_rows(rows)
        return payload

    def refresh_global_portrait(self, summarizer: MemoryBankSummarizer) -> dict:
        rows = [
            row
            for row in self._read_daily_rows()
            if row.get("scope") == "daily" and row.get("status", "active") == "active"
        ]
        daily_summaries = [str(row.get("summary", "")) for row in rows if str(row.get("summary", "")).strip()]
        portrait_summary = summarizer.summarize_global_portrait(daily_summaries).strip()
        profile = self.load()
        profile.setdefault("schema_version", 2)
        profile["updated_at"] = now_iso()
        profile["time_source"] = "transaction_time"
        profile["portrait_summary"] = portrait_summary
        profile["daily_personality_count"] = len(daily_summaries)
        profile["daily_personality_ids"] = [str(row.get("id", "")) for row in rows[-20:]]
        self.save(profile)
        return profile

    def update_from_user_text(self, user_text: str) -> dict:
        """Backward-compatible rule update; prefer refresh_daily/global methods."""
        text = user_text.strip()
        if not text:
            return self.load()

        profile = self.load()
        profile.setdefault("schema_version", 2)
        profile.setdefault("stable_facts", [])
        profile.setdefault("preferences", [])
        profile.setdefault("temporary_states", [])

        lower = text.lower()
        self._maybe_add(profile["stable_facts"], text, ["我叫", "我是", "我在", "工作", "my name is", "i work", "i am ", "i'm "], lower)
        self._maybe_add(profile["preferences"], text, ["喜欢", "不喜欢", "讨厌", "偏好", "like", "dislike", "prefer"], lower)
        self._maybe_add(profile["temporary_states"], text, ["焦虑", "压力", "难过", "开心", "最近", "stress", "anxious", "sad", "recently"], lower)

        profile["updated_at"] = now_iso()
        profile["time_source"] = "transaction_time"
        profile["portrait_summary"] = self._build_summary(profile)
        self.save(profile)
        return profile

    def save(self, profile: dict) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_daily_rows(self) -> list[dict]:
        if not self.daily_file_path.exists():
            return []
        rows: list[dict] = []
        for line in self.daily_file_path.read_text(encoding="utf-8").splitlines():
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

    def _write_daily_rows(self, rows: list[dict]) -> None:
        self.daily_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_file_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    def _find(self, rows: list[dict], row_id: str) -> dict:
        for row in rows:
            if str(row.get("id", "")) == row_id:
                return row
        return {}

    def _upsert(self, rows: list[dict], payload: dict) -> None:
        for index, row in enumerate(rows):
            if str(row.get("id", "")) == str(payload.get("id", "")):
                rows[index] = payload
                return
        rows.append(payload)

    def _maybe_add(self, values: list[str], text: str, triggers: list[str], lower_text: str) -> None:
        if not any(trigger in lower_text for trigger in triggers):
            return
        if text not in values:
            values.append(text)
        del values[:-8]

    def _build_summary(self, profile: dict) -> str:
        parts: list[str] = []
        facts = profile.get("stable_facts", []) or []
        preferences = profile.get("preferences", []) or []
        states = profile.get("temporary_states", []) or []
        if facts:
            parts.append("稳定信息：" + "；".join(str(item) for item in facts[-3:]))
        if preferences:
            parts.append("偏好：" + "；".join(str(item) for item in preferences[-3:]))
        if states:
            parts.append("近期状态：" + "；".join(str(item) for item in states[-3:]))
        return "\n".join(parts)
