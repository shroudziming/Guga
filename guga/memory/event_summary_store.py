from __future__ import annotations

import json
from pathlib import Path

from guga.memory.forgetting import now_iso, normalize_memorybank_fields


class EventSummaryStore:
    """Store derived batch summaries for semantic events."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load_active(self) -> list[dict]:
        rows: list[dict] = []
        for payload in self._read_rows():
            payload = normalize_memorybank_fields(payload)
            if payload.get("status") == "active":
                rows.append(payload)
        return rows

    def upsert_batch_summary(
        self,
        *,
        session_id: str,
        batch_seq: int,
        payload: dict,
        source_message_ids: list[str],
        event_result,
        covered_events: list[dict],
    ) -> dict:
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            return {}
        rows = self._read_rows()
        row_id = str(payload.get("id") or f"evt_{session_id}_batch_{batch_seq}").strip()
        existing = self._find(rows, row_id)
        created_at = str(existing.get("created_at") or now_iso()) if existing else now_iso()
        updated_at = now_iso()
        payload_source_ids = list(payload.get("source_message_ids") or source_message_ids)
        event_ids = [str(event.get("id", "")) for event in covered_events if str(event.get("id", ""))]
        time_values = [
            str(event.get(field, ""))
            for event in covered_events
            for field in ("start_at", "end_at")
            if str(event.get(field, ""))
        ]
        event = normalize_memorybank_fields(
            {
                "id": row_id,
                "type": "event_summary",
                "summary": summary,
                "source_of_truth": False,
                "covered_event_ids": event_ids,
                "created_event_ids": list(event_result.created_event_ids),
                "updated_event_ids": list(event_result.updated_event_ids),
                "deactivated_event_ids": list(event_result.deactivated_event_ids),
                "time_window_start": min(time_values) if time_values else None,
                "time_window_end": max(time_values) if time_values else None,
                "source_session_id": session_id,
                "source_message_ids": [item for item in payload_source_ids if item],
                "created_at": created_at,
                "updated_at": updated_at,
                "last_recalled_at": str(existing.get("last_recalled_at") or created_at) if existing else created_at,
                "memory_strength": int(existing.get("memory_strength", 1) or 1) if existing else 1,
                "importance": float(payload.get("importance", 0.75) or 0.75),
                "confidence": float(payload.get("confidence", 0.8) or 0.8),
                "status": "active",
            }
        )
        self._upsert(rows, event)
        self._write_rows(rows)
        return event

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

