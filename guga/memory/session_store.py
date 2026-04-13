from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from guga.memory.clock import today_bucket
from guga.memory.schema import MessageRecord
from guga.memory.storage import append_jsonl, read_jsonl


class SessionStore:
    def __init__(self, sessions_root: Path) -> None:
        self.sessions_root = sessions_root

    def create_session_id(self) -> str:
        return f"sess_{uuid4().hex[:12]}"

    def append(self, record: MessageRecord) -> None:
        append_jsonl(self._session_file(record.session_id), record.to_dict())

    def read_session(self, session_id: str) -> list[MessageRecord]:
        rows = read_jsonl(self._session_file(session_id))
        return [MessageRecord.from_dict(row) for row in rows]

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_root / today_bucket() / f"{session_id}.jsonl"
