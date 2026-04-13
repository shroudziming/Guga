from __future__ import annotations

import json
from pathlib import Path

from guga.memory.clock import now_iso
from guga.memory.schema import ArchivalMemoryRecord
from guga.memory.storage import append_jsonl, read_jsonl


class ArchivalStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def append(self, record: ArchivalMemoryRecord) -> None:
        append_jsonl(self.file_path, record.to_dict())

    def list_all(self, status: str | None = None) -> list[ArchivalMemoryRecord]:
        rows = read_jsonl(self.file_path)
        records = [ArchivalMemoryRecord.from_dict(row) for row in rows]
        if status is None:
            return records
        return [record for record in records if record.status == status]

    def mark_accessed(self, memory_id: str) -> bool:
        rows = read_jsonl(self.file_path)
        changed = False
        for row in rows:
            if str(row.get("id")) != memory_id:
                continue
            row["last_accessed_at"] = now_iso()
            row["access_count"] = int(row.get("access_count", 0)) + 1
            changed = True
            break

        if changed:
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(text, encoding="utf-8")

        return changed
