from __future__ import annotations

import json
from pathlib import Path

from guga.memory.schema import CoreMemoryRecord
from guga.memory.storage import append_jsonl, read_jsonl


class CoreMemoryStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def append(self, record: CoreMemoryRecord) -> None:
        append_jsonl(self.file_path, record.to_dict())

    def list_active(self) -> list[CoreMemoryRecord]:
        rows = read_jsonl(self.file_path)
        records = [CoreMemoryRecord.from_dict(row) for row in rows]
        return [record for record in records if record.status == "active"]

    def update_status(self, memory_id: str, status: str) -> bool:
        rows = read_jsonl(self.file_path)
        changed = False
        for row in rows:
            if str(row.get("id")) != memory_id:
                continue
            row["status"] = status
            changed = True
            break

        if changed:
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(text, encoding="utf-8")

        return changed
