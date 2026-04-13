from __future__ import annotations

from pathlib import Path

from guga.memory.clock import now_iso
from guga.memory.schema import ProfileRecord
from guga.memory.storage import read_json, write_json


class ProfileStore:
    """存储用户结构化档案（本地 JSON）。"""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> ProfileRecord:
        payload = read_json(self.file_path, default=None)
        if not isinstance(payload, dict):
            return ProfileRecord(updated_at=now_iso())
        profile = ProfileRecord.from_dict(payload)
        if not profile.updated_at:
            profile.updated_at = now_iso()
        return profile

    def save(self, profile: ProfileRecord) -> None:
        profile.updated_at = now_iso()
        write_json(self.file_path, profile.to_dict())
