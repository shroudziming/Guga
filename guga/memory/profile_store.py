from __future__ import annotations

import json
from pathlib import Path


class ProfileStore:
    """阶段二预留：存储长期偏好信息（本地 JSON）。"""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> dict[str, str]:
        if not self.file_path.exists():
            return {}
        return json.loads(self.file_path.read_text(encoding="utf-8"))

    def save(self, profile: dict[str, str]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
