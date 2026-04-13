from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guga.memory.clock import now_iso


@dataclass
class ShortTermItem:
    role: str
    content: str
    created_at: str = field(default_factory=now_iso)
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """短期记忆容器：保存带元信息的最近对话片段。"""

    def __init__(self, max_items: int = 20) -> None:
        self.max_items = max_items
        self.items: list[ShortTermItem] = []

    def add(
        self,
        text: str,
        role: str = "user",
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.items.append(
            ShortTermItem(
                role=role,
                content=text,
                topic=topic,
                metadata=metadata or {},
            )
        )
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def snapshot(self) -> list[ShortTermItem]:
        return list(self.items)
