from __future__ import annotations


class ShortTermMemory:
    """阶段二预留：后续可在这里加入摘要策略。"""

    def __init__(self, max_items: int = 20) -> None:
        self.max_items = max_items
        self.items: list[str] = []

    def add(self, text: str) -> None:
        self.items.append(text)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def snapshot(self) -> list[str]:
        return list(self.items)
