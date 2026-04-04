from __future__ import annotations


class ChatHistory:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def clear(self) -> None:
        self.messages.clear()

    def as_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    def _trim(self) -> None:
        max_len = self.max_turns * 2
        if len(self.messages) > max_len:
            self.messages = self.messages[-max_len:]
