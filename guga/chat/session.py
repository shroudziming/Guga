from __future__ import annotations

from collections.abc import Iterator
from threading import Event

from guga.chat.history import ChatHistory
from guga.models import ChatModel
from guga.types import GenerationConfig


class ChatSession:
    def __init__(
        self,
        model: ChatModel,
        system_prompt: str,
        generation: GenerationConfig,
        max_turns: int = 10,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.generation = generation
        self.history = ChatHistory(max_turns=max_turns)

    def reply(self, user_input: str) -> str:
        self.history.add_user(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history.as_messages())

        answer = self.model.generate_reply(messages, self.generation)
        self.history.add_assistant(answer)
        return answer

    def reply_stream(self, user_input: str, cancel_event: Event | None = None) -> Iterator[str]:
        self.history.add_user(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history.as_messages())

        chunks: list[str] = []

        stream_fn = getattr(self.model, "generate_reply_stream", None)
        if callable(stream_fn):
            for chunk in stream_fn(messages, self.generation, cancel_event=cancel_event):
                chunks.append(chunk)
                yield chunk
        else:
            answer = self.model.generate_reply(messages, self.generation)
            chunks.append(answer)
            yield answer

        answer = "".join(chunks).strip()
        self.history.add_assistant(answer)

    def clear(self) -> None:
        self.history.clear()
