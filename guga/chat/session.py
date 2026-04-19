from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Any

from guga.chat.history import ChatHistory
from guga.memory import MemoryManager
from guga.types import GenerationConfig

if TYPE_CHECKING:
    from guga.models.base import ChatModel
else:
    ChatModel = Any


class ChatSession:
    def __init__(
        self,
        model: ChatModel,
        system_prompt: str,
        generation: GenerationConfig,
        max_turns: int = 10,
        memory_manager: MemoryManager | None = None,
        session_id: str | None = None,
        debug: bool = False,
        debug_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.generation = generation
        self.history = ChatHistory(max_turns=max_turns)
        self.debug = debug
        self.debug_sink = debug_sink
        self.memory_manager = memory_manager or MemoryManager(model=model, debug=debug, debug_sink=debug_sink)
        self.session_id = session_id or self.memory_manager.session_store.create_session_id()
        self._debug("session_ready")

    def reply(self, user_input: str) -> str:
        self._debug("reply_start")
        self.history.add_user(user_input)
        self.memory_manager.record_user_message(session_id=self.session_id, text=user_input)

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history.as_messages())

        generate_started = perf_counter()
        self._debug("model_generate_start")
        answer = self.model.generate_reply(messages, self.generation)
        generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
        self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")

        self.history.add_assistant(answer)
        self.memory_manager.record_assistant_message(session_id=self.session_id, text=answer)
        self.memory_manager.finalize_turn(self.session_id)
        self._debug("finalize_done")
        return answer

    def reply_stream(self, user_input: str, cancel_event: Event | None = None) -> Iterator[str]:
        self._debug("reply_start")
        self.history.add_user(user_input)
        self.memory_manager.record_user_message(session_id=self.session_id, text=user_input)

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history.as_messages())

        chunks: list[str] = []

        stream_fn = getattr(self.model, "generate_reply_stream", None)
        if callable(stream_fn):
            generate_started = perf_counter()
            self._debug("model_generate_start")
            for chunk in stream_fn(messages, self.generation, cancel_event=cancel_event):
                chunks.append(chunk)
                yield chunk
            generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
            self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")
        else:
            generate_started = perf_counter()
            self._debug("model_generate_start")
            answer = self.model.generate_reply(messages, self.generation)
            chunks.append(answer)
            yield answer
            generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
            self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")

        answer = "".join(chunks).strip()
        self.history.add_assistant(answer)
        self.memory_manager.record_assistant_message(session_id=self.session_id, text=answer)
        self.memory_manager.finalize_turn(self.session_id)
        self._debug("finalize_done")

    def clear(self) -> None:
        self.history.clear()

    def _debug(self, message: str) -> None:
        if not self.debug:
            return
        output = f"[DEBUG][ChatSession][{self.session_id}] {message}"
        if self.debug_sink is not None:
            self.debug_sink(output)
            return
        print(output)
