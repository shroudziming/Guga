from __future__ import annotations

import json
import os
from collections.abc import Callable
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Any

from guga.chat.history import ChatHistory
from guga.memory import MemoryManager
from guga.tools import ToolRegistry, default_tool_registry, encode_tool_result
from guga.types import GenerationConfig

if TYPE_CHECKING:
    from guga.models.base import ChatModel
else:
    ChatModel = Any


class ChatSession:
    """Orchestrate one chat session around the model + memory/RAG pipeline.

    Flow per turn:
    1) ingest user text into in-memory history and persistent session store
    2) retrieve memory/document context via MemoryManager (RAG step)
    3) compose system prompt with retrieved context and generate answer
    4) persist assistant response and queue memory writeback in the background
    """

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
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize session state and bind model/memory dependencies.

        Args:
            model: Chat model implementation (local or API-backed).
            system_prompt: Base persona/system instruction.
            generation: Sampling config used by model generation.
            max_turns: Max recent dialogue turns kept in ChatHistory.
            memory_manager: Optional external memory manager.
            session_id: Optional existing session id for resume scenarios.
            debug: Whether to emit debug logs.
            debug_sink: Optional sink function for debug output.
            tool_registry: Optional callable tool registry for model tool use.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.generation = generation
        self.history = ChatHistory(max_turns=max_turns)
        self.debug = debug
        self.debug_sink = debug_sink
        self.memory_manager = memory_manager or MemoryManager(model=model, debug=debug, debug_sink=debug_sink)
        self.session_id = session_id or self.memory_manager.session_store.create_session_id()
        self.tool_registry = tool_registry or default_tool_registry()
        self.max_tool_rounds = self._env_int("Guga_MAX_TOOL_ROUNDS", 3, minimum=0, maximum=8)
        self._debug("session_ready")

    def reply(self, user_input: str) -> str:
        """Run one non-streaming dialogue turn.

        Upstream input:
            Raw user text from CLI/UI.

        Downstream output:
            Final assistant text returned by model.generate_reply.

        Side effects:
            - Persist user/assistant messages to session storage.
            - Perform RAG retrieval and inject context into system prompt.
            - Queue memory writeback in MemoryManager.finalize_turn_async.
        """
        self._debug("reply_start")
        self.history.add_user(user_input)
        self.memory_manager.record_user_message(session_id=self.session_id, text=user_input)

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")
        self._debug(f"system_prompt={json.dumps(system_prompt, ensure_ascii=False)}")

        messages = [{"role": "system", "content": self._tool_system_prompt(system_prompt)}]
        messages.extend(self.history.as_messages())

        generate_started = perf_counter()
        self._debug("model_generate_start")
        answer = self._generate_reply_with_optional_tools(messages)
        if not str(answer).strip():
            self._debug("model_generate_empty_retry")
            answer = self._generate_reply_with_optional_tools(messages, generation=self._retry_generation_config())
        if not str(answer).strip():
            answer = "刚才没有生成出有效回复，可以再说一遍吗？"
        generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
        self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")

        self.history.add_assistant(answer)
        self.memory_manager.record_assistant_message(session_id=self.session_id, text=answer)
        self.memory_manager.finalize_turn_async(self.session_id)
        self._debug("finalize_queued")
        return answer

    def reply_stream(self, user_input: str, cancel_event: Event | None = None) -> Iterator[str]:
        """Run one streaming dialogue turn and yield output chunks.

        Args:
            user_input: Raw user text from CLI/UI.
            cancel_event: Optional cancellation signal for streaming models.

        Yields:
            Incremental text chunks from model streaming generation.

        Notes:
            Retrieval/prompt assembly is identical to reply(); only generation
            output mode differs (streamed chunks vs single final string).
        """
        self._debug("reply_start")
        self.history.add_user(user_input)
        self.memory_manager.record_user_message(session_id=self.session_id, text=user_input)

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")
        self._debug(f"system_prompt={json.dumps(system_prompt, ensure_ascii=False)}")

        messages = [{"role": "system", "content": self._tool_system_prompt(system_prompt)}]
        messages.extend(self.history.as_messages())

        chunks: list[str] = []

        stream_fn = getattr(self.model, "generate_reply_stream", None)
        if self._can_use_tools():
            generate_started = perf_counter()
            self._debug("model_generate_start")
            answer = self._generate_reply_with_optional_tools(messages)
            chunks.append(answer)
            if answer:
                yield answer
            generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
            self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")
        elif callable(stream_fn):
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
        if not answer:
            self._debug("model_generate_empty_retry")
            try:
                answer = self.model.generate_reply(messages, self._retry_generation_config()).strip()
            except Exception as exc:
                self._debug(f"model_generate_empty_retry_failed reason={exc}")
                answer = ""
            if answer:
                yield answer
            else:
                answer = "刚才没有生成出有效回复，可以再说一遍吗？"
                yield answer

        self.history.add_assistant(answer)
        self.memory_manager.record_assistant_message(session_id=self.session_id, text=answer)
        self.memory_manager.finalize_turn_async(self.session_id)
        self._debug("finalize_queued")

    def clear(self) -> None:
        """Clear only in-memory short history; persisted memory files remain."""
        self.history.clear()

    def _debug(self, message: str) -> None:
        if not self.debug:
            return
        output = f"[DEBUG][ChatSession][{self.session_id}] {message}"
        if self.debug_sink is not None:
            self.debug_sink(output)
            return
        print(output)

    def _retry_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=max(self.generation.max_new_tokens, 1536),
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
        )

    def _generate_reply_with_optional_tools(
        self,
        messages: list[dict],
        generation: GenerationConfig | None = None,
    ) -> str:
        gen = generation or self.generation
        if not self._can_use_tools():
            return self.model.generate_reply(messages, gen)

        tool_messages = [dict(message) for message in messages]
        tools = self.tool_registry.openai_tools()
        generate_with_tools = getattr(self.model, "generate_reply_with_tools")
        for round_index in range(self.max_tool_rounds):
            response = generate_with_tools(tool_messages, gen, tools)
            if not response.tool_calls:
                return response.content

            assistant_message = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in response.tool_calls
                ],
            }
            tool_messages.append(assistant_message)
            for call in response.tool_calls:
                result = self.tool_registry.execute(call)
                self._debug(
                    "tool_call "
                    f"round={round_index + 1} name={call.name} ok={bool(result.get('ok'))} "
                    f"args={json.dumps(call.arguments, ensure_ascii=False)}"
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": encode_tool_result(result),
                    }
                )

        self._debug("tool_call_max_rounds_exceeded")
        return self.model.generate_reply(tool_messages, gen)

    def _can_use_tools(self) -> bool:
        return (
            self.max_tool_rounds > 0
            and self.tool_registry.has_tools()
            and callable(getattr(self.model, "generate_reply_with_tools", None))
        )

    def _tool_system_prompt(self, base_prompt: str) -> str:
        if not self._can_use_tools():
            return base_prompt
        return (
            base_prompt
            + "\n\n[Tool Use]\n"
            + "你可以在需要精确时间解析、读取项目文件、列目录或执行已启用工具时调用工具。"
            + "工具结果是内部证据，收到结果后继续用自然语言回答用户。"
        )

    def _env_int(self, name: str, default: int, minimum: int, maximum: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))
