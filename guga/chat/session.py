from __future__ import annotations

import json
import os
from collections.abc import Callable
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Any

from guga.chat.history import ChatHistory
from guga.chat.turn_trace import TurnTrace
from guga.memory import MemoryManager
from guga.memory.agent_identity import AgentIdentity
from guga.memory.time_utils import now_beijing_iso
from guga.tools import (
    ToolCall,
    ToolRegistry,
    ToolStreamText,
    ToolStreamToolCalls,
    default_tool_registry,
    encode_tool_result,
    memory_time_search_tool,
)
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
        max_turns: int = 45,
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
        self.memory_manager = memory_manager or MemoryManager(
            model=model,
            debug=debug,
            debug_sink=debug_sink,
            agent_identity=AgentIdentity(
                agent_id="default",
                reflection_context="default persona",
                persona_source="builtin:chat_session_default",
                persona_fingerprint="builtin:chat_session_default",
            ),
        )
        self.session_id = session_id or self.memory_manager.session_store.create_session_id()
        self.tool_registry = tool_registry.copy() if tool_registry is not None else default_tool_registry()
        self._active_turn_user_text = ""
        self._active_turn_reference_time = ""
        self.tool_registry.add(memory_time_search_tool(self._execute_memory_time_search))
        self.max_tool_rounds = self._env_int("Guga_MAX_TOOL_ROUNDS", 3, minimum=0, maximum=8)
        self._debug("session_ready")

    def reply(self, user_input: str, finalize_memory: bool = True, created_at: str | None = None) -> str:
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
        reference_time = created_at or now_beijing_iso()
        self.history.add_user(user_input)
        user_message_id = self.memory_manager.record_user_message(
            session_id=self.session_id,
            text=user_input,
            created_at=reference_time,
        )
        trace = TurnTrace(
            turn_id=f"turn_{user_message_id}",
            user_message_id=user_message_id,
            reference_time=reference_time,
        )

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")
        self._debug(f"system_prompt={json.dumps(system_prompt, ensure_ascii=False)}")

        messages = [{"role": "system", "content": self._tool_system_prompt(system_prompt, reference_time)}]
        messages.extend(self.history.as_messages())

        generate_started = perf_counter()
        self._debug("model_generate_start")
        self._active_turn_user_text = user_input
        self._active_turn_reference_time = reference_time
        try:
            answer = self._generate_reply_with_optional_tools(messages, trace=trace)
            if not str(answer).strip():
                self._debug("model_generate_empty_retry")
                answer = self._generate_reply_with_optional_tools(
                    messages,
                    generation=self._retry_generation_config(),
                    trace=trace,
                )
            if not str(answer).strip():
                answer = "刚才没有生成出有效回复，可以再说一遍吗？"
        finally:
            self._active_turn_user_text = ""
            self._active_turn_reference_time = ""
        generate_elapsed_ms = int((perf_counter() - generate_started) * 1000)
        self._debug(f"model_generate_done latency_ms={generate_elapsed_ms}")

        self.history.add_assistant(answer)
        assistant_message_id = self.memory_manager.record_assistant_message(
            session_id=self.session_id,
            text=answer,
            created_at=reference_time,
        )
        self.memory_manager.attach_turn_metadata(
            session_id=self.session_id,
            user_message_id=user_message_id,
            metadata=trace.as_metadata(assistant_message_id),
        )
        if finalize_memory:
            self.memory_manager.finalize_turn_async(self.session_id)
            self._debug("finalize_queued")
        else:
            self._debug("finalize_skipped")
        return answer

    def reply_stream(
        self,
        user_input: str,
        cancel_event: Event | None = None,
        created_at: str | None = None,
    ) -> Iterator[str]:
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
        reference_time = created_at or now_beijing_iso()
        self.history.add_user(user_input)
        user_message_id = self.memory_manager.record_user_message(
            session_id=self.session_id,
            text=user_input,
            created_at=reference_time,
        )
        trace = TurnTrace(
            turn_id=f"turn_{user_message_id}",
            user_message_id=user_message_id,
            reference_time=reference_time,
        )

        context_started = perf_counter()
        memory_context = self.memory_manager.prepare_context(user_text=user_input, session_id=self.session_id)
        context_elapsed_ms = int((perf_counter() - context_started) * 1000)
        self._debug(f"prepare_context_done latency_ms={context_elapsed_ms} hits={len(memory_context.hits)}")

        system_prompt = self.memory_manager.compose_system_prompt(self.system_prompt, memory_context)
        self._debug("prompt_assemble_done")
        self._debug(f"system_prompt={json.dumps(system_prompt, ensure_ascii=False)}")

        messages = [{"role": "system", "content": self._tool_system_prompt(system_prompt, reference_time)}]
        messages.extend(self.history.as_messages())

        chunks: list[str] = []

        stream_fn = getattr(self.model, "generate_reply_stream", None)
        self._active_turn_user_text = user_input
        self._active_turn_reference_time = reference_time
        try:
            if self._can_use_tools():
                generate_started = perf_counter()
                self._debug("model_generate_start")
                stream_tools_fn = getattr(self.model, "generate_reply_with_tools_stream", None)
                if callable(stream_tools_fn):
                    for chunk in self._generate_reply_with_optional_tools_stream(
                        messages,
                        trace=trace,
                        cancel_event=cancel_event,
                    ):
                        chunks.append(chunk)
                        yield chunk
                else:
                    answer = self._generate_reply_with_optional_tools(messages, trace=trace)
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

            if cancel_event is not None and cancel_event.is_set():
                return
            answer = "".join(chunks).strip()
            if not answer:
                self._debug("model_generate_empty_retry")
                try:
                    answer = self._generate_reply_with_optional_tools(
                        messages,
                        generation=self._retry_generation_config(),
                        trace=trace,
                    ).strip()
                except Exception as exc:
                    self._debug(f"model_generate_empty_retry_failed reason={exc}")
                    answer = ""
                if cancel_event is not None and cancel_event.is_set():
                    return
                if answer:
                    yield answer
                else:
                    answer = "刚才没有生成出有效回复，可以再说一遍吗？"
                    yield answer
        finally:
            self._active_turn_user_text = ""
            self._active_turn_reference_time = ""

        if cancel_event is not None and cancel_event.is_set():
            return
        self.history.add_assistant(answer)
        assistant_message_id = self.memory_manager.record_assistant_message(
            session_id=self.session_id,
            text=answer,
            created_at=reference_time,
        )
        self.memory_manager.attach_turn_metadata(
            session_id=self.session_id,
            user_message_id=user_message_id,
            metadata=trace.as_metadata(assistant_message_id),
        )
        self.memory_manager.finalize_turn_async(self.session_id)
        self._debug("finalize_queued")

    def flush_memory(self) -> dict[str, int]:
        """Force consolidation of pending completed turns for this session."""
        return self.memory_manager.flush_session_memory(self.session_id)

    def settle_memory_for_shutdown(self) -> dict[str, object]:
        """Finish queued writeback and force-consolidate the session tail."""
        self.memory_manager.wait_for_background_tasks()
        return self.memory_manager.consolidate_until_settled(self.session_id)

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
        trace: TurnTrace,
        generation: GenerationConfig | None = None,
    ) -> str:
        gen = generation or self.generation
        if not self._can_use_tools():
            return self.model.generate_reply(messages, gen)

        # Keep tool results in the turn-owned list so an empty-answer retry resumes it.
        tool_messages = messages
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
                result = self._execute_tool_call(call, trace, round_index)
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

    def _generate_reply_with_optional_tools_stream(
        self,
        messages: list[dict],
        trace: TurnTrace,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        # Share the same accumulated context with the synchronous retry path.
        tool_messages = messages
        tools = self.tool_registry.openai_tools()
        generate_stream_with_tools = getattr(self.model, "generate_reply_with_tools_stream")

        for round_index in range(self.max_tool_rounds):
            round_chunks: list[str] = []
            tool_calls = []

            for event in generate_stream_with_tools(tool_messages, self.generation, tools, cancel_event=cancel_event):
                if isinstance(event, ToolStreamText):
                    if event.content:
                        round_chunks.append(event.content)
                        yield event.content
                    continue
                if isinstance(event, ToolStreamToolCalls):
                    tool_calls = event.tool_calls

            if not tool_calls:
                return

            assistant_message = {
                "role": "assistant",
                "content": "".join(round_chunks),
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ],
            }
            tool_messages.append(assistant_message)
            for call in tool_calls:
                result = self._execute_tool_call(call, trace, round_index)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": encode_tool_result(result),
                    }
                )

        self._debug("tool_call_max_rounds_exceeded")
        fallback = self.model.generate_reply(tool_messages, self.generation)
        if fallback:
            yield fallback

    def _can_use_tools(self) -> bool:
        return (
            self.max_tool_rounds > 0
            and self.tool_registry.has_tools()
            and callable(getattr(self.model, "generate_reply_with_tools", None))
        )

    def _execute_tool_call(self, call: ToolCall, trace: TurnTrace, round_index: int) -> dict[str, Any]:
        result = self.tool_registry.execute(call)
        trace.record_tool(call, result)
        self._debug(
            "tool_call "
            f"round={round_index + 1} name={call.name} ok={bool(result.get('ok'))} "
            f"turn_id={trace.turn_id} tool_call_id={call.id} "
            f"args={json.dumps(call.arguments, ensure_ascii=False)}"
        )
        return result

    def _execute_memory_time_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._active_turn_user_text or not self._active_turn_reference_time:
            return {
                "ok": False,
                "type": "time_context",
                "error": {
                    "code": "NO_ACTIVE_TURN",
                    "message": "time search requires an active turn",
                    "query_index": -1,
                },
            }
        return self.memory_manager.search_time_context(
            session_id=self.session_id,
            user_text=self._active_turn_user_text,
            reference_time=self._active_turn_reference_time,
            time_queries=args.get("time_queries"),
        )

    def _tool_system_prompt(self, base_prompt: str, reference_time: str) -> str:
        if not self._can_use_tools():
            return base_prompt
        return (
            base_prompt
            + "\n\n[Tool Use]\n"
            + "你可以在需要精确时间解析、读取项目文件、列目录或执行已启用工具时调用工具。"
            + "工具结果是内部证据，收到结果后继续用自然语言回答用户。"
            + "\n\n[Current Time]\n"
            + f"reference_time={reference_time}\n"
            + "timezone=Asia/Hong_Kong (+08:00)\n\n"
            + "[Temporal Memory Tool]\n"
            + "When the user states, changes, cancels, compares, or asks about time-bound information, "
            + "resolve every relevant expression to an absolute timezone-aware left-closed/right-open interval "
            + "and call memory_time_search.\n"
            + "Use one normalized interval for all relevant time fields. Interpret the result labels as follows:\n"
            + "- start_at/end_at: when an event occurs.\n"
            + "- valid_at/invalid_at: when a fact is valid.\n"
            + "- time_window_start/time_window_end: the period covered by an event summary.\n"
            + "- temporal_context.time_queries: time mentioned or resolved in that conversation turn.\n"
            + "- created_at: when the message was created, not when an event described by the message occurred.\n"
            + "Use matched_time_field and time_meaning together with the user's question to decide which evidence is relevant.\n"
            + "If a time reference cannot be resolved from the current time and conversation context, ask the user "
            + "to clarify instead of inventing an interval.\n"
            + "The time tool is one normal tool among all available tools. Continue the Agent Loop after any tool call, "
            + "and finish only when no tool_calls remain."
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
