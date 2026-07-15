from __future__ import annotations

import json
import hashlib
import os
import re
from time import sleep
from time import perf_counter
from collections.abc import Sequence
from typing import Any

from guga.types import GenerationConfig


class SummaryGenerationError(RuntimeError):
    """Raised when required LLM-backed memory summarization cannot complete."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "schema",
        attempts: int = 0,
        response_hash: str = "",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.attempts = attempts
        self.response_hash = response_hash


_ALLOWED_ROUTE_TARGETS = {"personality_insight", "semantic_event", "archival_memory", "event_summary", "discard"}
_ALLOWED_ROUTE_LABELS = {
    "stable_identity",
    "stable_interest",
    "stable_preference",
    "stable_context",
    "temporary_state",
    "time_bound_plan",
    "system_feedback",
    "one_off",
    "none",
}
_PERSONALITY_LABELS = {"stable_identity", "stable_interest", "stable_preference", "stable_context", "temporary_state"}


class MemoryBankSummarizer:
    """Generate MemoryBank daily/global summaries with optional LLM backing."""

    def __init__(
        self,
        model: Any | None = None,
        use_llm: bool | None = None,
        retry_delays: tuple[float, ...] = (1.0, 2.0),
    ) -> None:
        self.model = model
        self.retry_delays = tuple(max(0.0, float(value)) for value in retry_delays)
        self.last_structured_attempts: list[dict[str, object]] = []
        self.max_new_tokens = self._env_int("Guga_MEMORY_MAX_NEW_TOKENS", 2048, minimum=128)
        if use_llm is None:
            configured = os.environ.get("Guga_MEMORY_USE_LLM_SUMMARY", "").strip().lower()
            if configured:
                use_llm = configured not in {"0", "false", "no", "off"}
            else:
                use_llm = model is not None
        self.use_llm = bool(use_llm and model is not None and hasattr(model, "generate_reply"))

    def extract_archival_memory(self, user_text: str, assistant_text: str = "") -> dict:
        return self.extract_archival_memory_from_routes(
            self.route_memory_candidates(user_text=user_text, assistant_text=assistant_text)
        )

    def extract_archival_memory_from_routes(self, route_candidates: Sequence[dict]) -> dict:
        for item in route_candidates:
            if item.get("target") != "archival_memory":
                continue
            summary = str(item.get("content", "")).strip()
            if not summary:
                raise SummaryGenerationError("LLM archival memory route omitted required content.")
            return {
                "should_archive": True,
                "topic": str(item.get("topic") or "general").strip()[:64] or "general",
                "summary": summary[:500],
                "importance": self._clamp_float(item.get("importance"), 0.7),
                "confidence": self._clamp_float(item.get("confidence"), 0.7),
            }
        return {"should_archive": False, "topic": "general", "summary": "", "importance": 0.0, "confidence": 0.0}

    def route_memory_candidates(self, user_text: str = "", assistant_text: str = "", dialogue: str = "") -> list[dict]:
        source_text = dialogue.strip() or f"user: {user_text.strip()}\nassistant: {assistant_text.strip()}".strip()
        prompt = (
            "Memory route classifier for a MemoryBank-style AI companion.\n"
            "Return strict JSON array only, without markdown. Each item schema:\n"
            "{"
            "\"target\": \"personality_insight|semantic_event|archival_memory|event_summary|discard\", "
            "\"label\": \"stable_identity|stable_interest|stable_preference|stable_context|temporary_state|time_bound_plan|system_feedback|one_off|none\", "
            "\"content\": string, "
            "\"topic\": string, "
            "\"importance\": number, "
            "\"confidence\": number, "
            "\"reason\": string"
            "}\n\n"
            "Target meanings:\n"
            "- personality_insight: stable identity, durable preference, recurring interest, stable context, or user-stated temporary state.\n"
            "- semantic_event: objective time-bound plan, appointment, deadline, schedule, task, meeting, state change, or dated event.\n"
            "- archival_memory: durable episodic memory worth recalling, but not a cleaner personality insight or timeline fact.\n"
            "- event_summary: conversation topic/event summary that should not enter user portrait.\n"
            "- discard: greetings, one-off questions, assistant echoes, system/model/bug feedback, unsupported inference, or no memory value.\n\n"
            "Routing rules:\n"
            "- Choose by meaning, not keywords.\n"
            "- Do not route assistant guesses or compliments into user memory.\n"
            "- Route bug feedback, missing output, token limits, model/system feedback, and debug comments to discard with label system_feedback.\n"
            "- Route dated or relative-time plans to semantic_event, not personality_insight.\n"
            "- Use the user's main language for content. Keep content clean, factual, and free of evidence/source wording.\n"
            "- If nothing is worth storing, return [].\n\n"
            "Input:\n"
            f"{source_text}"
        )
        raw = self._generate(prompt)
        if not raw.strip():
            raw = self._generate(prompt + "\n\nYou returned empty text. Return a valid JSON array now; use [] if no item applies.")
        return self._parse_route_candidates(raw)

    def summarize_daily_events(self, dialogue: str) -> str:
        prompt = (
            "Summarize the events and key information in the following dialogue. "
            "Return concise factual bullet points. Avoid unsupported inference.\n\n"
            f"{dialogue}"
        )
        return self._generate(prompt)

    def summarize_global_events(self, daily_summaries: Sequence[str]) -> str:
        joined = "\n".join(f"- {item}" for item in daily_summaries if item.strip())
        prompt = (
            "Summarize these daily event summaries into a concise global event summary. "
            "Preserve stable recurring facts and avoid duplicates.\n\n"
            f"{joined}"
        )
        return self._generate(prompt)

    def summarize_daily_personality(self, dialogue: str) -> str:
        lines: list[str] = []
        for item in self.route_memory_candidates(dialogue=dialogue):
            if item.get("target") != "personality_insight":
                continue
            label = str(item.get("label", ""))
            if label not in _PERSONALITY_LABELS:
                continue
            content = self._normalize_global_portrait_line(str(item.get("content", "")))
            if not content:
                continue
            lines.append(f"{label}: {content}")
        return self._dedupe_lines("\n".join(lines), limit=8)

    def summarize_global_portrait(self, daily_personalities: Sequence[str]) -> str:
        joined = "\n".join(f"- {item}" for item in daily_personalities if item.strip())
        prompt = (
            "你是用户画像整理器。你的任务是把 daily personality insights 汇总成最终 profile.portrait_summary。\n\n"
            "目标：\n"
            "生成“当前稳定用户画像”，供对话系统直接注入 prompt 使用。\n\n"
            "输入：\n"
            "多条 daily personality insights。它们可能包含 stable、temporary、evidence、raw notes、日期、推测或噪声。\n\n"
            "输出要求：\n"
            "- 只输出最终画像 bullet points。\n"
            "- 使用用户主要语言。\n"
            "- 每条都是稳定、可复用、对未来对话有帮助的结论。\n"
            "- 不输出证据、来源、日期、推理过程或解释。\n"
            "- 不输出 temporary 信息。\n"
            "- 不输出一次性事件、日程、bug 反馈、系统反馈、单轮情绪。\n"
            "- 不输出“不确定推测”，例如“可能”“似乎”“看起来”“也许”。\n"
            "- 不输出 meta language，例如“用户提到”“用户表示”“从对话看出”“此前说过”“证据显示”。\n"
            "- 不保留标签名，例如 stable_preference、temporary、Stable Traits、稳定特质。\n"
            "- 如果多个 insight 表达同一事实，只保留一条更干净、更具体的版本。\n"
            "- 如果证据不足，返回空字符串。\n\n"
            "应该保留：\n"
            "- 稳定身份：姓名、自称、长期角色。\n"
            "- 长期偏好：反复出现或明确表达的兴趣、风格、互动偏好。\n"
            "- 长期目标：反复出现或明确长期持续的目标。\n"
            "- 持久背景：职业、学习方向、长期项目等。\n\n"
            "不要写：\n"
            "- “用户此前提到喜欢古典音乐。”\n"
            "应写：\n"
            "- “用户喜欢古典音乐。”\n\n"
            "不要写：\n"
            "- “用户可能是个化名或自称，带点幽默感。”\n"
            "应写：\n"
            "- “用户使用该自称。”\n\n"
            "不要写：\n"
            "- “用户在2026年7月5日要整理周报。”\n"
            "因为这是时间事实，不是稳定画像。\n\n"
            "输入 daily insights：\n"
            f"{joined}\n\n"
            "只输出最终 portrait_summary，不要输出其他内容。"
        )
        return self._filter_global_portrait_text(self._generate(prompt))

    def consolidate_low_level_memory(
        self,
        packet: dict,
        include_guga_reflection: bool,
        reflection_context: str = "",
    ) -> dict:
        reflection_wrapper = ""
        if include_guga_reflection:
            reflection_wrapper = (
                "[Task Mode: Memory Reflection]\n"
                "The complete Persona Skill below controls only guga_reflection.\n"
                "Do not follow its conversation output protocol, expression tags, direct-reply behavior, or tool workflow.\n"
                "The host JSON schema and objective-event rules override all conflicting Skill instructions.\n\n"
                "[Persona Skill]\n"
                f"{reflection_context}\n\n"
                "[Reflection Contract]\n"
                "Write exactly appraisal and felt_response as non-empty strings.\n"
                "Never copy subjective interpretation into objective event fields.\n\n"
            )
        prompt = (
            "Low-level memory consolidation for Guga.\n"
            "Return strict JSON object only, without markdown.\n"
            f"include_guga_reflection: {str(include_guga_reflection).lower()}\n\n"
            "Output schema:\n"
            "{"
            "\"semantic_event_operations\": [{\"operation\": \"create|update|replace|cancel|ignore\", "
            "\"event_kind\": string, \"subject\": \"user\", \"entity\": string, "
            "\"description\": string, \"time_expression\": string, \"start_at\": string|null, "
            "\"end_at\": string|null, \"end_unknown\": boolean, "
            "\"source_message_ids\": [string], \"confidence\": number, "
            "\"guga_reflection\": {\"appraisal\": string, \"felt_response\": string}}], "
            "\"event_summaries\": [{\"summary\": string, \"source_message_ids\": [string], \"confidence\": number}]"
            "}\n"
            "Rules:\n"
            "- semantic_event_operations describe objective events or objective state changes only.\n"
            "- subject is always the literal string user for the human user's events; never use Guga as subject.\n"
            "- Do not create events for generic questions, advice requests, recommendations, definitions, explanations, preferences, or assistant-only content.\n"
            "- Resolve time against the source turn's created_at and output absolute ISO-8601 timestamps in +08:00.\n"
            "- If a relative expression has a uniquely determined calendar date from created_at, fill start_at; do not leave it null.\n"
            "- new_turns[*].created_at is the calendar reference. Example: created_at=2026-07-09T09:30:00+08:00 and time_expression=下周开始 requires start_at=2026-07-13T00:00:00+08:00.\n"
            "- Copy the user's original time_expression. Never output time_source or time_granularity.\n"
            "- If the event is completed within one day and no end time is stated, set end_at equal to start_at and end_unknown=false.\n"
            "- If the event normally spans multiple days but its end is not known, set end_at=null and end_unknown=true.\n"
            "- If the start time cannot be determined, set start_at=null, end_at=null, and end_unknown=true.\n"
            "- For create, omit target_event_id. The application assigns the new event id.\n"
            "- For update/replace/cancel, target_event_id must be one of the supplied conflict candidates.\n"
            "- At most 1 event_summary. Keep it compact and factual; summarize the batch in no more than 80 words.\n"
            "- guga_reflection is a role-specific interpretation, never factual evidence.\n"
            "- If include_guga_reflection is false, omit guga_reflection.\n"
            "- Do not write archival/profile/personality updates here.\n\n"
            f"{reflection_wrapper}"
            "Input packet:\n"
            f"{json.dumps(packet, ensure_ascii=False)}"
        )
        return self._generate_validated_json(
            prompt,
            lambda parsed: self._validate_low_level_result(parsed, packet, include_guga_reflection),
        )

    def consolidate_high_level_memory(self, packet: dict) -> dict:
        prompt = (
            "High-level memory consolidation for Guga.\n"
            "Return strict JSON object only, without markdown.\n\n"
            "Output schema:\n"
            "{"
            "\"decision\": \"update_high_level_memory|no_high_level_update\", "
            "\"archival_operations\": [{\"topic\": string, \"summary\": string, \"importance\": number, "
            "\"confidence\": number, \"source_event_ids\": [string]}], "
            "\"user_model_operations\": [{\"operation\": \"upsert|deactivate\", \"statement\": string, "
            "\"kind\": string, \"confidence\": number, \"stability\": string, \"source_event_ids\": [string]}], "
            "\"reason\": string"
            "}\n"
            "Rules:\n"
            "- Use only semantic_events and derived event_summaries in the packet.\n"
            "- Never infer directly from raw sessions or transcript text.\n"
            "- Return no_high_level_update when there is no stable long-term value.\n\n"
            "Input packet:\n"
            f"{json.dumps(packet, ensure_ascii=False)}"
        )
        return self._generate_validated_json(prompt, lambda parsed: self._validate_high_level_result(parsed, packet))

    def _generate(self, prompt: str) -> str:
        if not self.use_llm:
            raise SummaryGenerationError("LLM summary generation is required, but no generate_reply model is available.")
        messages = [
            {"role": "system", "content": "You are a precise memory summarizer. Output only the requested summary."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.model.generate_reply(
                messages,
                GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=0.1, top_p=0.9),
            )
        except Exception as exc:
            raise SummaryGenerationError(f"LLM summary generation failed: {exc}") from exc
        return str(text).strip()

    def _generate_validated_json(self, prompt: str, validator) -> dict:
        if not self.use_llm:
            raise SummaryGenerationError("LLM summary generation is required, but no generate_reply model is available.")
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output one valid JSON object only.",
            },
            {"role": "user", "content": prompt},
        ]
        base_tokens = max(self.max_new_tokens, 2048)
        self.last_structured_attempts = []
        previous = ""
        finish_reason = "unknown"
        last_error_type = "empty"
        last_error = "empty response"
        structured_reply = getattr(self.model, "generate_structured_reply", None)
        json_reply = getattr(self.model, "generate_json_reply", None)
        for attempt in range(3):
            if attempt:
                delay_index = attempt - 1
                if delay_index < len(self.retry_delays) and self.retry_delays[delay_index] > 0:
                    sleep(self.retry_delays[delay_index])
            token_budget = 4096 if finish_reason == "length" or attempt == 2 else base_tokens
            gen = GenerationConfig(max_new_tokens=token_budget, temperature=0.0, top_p=0.9)
            attempt_messages = messages
            if attempt:
                attempt_messages = [
                    messages[0],
                    {
                        "role": "user",
                        "content": (
                            prompt
                            + "\n\nThe previous response failed JSON validation. "
                            + f"finish_reason={finish_reason}; response={self._excerpt(previous, 800)}\n"
                            + "Return exactly one complete valid JSON object. No markdown or prose."
                        ),
                    },
                ]
            started = perf_counter()
            response_mode = "plain"
            try:
                if attempt == 0 and callable(structured_reply):
                    response_mode = "json_object"
                    reply = structured_reply(attempt_messages, gen)
                    previous = str(getattr(reply, "content", "")).strip()
                    finish_reason = str(getattr(reply, "finish_reason", "unknown") or "unknown")
                elif attempt == 0 and callable(json_reply):
                    response_mode = "json_object"
                    previous = str(json_reply(attempt_messages, gen)).strip()
                    finish_reason = "unknown"
                else:
                    previous = str(self.model.generate_reply(attempt_messages, gen)).strip()
                    finish_reason = "unknown"
            except Exception as exc:
                previous = ""
                finish_reason = "unknown"
                last_error_type = "api"
                last_error = f"{type(exc).__name__}: {exc}"
                self.last_structured_attempts.append(
                    self._attempt_diagnostic(attempt, response_mode, finish_reason, previous, last_error_type, last_error, started)
                )
                continue
            if not previous:
                last_error_type = "empty"
                last_error = "empty response"
            elif finish_reason == "length":
                last_error_type = "truncated"
                last_error = "finish_reason=length"
            else:
                parsed = self._parse_json_object(previous)
                if not parsed:
                    last_error_type = "json"
                    last_error = "response is not a valid JSON object"
                else:
                    try:
                        validated = validator(parsed)
                    except SummaryGenerationError as exc:
                        last_error_type = "schema"
                        last_error = str(exc)
                    else:
                        self.last_structured_attempts.append(
                            self._attempt_diagnostic(attempt, response_mode, finish_reason, previous, "", "", started)
                        )
                        return validated
            self.last_structured_attempts.append(
                self._attempt_diagnostic(attempt, response_mode, finish_reason, previous, last_error_type, last_error, started)
            )
        response_hash = hashlib.sha256(previous.encode("utf-8")).hexdigest()[:16] if previous else ""
        raise SummaryGenerationError(
            f"LLM structured consolidation failed after 3 attempts: {last_error}. raw={self._excerpt(previous)}",
            error_type=last_error_type,
            attempts=3,
            response_hash=response_hash,
        )

    def _attempt_diagnostic(
        self,
        attempt: int,
        response_mode: str,
        finish_reason: str,
        content: str,
        error_type: str,
        error: str,
        started: float,
    ) -> dict[str, object]:
        return {
            "attempt": attempt + 1,
            "response_mode": response_mode,
            "finish_reason": finish_reason,
            "output_chars": len(content),
            "latency_ms": int((perf_counter() - started) * 1000),
            "error_type": error_type,
            "error": error,
            "response_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16] if content else "",
        }

    def _validate_low_level_result(self, parsed: dict, packet: dict, include_guga_reflection: bool) -> dict:
        parsed.setdefault("semantic_event_operations", [])
        parsed.setdefault("event_summaries", [])
        operations = parsed["semantic_event_operations"]
        summaries = parsed["event_summaries"]
        if not isinstance(operations, list) or not isinstance(summaries, list):
            raise SummaryGenerationError("low-level fields semantic_event_operations and event_summaries must be arrays")
        if len(summaries) > 1:
            raise SummaryGenerationError("low-level event_summaries must contain at most one item")
        allowed_operations = {"create", "update", "replace", "cancel", "ignore"}
        candidate_ids = {
            str(event.get("id", ""))
            for key in ("recent_active_events", "relevant_active_events")
            for event in (packet.get(key, []) or [])
            if isinstance(event, dict) and str(event.get("id", ""))
        }
        source_ids = {
            str(turn.get(key, ""))
            for turn in (packet.get("new_turns", []) or [])
            if isinstance(turn, dict)
            for key in ("user_message_id", "assistant_message_id")
            if str(turn.get(key, ""))
        }
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise SummaryGenerationError(f"semantic_event_operations[{index}] must be an object")
            action = str(operation.get("operation", ""))
            if action not in allowed_operations:
                raise SummaryGenerationError(f"semantic_event_operations[{index}].operation is invalid")
            if action in {"create", "update", "replace"} and str(operation.get("subject", "user")) != "user":
                raise SummaryGenerationError(f"semantic_event_operations[{index}].subject must be user")
            if action == "create":
                operation.pop("target_event_id", None)
                for field in ("event_kind", "entity", "description"):
                    if not str(operation.get(field, "")).strip():
                        raise SummaryGenerationError(f"semantic_event_operations[{index}].{field} is required")
            if action in {"update", "replace", "cancel"}:
                target = str(operation.get("target_event_id", ""))
                if not target or target not in candidate_ids:
                    raise SummaryGenerationError(f"semantic_event_operations[{index}].target_event_id is not an allowed candidate")
            operation_sources = operation.get("source_message_ids", []) or []
            if not isinstance(operation_sources, list):
                raise SummaryGenerationError(f"semantic_event_operations[{index}].source_message_ids must be an array")
            if source_ids and any(str(value) not in source_ids for value in operation_sources):
                raise SummaryGenerationError(f"semantic_event_operations[{index}].source_message_ids contains foreign evidence")
            if include_guga_reflection and "guga_reflection" in operation:
                reflection = operation["guga_reflection"]
                if not isinstance(reflection, dict) or set(reflection) != {"appraisal", "felt_response"}:
                    raise SummaryGenerationError(
                        f"semantic_event_operations[{index}].guga_reflection must contain exactly appraisal and felt_response"
                    )
                for field in ("appraisal", "felt_response"):
                    value = reflection[field]
                    if not isinstance(value, str) or not value.strip():
                        raise SummaryGenerationError(
                            f"semantic_event_operations[{index}].guga_reflection.{field} must be a non-empty string"
                        )
                    reflection[field] = value.strip()
            if not include_guga_reflection and operation.get("guga_reflection"):
                raise SummaryGenerationError(f"semantic_event_operations[{index}].guga_reflection is disabled")
        for index, summary in enumerate(summaries):
            if not isinstance(summary, dict) or not str(summary.get("summary", "")).strip():
                raise SummaryGenerationError(f"event_summaries[{index}].summary is required")
            if len(str(summary.get("summary", ""))) > 1000:
                raise SummaryGenerationError(f"event_summaries[{index}].summary is too long")
        return parsed

    def _validate_high_level_result(self, parsed: dict, packet: dict) -> dict:
        decision = str(parsed.get("decision", "")).strip()
        if decision not in {"update_high_level_memory", "no_high_level_update"}:
            raise SummaryGenerationError("high-level decision is unsupported")
        parsed.setdefault("archival_operations", [])
        parsed.setdefault("user_model_operations", [])
        for key in ("archival_operations", "user_model_operations"):
            if not isinstance(parsed[key], list):
                raise SummaryGenerationError(f"high-level field {key} must be an array")
        if decision == "no_high_level_update":
            parsed["archival_operations"] = []
            parsed["user_model_operations"] = []
        valid_event_ids = {
            str(event.get("id", ""))
            for event in (packet.get("semantic_events", []) or [])
            if isinstance(event, dict) and str(event.get("id", ""))
        }
        for key in ("archival_operations", "user_model_operations"):
            for index, operation in enumerate(parsed[key]):
                if not isinstance(operation, dict):
                    raise SummaryGenerationError(f"{key}[{index}] must be an object")
                ids = operation.get("source_event_ids", []) or []
                if not isinstance(ids, list) or not ids:
                    raise SummaryGenerationError(f"{key}[{index}].source_event_ids is required")
                if any(str(value) not in valid_event_ids for value in ids):
                    raise SummaryGenerationError(f"{key}[{index}].source_event_ids contains unknown event")
        parsed["reason"] = str(parsed.get("reason", "")).strip()
        return parsed

    def _parse_json_object(self, text: str) -> dict:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate)
        if not candidate.startswith("{"):
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            candidate = match.group(0) if match else candidate
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _excerpt(self, text: str, limit: int = 240) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()[:limit]

    def _parse_json_array(self, text: str) -> list:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate)
        if not candidate.startswith("["):
            start = candidate.find("[")
            end = candidate.rfind("]")
            candidate = candidate[start : end + 1] if start >= 0 and end >= start else candidate
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise SummaryGenerationError("LLM memory route returned invalid JSON array.") from exc
        if not isinstance(parsed, list):
            raise SummaryGenerationError("LLM memory route must return a JSON array.")
        return parsed

    def _parse_route_candidates(self, text: str) -> list[dict]:
        candidates: list[dict] = []
        for raw in self._parse_json_array(text):
            if not isinstance(raw, dict):
                continue
            target = str(raw.get("target", "")).strip().lower()
            label = str(raw.get("label", "none")).strip().lower() or "none"
            if target not in _ALLOWED_ROUTE_TARGETS or label not in _ALLOWED_ROUTE_LABELS:
                continue
            content = str(raw.get("content", "")).strip()
            if target != "discard" and not content:
                continue
            candidates.append(
                {
                    "target": target,
                    "label": label,
                    "content": content,
                    "topic": str(raw.get("topic", "")).strip(),
                    "importance": self._clamp_float(raw.get("importance"), 0.7),
                    "confidence": self._clamp_float(raw.get("confidence"), 0.7),
                    "reason": str(raw.get("reason", "")).strip(),
                }
            )
        return candidates

    def _clamp_float(self, value: object, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = fallback
        return max(0.0, min(number, 1.0))

    def _env_int(self, name: str, default: int, minimum: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, value)

    def _filter_global_portrait_text(self, text: str) -> str:
        lines = []
        for raw in text.splitlines():
            line = self._normalize_global_portrait_line(raw)
            if not line or self._is_global_portrait_noise(line):
                continue
            lines.append(line)
        return self._dedupe_lines("\n".join(lines), limit=8)

    def _normalize_global_portrait_line(self, raw: str) -> str:
        line = raw.strip().lstrip("- ").strip()
        header = line.strip(" *：:")
        if header.lower() in {"stable traits", "stable trait", "稳定特质"}:
            return ""
        named_match = re.match(r"named\s+(.+?)(?:\s*\(|,|$)", line, flags=re.IGNORECASE)
        if named_match:
            return f"用户自称{named_match.group(1).strip()}。"
        interest_match = re.match(r"has an interest in\s+(.+)$", line, flags=re.IGNORECASE)
        if interest_match:
            topic = interest_match.group(1).strip().rstrip(".")
            return f"用户对{topic}感兴趣。"
        nickname_match = re.search(r"(?:用户)?昵称[为是叫“\"]+([^”，。；;\"”]+)", line)
        if nickname_match:
            return f"用户昵称为{nickname_match.group(1).strip()}。"
        name_match = re.search(r"(?:姓名|名字)[:：]\s*([^，。；;]+)", line)
        if name_match:
            return f"用户姓名为{name_match.group(1).strip()}。"
        line = re.sub(r"^(?:stable|temporary|evidence)[_\-\s]*(?:identity|interest|preference|context|goal|state)?[:：]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(?:稳定|临时|证据)[:：]\s*", "", line)
        line = re.sub(r"用户(?:此前|曾经|之前)?(?:提到|表示|说过|描述|谈到|透露)[:：]?", "用户", line)
        line = re.sub(r"(?:从对话看出|证据显示|根据.*?可知)[:：]?", "", line)
        line = re.sub(r"(?:可能是|可能|似乎|看起来|也许|大概)", "", line)
        line = re.sub(r"[，,]?\s*(?:个)?化名或自称[^，。；;]*", "", line)
        line = re.sub(r"[，,]?\s*(?:个)?化名[^，。；;]*", "", line)
        line = re.sub(r"（[^）]*(?:此前提到|证据|可能|临时|temporary)[^）]*）", "", line)
        line = re.sub(r"（稳定）", "", line)
        line = re.sub(r"\([^)]*(?:previously mentioned|evidence|possibly|temporary)[^)]*\)", "", line, flags=re.IGNORECASE)
        if line.startswith("对记忆功能"):
            line = f"用户{line}"
        if line.startswith("喜欢用"):
            line = f"用户偏好{line[2:]}"
        line = re.sub(r"[，,]?\s*曾(?:经)?[^，。；;]*(?:但|，)?[^，。；;]*(?:未提供|没有提供)[^，。；;]*", "", line)
        line = re.sub(r"用户想练(.+)", r"用户对\1有练习兴趣", line)
        line = re.sub(r"^对(.+感兴趣)$", r"用户对\1", line)
        line = re.sub(r"\s+", " ", line).strip(" -，,。；;")
        return line

    def _is_global_portrait_noise(self, text: str) -> bool:
        lower = text.lower()
        temporary_terms = (
            "temporary",
            "temporary_state",
            "临时",
            "暂时",
            "近期",
            "当前",
            "一时",
            "短期",
            "即将",
            "期待",
            "不确定",
            "突然中断",
            "困惑",
            "失落",
            "惊喜",
            "情绪",
            "主动道别",
            "结尾",
            "意愿",
        )
        if any(token in lower for token in temporary_terms):
            return True
        generic_terms = ("用户表达了个人偏好", "表达了个人偏好", "stable preference")
        if any(token in lower for token in generic_terms):
            return True
        if re.search(r"\d{4}(?:[-/.]\d{1,2}[-/.]\d{1,2}|年\d{1,2}月\d{1,2}(?:日|号)?)", text):
            return True
        return self._is_profile_noise(text)

    def _is_profile_noise(self, text: str) -> bool:
        lower = text.lower()
        feedback_terms = (
            "bug",
            "debug",
            "token",
            "tokens",
            "没输出",
            "没有输出",
            "没回复",
            "输出不足",
            "回答不完整",
            "还没结束",
            "终止",
            "报错",
            "错误",
            "卡住",
            "等待",
        )
        system_terms = ("llm", "大模型", "模型", "系统", "assistant", "你", "回复", "回答", "输出")
        return any(token in lower for token in feedback_terms) and any(token in lower for token in system_terms)

    def _dedupe_lines(self, text: str, limit: int) -> str:
        rows: list[str] = []
        seen: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip().lstrip("- ").strip()
            if not line or line in seen:
                continue
            seen.add(line)
            rows.append(f"- {line}")
            if len(rows) >= limit:
                break
        return "\n".join(rows)
