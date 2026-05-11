from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from guga.types import GenerationConfig


class MemoryBankSummarizer:
    """Generate MemoryBank daily/global summaries with optional LLM backing."""

    def __init__(self, model: Any | None = None, use_llm: bool | None = None) -> None:
        self.model = model
        self.max_new_tokens = self._env_int("Guga_MEMORY_MAX_NEW_TOKENS", 512, minimum=128)
        if use_llm is None:
            configured = os.environ.get("Guga_MEMORY_USE_LLM_SUMMARY", "").strip().lower()
            if configured:
                use_llm = configured not in {"0", "false", "no", "off"}
            else:
                use_llm = model is not None
        self.use_llm = bool(use_llm and model is not None and hasattr(model, "generate_reply"))

    def extract_archival_memory(self, user_text: str, assistant_text: str = "") -> dict:
        fallback = self._fallback_archival_memory(user_text)
        if not self.use_llm:
            return fallback

        prompt = (
            "Extract one long-term memory candidate from this chat turn for a MemoryBank-style AI companion.\n"
            "Return strict JSON only, without markdown. Schema:\n"
            "{"
            "\"should_archive\": boolean, "
            "\"topic\": string, "
            "\"summary\": string, "
            "\"importance\": number, "
            "\"confidence\": number"
            "}\n"
            "Rules: archive stable facts, preferences, recurring goals, work/school/family context, and notable emotional states. "
            "Do not archive trivial greetings, one-off commands, or unsupported inference. "
            "Use the same language as the user. Keep summary concise and factual.\n\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}"
        )
        raw = self._generate(prompt, fallback=json.dumps(fallback, ensure_ascii=False))
        parsed = self._parse_json_object(raw)
        if not parsed:
            return fallback

        should_archive = bool(parsed.get("should_archive", fallback["should_archive"]))
        summary = str(parsed.get("summary") or fallback["summary"]).strip()
        topic = str(parsed.get("topic") or fallback["topic"]).strip() or "general"
        return {
            "should_archive": should_archive,
            "topic": topic[:64],
            "summary": summary[:500],
            "importance": self._clamp_float(parsed.get("importance"), fallback["importance"]),
            "confidence": self._clamp_float(parsed.get("confidence"), fallback["confidence"]),
        }

    def summarize_daily_events(self, dialogue: str) -> str:
        fallback = self._fallback_event_summary(dialogue)
        if not self.use_llm:
            return fallback
        prompt = (
            "Summarize the events and key information in the following dialogue. "
            "Return concise factual bullet points. Avoid unsupported inference.\n\n"
            f"{dialogue}"
        )
        return self._generate(prompt, fallback=fallback)

    def summarize_global_events(self, daily_summaries: Sequence[str]) -> str:
        joined = "\n".join(f"- {item}" for item in daily_summaries if item.strip())
        fallback = self._dedupe_lines(joined, limit=8)
        if not self.use_llm:
            return fallback
        prompt = (
            "Summarize these daily event summaries into a concise global event summary. "
            "Preserve stable recurring facts and avoid duplicates.\n\n"
            f"{joined}"
        )
        return self._generate(prompt, fallback=fallback)

    def summarize_daily_personality(self, dialogue: str) -> str:
        fallback = self._fallback_personality(dialogue)
        if not self.use_llm:
            return fallback
        prompt = (
            "Based on the following dialogue, summarize the user's personality traits, preferences, "
            "and emotional state. Separate stable traits from temporary emotions.\n\n"
            f"{dialogue}"
        )
        return self._generate(prompt, fallback=fallback)

    def summarize_global_portrait(self, daily_personalities: Sequence[str]) -> str:
        joined = "\n".join(f"- {item}" for item in daily_personalities if item.strip())
        fallback = self._dedupe_lines(joined, limit=8)
        if not self.use_llm:
            return fallback
        prompt = (
            "The following are user traits and emotions observed across multiple days. "
            "Provide a concise, general, non-duplicative user portrait. Preserve uncertainty when evidence is weak.\n\n"
            f"{joined}"
        )
        return self._generate(prompt, fallback=fallback)

    def _generate(self, prompt: str, fallback: str) -> str:
        if self.model is None:
            return fallback
        messages = [
            {"role": "system", "content": "You are a precise memory summarizer. Output only the requested summary."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.model.generate_reply(
                messages,
                GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=0.1, top_p=0.9),
            )
        except Exception:
            return fallback
        text = str(text).strip()
        return text or fallback

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

    def _fallback_archival_memory(self, user_text: str) -> dict:
        text = user_text.strip()
        lower = text.lower()
        stable_triggers = (
            "我叫",
            "我是",
            "我在",
            "工作",
            "喜欢",
            "不喜欢",
            "焦虑",
            "压力",
            "my name is",
            "i am ",
            "i'm ",
            "i work",
            "like",
            "dislike",
            "prefer",
            "stress",
            "anxious",
        )
        should_archive = len(text) >= 12 or any(token in lower for token in stable_triggers)
        return {
            "should_archive": should_archive,
            "topic": "general",
            "summary": f"用户提到：{text}",
            "importance": 0.7,
            "confidence": 0.7,
        }

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

    def _fallback_event_summary(self, dialogue: str) -> str:
        lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
        user_lines = [line for line in lines if line.lower().startswith("user:") or line.startswith("用户:")]
        selected = user_lines or lines
        return self._dedupe_lines("\n".join(selected), limit=6)

    def _fallback_personality(self, dialogue: str) -> str:
        lower = dialogue.lower()
        traits: list[str] = []
        if any(token in lower for token in ("不喜欢", "dislike", "don't like", "讨厌")):
            traits.append("用户表达了明确的负向偏好或互动边界。")
        if any(token in lower for token in ("喜欢", "like", "prefer", "偏好")):
            traits.append("用户表达了个人偏好。")
        if any(token in lower for token in ("焦虑", "压力", "stress", "anxious", "sad", "难过")):
            traits.append("用户近期可能存在压力或情绪波动。")
        if any(token in lower for token in ("工作", "work", "job")):
            traits.append("用户谈到了工作或职业背景。")
        if any(token in lower for token in ("我叫", "my name is", "i am ", "i'm ")):
            traits.append("用户提供了身份相关信息。")
        if not traits:
            return self._fallback_event_summary(dialogue)
        return "\n".join(f"- {item}" for item in traits)

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
