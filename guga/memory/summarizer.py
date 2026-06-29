from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from guga.types import GenerationConfig


class SummaryGenerationError(RuntimeError):
    """Raised when required LLM-backed memory summarization cannot complete."""


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
        raw = self._generate(prompt)
        parsed = self._parse_json_object(raw)
        if not parsed:
            raise SummaryGenerationError("LLM archival memory extraction returned invalid JSON.")

        should_archive = bool(parsed.get("should_archive", False))
        summary = str(parsed.get("summary") or "").strip()
        if should_archive and not summary:
            raise SummaryGenerationError("LLM archival memory extraction omitted required summary.")
        topic = str(parsed.get("topic") or "general").strip() or "general"
        return {
            "should_archive": should_archive,
            "topic": topic[:64],
            "summary": summary[:500],
            "importance": self._clamp_float(parsed.get("importance"), 0.7),
            "confidence": self._clamp_float(parsed.get("confidence"), 0.7),
        }

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
        prompt = (
            "你是用户画像候选提取器。请只基于 user messages 提取 memory-worthy user profile observations。\n\n"
            "参考实践：像 Mem0 一样只抽取用户明确表达的事实/偏好；像 Graphiti 一样不要从弱证据推断偏好、习惯或长期特质。\n\n"
            "输出格式：\n"
            "- 每行一个 bullet。\n"
            "- 必须使用用户主要语言。\n"
            "- 每行以 stable_identity、stable_interest、stable_preference、stable_context、temporary_state 之一开头。\n"
            "- 如果没有足够画像信息，返回空字符串。\n\n"
            "stable 只允许：身份/自称、长期偏好、反复兴趣、长期目标、职业/学习/家庭等持久背景。\n"
            "temporary_state 只允许：用户自己表达的短期情绪、状态或互动需求。\n\n"
            "禁止：\n"
            "- 不要从 assistant 的复述、建议、道歉、夸奖或猜测中抽取用户画像。\n"
            "- 不要把时间事实、日程、deadline、某天要做什么写入 personality_insights；这些属于 timeline_facts。\n"
            "- 不要把 event summary、对话主题、一次性问题写成用户画像。\n"
            "- 不要把 bug 反馈、系统反馈、输出问题、模型问题、测试记忆机制写成用户画像。\n"
            "- 不要输出证据语言，例如“此前提到”“用户表示”“从对话看出”“证据显示”。\n"
            "- 不要输出泛泛标签，例如“用户表达了个人偏好”。必须写出具体偏好。\n"
            "- 不要使用不确定推测，例如“可能”“似乎”“看起来”“也许”。\n\n"
            "示例：\n"
            "输入 user: 我最近在读科幻小说《沙丘》。\n"
            "输出 - stable_interest: 用户对科幻小说感兴趣。\n"
            "输入 user: 你刚才没有输出，有 bug。\n"
            "输出 空字符串。\n"
            "输入 user: 我在2026年7月5日要整理周报。\n"
            "输出 空字符串。\n\n"
            "Dialogue:\n"
            f"{dialogue}"
        )
        return self._filter_daily_personality_text(self._generate(prompt))

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

    def _filter_daily_personality_text(self, text: str) -> str:
        lines: list[str] = []
        for raw in text.splitlines():
            line = self._normalize_daily_personality_line(raw)
            if not line:
                continue
            if self._is_daily_personality_noise(line):
                continue
            lines.append(line)
        return self._dedupe_lines("\n".join(lines), limit=8)

    def _normalize_daily_personality_line(self, raw: str) -> str:
        line = raw.strip().lstrip("- ").strip()
        if not line:
            return ""
        match = re.match(r"^(stable|temporary)[_\-\s]*(identity|interest|preference|context|goal|state|trait)?[:：]\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            return ""
        kind = match.group(1).lower()
        subtype = (match.group(2) or "").lower()
        label = "temporary_state" if kind == "temporary" else f"stable_{subtype}"
        allowed_labels = {"stable_identity", "stable_interest", "stable_preference", "stable_context", "temporary_state"}
        if label not in allowed_labels:
            return ""
        line = match.group(3).strip()
        line = self._normalize_global_portrait_line(line)
        if not line:
            return ""
        return f"{label}: {line}"

    def _is_daily_personality_noise(self, text: str) -> bool:
        payload = re.sub(r"^(?:stable|temporary)[_\-\s]*(?:identity|interest|preference|context|goal|state|trait|observation)?[:：]\s*", "", text, flags=re.IGNORECASE)
        if self._is_profile_noise(payload):
            return True
        if self._is_schedule_or_time_fact(payload):
            return True
        generic_terms = ("用户表达了个人偏好", "表达了个人偏好", "用户表达了偏好", "用户有偏好")
        if any(token in payload for token in generic_terms):
            return True
        return False

    def _is_schedule_or_time_fact(self, text: str) -> bool:
        lower = text.lower()
        absolute_date = re.search(
            r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}(?:日|号)?|\d{1,2}月\d{1,2}(?:日|号)?)",
            text,
        )
        if absolute_date:
            return True

        relative_time = re.search(
            r"(?:今天|明天|后天|昨天|前天|今晚|明早|明晚|今早|本周|这周|下周|上周|本月|下月|"
            r"周[一二三四五六日天]|星期[一二三四五六日天]|上午|下午|晚上|早上|中午|凌晨|"
            r"\b(?:today|tomorrow|yesterday|tonight|next week|last week|this week)\b)",
            lower,
        )
        if not relative_time:
            return False

        schedule_cues = (
            "要",
            "需要",
            "打算",
            "计划",
            "准备",
            "将",
            "会去",
            "去",
            "参加",
            "开会",
            "见面",
            "提交",
            "完成",
            "复查",
            "考试",
            "面试",
            "deadline",
            "due",
            "appointment",
            "meeting",
            "submit",
        )
        return any(cue in lower for cue in schedule_cues)

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
