from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.manager import MemoryManager
from guga.memory.summarizer import MemoryBankSummarizer
from guga.rag.schemas import RetrievalHit


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class PromptSummaryModel:
    def generate_reply(self, messages, gen):
        _ = gen
        prompt = messages[-1]["content"]
        if "用户画像候选提取器" in prompt:
            if "你刚才没有输出" in prompt:
                return ""
            return "- stable_interest: 用户对蝴蝶刀感兴趣。"
        if "用户画像整理器" in prompt:
            return "\n".join(
                [
                    "- 用户对蝴蝶刀有练习兴趣。",
                    "- temporary_state: 用户近期可能存在压力或情绪波动。",
                ]
            )
        if "Extract one long-term memory candidate" in prompt:
            return '{"should_archive": true, "topic": "general", "summary": "LLM summary", "importance": 0.7, "confidence": 0.8}'
        return "- LLM generated summary"


class MemoryQualityPlanTest(unittest.TestCase):
    def test_current_turn_is_weakened_and_not_reinforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_advisor",
                    "type": "episodic",
                    "summary": "用户下周三要和导师见面。",
                    "raw_excerpt": "下周三和导师见面",
                    "created_at": "2099-01-01T00:00:00+00:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "importance": 0.8,
                    "confidence": 0.9,
                    "status": "active",
                },
            )

            session_id = "sess_current"
            manager.record_user_message(session_id, "你记得我下周三要去做什么吗")
            context = manager.prepare_context("你记得我下周三要去做什么吗", session_id=session_id)

            self.assertEqual(context.hits[0].id, "mem_advisor")
            current_hits = [hit for hit in context.hits if hit.is_current_turn]
            self.assertEqual(len(current_hits), 1)
            self.assertLess(current_hits[0].score, context.hits[0].score)

            session_memory = json.loads((memory_root / "session_memories.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(session_memory["memory_strength"], 1)

    def test_low_score_memories_are_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            manager.memory_min_score = 0.3
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_weak",
                    "type": "episodic",
                    "summary": "alpha detail",
                    "raw_excerpt": "alpha detail",
                    "created_at": "2099-01-01T00:00:00+00:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "importance": 0.0,
                    "confidence": 0.0,
                    "status": "active",
                },
            )

            context = manager.prepare_context("alpha beta gamma delta epsilon", session_id="sess_probe")

            self.assertEqual(context.hits, [])

    def test_debug_report_includes_score_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs: list[str] = []
            memory_root = Path(tmp)
            manager = MemoryManager(
                memory_root=memory_root,
                top_k=4,
                recency_weight=0.2,
                enable_semantic=False,
                debug=True,
                debug_sink=logs.append,
            )
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260509",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-09",
                    "summary": "当天讨论了导师安排。",
                    "raw_excerpt": "导师安排",
                    "created_at": "2026-05-09T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_may9",
                    "source_message_ids": ["msg_may9"],
                    "importance": 0.75,
                    "confidence": 0.8,
                    "status": "active",
                },
            )

            manager.prepare_context("2026-05-09那天的导师安排", session_id="sess_probe")

            retrieve_done = next(line for line in logs if "retrieve_done" in line)
            memory_raw = retrieve_done.split("memory_raw=", 1)[1].rsplit(" latency_ms=", 1)[0]
            payload = json.loads(memory_raw)
            components = payload[0]["score_components"]
            self.assertIn("lexical_overlap", components)
            self.assertIn("recency_bonus", components)
            self.assertIn("importance_bonus", components)
            self.assertIn("confidence_bonus", components)
            self.assertIn("temporal_adjustment", components)
            self.assertIn("final_score", components)

    def test_semantic_and_lexical_hits_are_fused_by_max_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=2, recency_weight=0.0, enable_semantic=False)
            record = {
                "id": "mem_mix",
                "type": "episodic",
                "summary": "lexical beta",
                "raw_excerpt": "lexical beta",
                "created_at": "2099-01-01T00:00:00+00:00",
                "last_recalled_at": "2099-01-01T00:00:00+00:00",
                "memory_strength": 1,
                "retention": 1.0,
                "source_session_id": "sess_old",
                "source_message_ids": ["msg_old"],
                "importance": 0.0,
                "confidence": 0.0,
                "status": "active",
            }
            lexical_hit = manager._to_hit(record, 0.8)
            semantic_hit = RetrievalHit(
                chunk_id="chunk_mem_mix",
                text="semantic beta",
                score=0.2,
                source_type="episodic",
                source_id="mem_mix",
                source_session_id="sess_old",
                source_message_id="msg_old",
                created_at="2099-01-01T00:00:00+00:00",
            )

            hits = manager._merge_memory_hits(
                [semantic_hit],
                [lexical_hit],
                [record],
                current_turn_ids=set(),
                time_hints={},
                session_id="sess_probe",
            )

            self.assertEqual(hits[0].id, "mem_mix")
            self.assertEqual(hits[0].score, 0.8)
            self.assertEqual(hits[0].semantic_score, 0.2)
            self.assertEqual(hits[0].lexical_score, 0.8)
            self.assertEqual(hits[0].score_source, "lexical")

    def test_explicit_date_query_prioritizes_daily_event_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260509",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-09",
                    "summary": "当天讨论了导师安排。",
                    "raw_excerpt": "导师安排",
                    "created_at": "2026-05-09T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_may9",
                    "source_message_ids": ["msg_may9"],
                    "importance": 0.75,
                    "confidence": 0.8,
                    "status": "active",
                },
            )
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_other_day",
                    "type": "episodic",
                    "summary": "那天的对话比较长。",
                    "raw_excerpt": "那天的对话",
                    "created_at": "2026-05-10T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_may10",
                    "source_message_ids": ["msg_may10"],
                    "status": "active",
                },
            )
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_wrong_day_strong",
                    "type": "episodic",
                    "summary": "2026-05-09那天的对话 那天的对话 那天的对话",
                    "raw_excerpt": "2026-05-09那天的对话",
                    "created_at": "2026-05-10T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_wrong",
                    "source_message_ids": ["msg_wrong"],
                    "importance": 1.0,
                    "confidence": 1.0,
                    "status": "active",
                },
            )

            context = manager.prepare_context("2026-05-09那天的对话", session_id="sess_probe")
            hit_ids = [hit.id for hit in context.hits]

            self.assertEqual(context.hits[0].id, "evt_daily_20260509")
            self.assertNotIn("mem_wrong_day_strong", hit_ids)

    def test_recent_and_last_session_time_references_are_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260508",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-08",
                    "summary": "旧会话内容。",
                    "created_at": "2026-05-08T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "status": "active",
                },
            )
            _append_jsonl(
                memory_root / "event_summaries.jsonl",
                {
                    "id": "evt_daily_20260510",
                    "type": "event_summary",
                    "scope": "daily",
                    "day": "2026-05-10",
                    "summary": "最近一次历史会话内容。",
                    "created_at": "2026-05-10T12:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+00:00",
                    "source_session_id": "sess_latest",
                    "source_message_ids": ["msg_latest"],
                    "status": "active",
                },
            )
            last_context = manager.prepare_context("上次我们聊了什么", session_id="sess_now")
            self.assertEqual(last_context.hits[0].id, "evt_daily_20260510")
            self.assertEqual([hit.id for hit in last_context.hits], ["evt_daily_20260510"])

            manager.record_user_message("sess_now", "我们先聊蝴蝶刀")
            manager.record_assistant_message("sess_now", "刚才你说什么：蝴蝶刀练习安排")
            manager.record_user_message("sess_now", "刚才你说什么")
            recent_context = manager.prepare_context("刚才你说什么", session_id="sess_now")

            self.assertTrue(recent_context.hits[0].summary.startswith("assistant:"))

    def test_portrait_query_uses_profile_without_episodic_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_root = Path(tmp)
            (memory_root / "profile.json").write_text(
                json.dumps({"portrait_summary": "- 用户自称叔本明。\n- 用户想练蝴蝶刀。"}, ensure_ascii=False),
                encoding="utf-8",
            )
            _append_jsonl(
                memory_root / "archival_memory.jsonl",
                {
                    "id": "mem_noisy_identity",
                    "type": "episodic",
                    "summary": "我是谁 我是谁 我是谁",
                    "raw_excerpt": "我是谁",
                    "created_at": "2099-01-01T00:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+08:00",
                    "memory_strength": 1,
                    "source_session_id": "sess_old",
                    "source_message_ids": ["msg_old"],
                    "importance": 1.0,
                    "confidence": 1.0,
                    "status": "active",
                },
            )
            manager = MemoryManager(memory_root=memory_root, top_k=4, recency_weight=0.0, enable_semantic=False)

            context = manager.prepare_context("你觉得我是谁？", session_id="sess_probe")

            self.assertEqual(context.hits, [])
            self.assertIn("叔本明", context.user_portrait)

    def test_portrait_summary_ignores_one_off_bug_feedback(self) -> None:
        summarizer = MemoryBankSummarizer(model=PromptSummaryModel())

        daily = summarizer.summarize_daily_personality("user: 你刚才没有输出，有 bug")
        global_portrait = summarizer.summarize_global_portrait(
            [
                daily,
                "- temporary_state: 用户近期可能存在压力或情绪波动。",
                "- stable_interest: 用户想练蝴蝶刀。",
            ]
        )

        self.assertEqual(daily, "")
        self.assertNotIn("bug", global_portrait.lower())
        self.assertNotIn("temporary", global_portrait.lower())
        self.assertIn("蝴蝶刀", global_portrait)

    def test_portrait_summary_strips_evidence_language_and_labels(self) -> None:
        class GlobalDirtyModel:
            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "\n".join(
                    [
                        "- **Stable Traits:**",
                        "- Named 叔本明 (Shu Benming), self-referred.",
                        "- stable_identity: 用户此前提到自己叫叔本明，可能是个化名。",
                        "- stable_interest: 用户此前提到想练蝴蝶刀。",
                        "- temporary: 用户在2026年7月5日要整理周报。",
                        "- 对即将与导师见面感到期待和些许不确定。",
                    ]
                )

        summarizer = MemoryBankSummarizer(model=GlobalDirtyModel())

        global_portrait = summarizer.summarize_global_portrait(
            [
                "- **Stable Traits:**",
                "- Named 叔本明 (Shu Benming), self-referred.",
                "- stable_identity: 用户此前提到自己叫叔本明，可能是个化名。",
                "- stable_interest: 用户此前提到想练蝴蝶刀。",
                "- temporary: 用户在2026年7月5日要整理周报。",
                "- 对即将与导师见面感到期待和些许不确定。",
            ]
        )

        self.assertIn("叔本明", global_portrait)
        self.assertIn("蝴蝶刀", global_portrait)
        self.assertNotIn("Stable Traits", global_portrait)
        self.assertNotIn("Named", global_portrait)
        self.assertNotIn("Has an interest", global_portrait)
        self.assertNotIn("stable_identity", global_portrait)
        self.assertNotIn("stable_interest", global_portrait)
        self.assertNotIn("temporary", global_portrait)
        self.assertNotIn("此前提到", global_portrait)
        self.assertNotIn("可能", global_portrait)
        self.assertNotIn("化名", global_portrait)
        self.assertNotIn("2026年7月5日", global_portrait)
        self.assertNotIn("即将", global_portrait)

    def test_daily_personality_prompt_uses_user_messages_only(self) -> None:
        class CaptureModel:
            def __init__(self) -> None:
                self.prompt = ""

            def generate_reply(self, messages, gen):
                self.prompt = messages[-1]["content"]
                return "- stable_interest: 用户对蝴蝶刀感兴趣。"

        model = CaptureModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True)

        result = summarizer.summarize_daily_personality(
            "user: 我最近想练蝴蝶刀。\nassistant: 你看起来很有毅力，也喜欢冒险。"
        )

        self.assertIn("只基于 user messages", model.prompt)
        self.assertIn("不要从 assistant 的复述", model.prompt)
        self.assertIn("时间事实", model.prompt)
        self.assertIn("蝴蝶刀", result)

    def test_daily_personality_filters_dirty_llm_output(self) -> None:
        class DirtyModel:
            def generate_reply(self, messages, gen):
                return "\n".join(
                    [
                        "- stable_preference: 用户表达了个人偏好。",
                        "- stable_trait: 用户反馈你没有输出，有 bug。",
                        "- stable_goal: 用户在2026年7月5日整理周报。",
                        "- stable_interest: 用户此前提到想练蝴蝶刀。",
                        "- temporary_state: 用户近期有点焦虑。",
                    ]
                )

        summarizer = MemoryBankSummarizer(model=DirtyModel(), use_llm=True)

        result = summarizer.summarize_daily_personality("user: 我想练蝴蝶刀，最近有点焦虑。")

        self.assertIn("蝴蝶刀", result)
        self.assertIn("焦虑", result)
        self.assertNotIn("表达了个人偏好", result)
        self.assertNotIn("bug", result.lower())
        self.assertNotIn("2026年7月5日", result)
        self.assertNotIn("此前提到", result)

    def test_daily_personality_filters_generic_schedule_without_hardcoded_examples(self) -> None:
        class ScheduleModel:
            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "\n".join(
                    [
                        "- stable_context: 用户明天下午要去医院复查。",
                        "- stable_context: 用户下周需要参加小组讨论。",
                        "- stable_preference: 用户偏好提前规划学习节奏。",
                        "- stable_interest: 用户对科幻小说感兴趣。",
                    ]
                )

        summarizer = MemoryBankSummarizer(model=ScheduleModel(), use_llm=True)

        result = summarizer.summarize_daily_personality("user: 我明天下午要去医院复查，最近也在读科幻小说。")

        self.assertNotIn("医院复查", result)
        self.assertNotIn("小组讨论", result)
        self.assertIn("规划学习节奏", result)
        self.assertIn("科幻小说", result)

    def test_daily_personality_does_not_infer_labels_from_keywords(self) -> None:
        class UnlabeledModel:
            def generate_reply(self, messages, gen):
                return "\n".join(
                    [
                        "- 用户自称叔本明。",
                        "- 用户想练蝴蝶刀。",
                        "- 用户有点焦虑。",
                    ]
                )

        summarizer = MemoryBankSummarizer(model=UnlabeledModel(), use_llm=True)

        result = summarizer.summarize_daily_personality("user: 我是叔本明，我想练蝴蝶刀。")

        self.assertEqual(result, "")

    def test_summary_requires_llm_model(self) -> None:
        summarizer = MemoryBankSummarizer()

        with self.assertRaises(RuntimeError):
            summarizer.summarize_daily_events("user: hello")

    def test_summary_raises_when_llm_api_fails(self) -> None:
        class FailingModel:
            def generate_reply(self, messages, gen):
                _ = messages, gen
                raise RuntimeError("api unavailable")

        summarizer = MemoryBankSummarizer(model=FailingModel())

        with self.assertRaises(RuntimeError):
            summarizer.summarize_daily_personality("user: 我喜欢科幻小说。")

    def test_summary_empty_llm_output_does_not_fallback(self) -> None:
        class EmptyModel:
            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "   "

        summarizer = MemoryBankSummarizer(model=EmptyModel())

        self.assertEqual(summarizer.summarize_global_events(["- user: hello"]), "")

    def test_archival_extraction_requires_parseable_llm_json(self) -> None:
        class InvalidJsonModel:
            def generate_reply(self, messages, gen):
                _ = messages, gen
                return "not json"

        summarizer = MemoryBankSummarizer(model=InvalidJsonModel())

        with self.assertRaises(RuntimeError):
            summarizer.extract_archival_memory("我叫小明")


if __name__ == "__main__":
    unittest.main()
