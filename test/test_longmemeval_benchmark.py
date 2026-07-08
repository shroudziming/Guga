from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.benchmark.longmemeval import (
    LONGMEMEVAL_SYSTEM_PROMPT,
    ingest_longmemeval_case,
    ingest_longmemeval_case_replay,
    load_longmemeval_cases,
    run_longmemeval_benchmark,
    run_longmemeval_case,
)
from guga.benchmark.workspace import benchmark_workspace
from guga.memory.manager import MemoryManager
from guga.types import GenerationConfig


class LongMemEvalBenchmarkTest(unittest.TestCase):
    def test_workspace_keeps_benchmark_state_under_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")

            self.assertEqual(workspace.root, root / "longmemeval" / "runs" / "run_001")
            self.assertEqual(workspace.memory_root, workspace.root / "memory")
            self.assertEqual(workspace.debug_reports_dir, workspace.root / "debug_reports")
            self.assertEqual(workspace.documents_dir, workspace.root / "documents")
            self.assertEqual(workspace.results_file, workspace.root / "results.jsonl")
            self.assertEqual(workspace.case_memory_root("q/1"), workspace.root / "cases" / "q_1" / "memory")
            self.assertEqual(
                workspace.case_debug_reports_dir("q/1"),
                workspace.root / "debug_reports" / "q_1",
            )

    def test_load_longmemeval_cases_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "longmemeval.jsonl"
            payload = {
                "question_id": "q1",
                "question_type": "information_extraction",
                "question": "What color does the user like?",
                "answer": "blue",
                "sessions": [
                    [
                        {"role": "user", "content": "I like blue notebooks."},
                        {"role": "assistant", "content": "Noted."},
                    ]
                ],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            cases = load_longmemeval_cases(path)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "q1")
            self.assertEqual(cases[0].question_type, "information_extraction")
            self.assertEqual(cases[0].question, "What color does the user like?")
            self.assertEqual(cases[0].answer, "blue")
            self.assertEqual(cases[0].sessions[0][0].role, "user")
            self.assertEqual(cases[0].sessions[0][0].content, "I like blue notebooks.")

    def test_load_and_ingest_preserves_session_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q_time",
                        "question": "When did the user mention blue notebooks?",
                        "answer": "2026-01-02",
                        "sessions": [
                            {
                                "date": "2026-01-02T09:30:00+08:00",
                                "messages": [{"role": "user", "content": "I like blue notebooks."}],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            case = load_longmemeval_cases(dataset)[0]
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            manager = MemoryManager(
                memory_root=workspace.case_memory_root(case.case_id),
                documents_dir=workspace.documents_dir,
                enable_semantic=False,
            )

            ingest_longmemeval_case(case, manager)

            self.assertEqual(case.sessions[0][0].created_at, "2026-01-02T09:30:00+08:00")
            session_row = json.loads(
                (workspace.case_memory_root(case.case_id) / "sessions" / "q_time_s0.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            memory_row = json.loads(
                (workspace.case_memory_root(case.case_id) / "session_memories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(session_row["created_at"], "2026-01-02T09:30:00+08:00")
            self.assertEqual(memory_row["created_at"], "2026-01-02T09:30:00+08:00")

    def test_ingest_case_writes_only_to_benchmark_memory_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What color does the user like?",
                        "answer": "blue",
                        "sessions": [[{"role": "user", "content": "I like blue notebooks."}]],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            case = load_longmemeval_cases(dataset)[0]
            manager = MemoryManager(
                memory_root=workspace.memory_root,
                documents_dir=workspace.documents_dir,
                enable_semantic=False,
            )

            stats = ingest_longmemeval_case(case, manager)

            self.assertEqual(stats["sessions"], 1)
            self.assertEqual(stats["messages"], 1)
            self.assertTrue((workspace.memory_root / "session_memories.jsonl").exists())
            self.assertTrue((workspace.memory_root / "sessions" / "q1_s0.jsonl").exists())
            self.assertFalse((root / "memory" / "session_memories.jsonl").exists())
            self.assertEqual(manager.documents_dir, workspace.documents_dir)

    def test_ingest_case_indexes_assistant_history_for_assistant_side_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q_assistant",
                        "question": "What reminder did the assistant give?",
                        "answer": "water the basil",
                        "sessions": [
                            [
                                {"role": "user", "content": "Help me plan my plants."},
                                {"role": "assistant", "content": "Remember to water the basil every Monday."},
                            ]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            case = load_longmemeval_cases(dataset)[0]
            manager = MemoryManager(
                memory_root=workspace.case_memory_root(case.case_id),
                documents_dir=workspace.documents_dir,
                enable_semantic=False,
            )

            ingest_longmemeval_case(case, manager)

            rows = [
                json.loads(line)
                for line in (workspace.case_memory_root(case.case_id) / "session_memories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertIn("assistant: Remember to water the basil every Monday.", rows[1]["summary"])

    def test_replay_ingest_finalizes_turns_into_archival_memory(self) -> None:
        class ReplayModel:
            def generate_reply(self, messages, gen):
                _ = gen
                prompt = messages[-1]["content"]
                if "Low-level memory consolidation" in prompt:
                    return json.dumps(
                        {
                            "timeline_facts": [],
                            "event_summaries": [
                                {
                                    "action": "upsert",
                                    "scope": "batch",
                                    "summary": "User likes blue notebooks.",
                                    "source_message_ids": [],
                                    "confidence": 0.9,
                                }
                            ],
                        }
                    )
                if "High-level memory consolidation" in prompt:
                    return json.dumps(
                        {
                            "decision": "update_high_level_memory",
                            "archival_updates": [
                                {
                                    "topic": "preference",
                                    "summary": "User likes blue notebooks.",
                                    "importance": 0.8,
                                    "confidence": 0.9,
                                }
                            ],
                            "profile_updates": [],
                            "personality_insight_updates": [],
                            "reason": "preference",
                        }
                    )
                if "Summarize" in prompt or "summary" in prompt:
                    return "- User likes blue notebooks."
                if "用户画像整理器" in prompt:
                    return "- 用户喜欢蓝色笔记本。"
                return "ok"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q_replay",
                        "question": "What does the user like?",
                        "answer": "blue notebooks",
                        "sessions": [
                            [
                                {"role": "user", "content": "I like blue notebooks."},
                                {"role": "assistant", "content": "I will remember that."},
                            ]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            case = load_longmemeval_cases(dataset)[0]
            manager = MemoryManager(
                memory_root=workspace.case_memory_root(case.case_id),
                model=ReplayModel(),
                documents_dir=workspace.documents_dir,
                enable_semantic=False,
            )

            stats = ingest_longmemeval_case_replay(case, manager)

            self.assertEqual(stats["sessions"], 1)
            self.assertEqual(stats["messages"], 2)
            self.assertEqual(stats["finalized_turns"], 1)
            self.assertEqual(stats["completed_turns"], 1)
            self.assertEqual(stats["consolidation_batches"], 1)
            archival_rows = (workspace.case_memory_root(case.case_id) / "archival_memory.jsonl").read_text(encoding="utf-8")
            self.assertIn("User likes blue notebooks.", archival_rows)

    def test_run_case_uses_benchmark_system_prompt_instead_of_daily_persona(self) -> None:
        class CaptureModel:
            def __init__(self) -> None:
                self.system_prompts: list[str] = []

            def generate_reply(self, messages, gen):
                _ = gen
                prompt = messages[-1]["content"]
                if "Memory route classifier" in prompt:
                    return json.dumps([{"target": "discard", "label": "benchmark_question"}])
                self.system_prompts.append(messages[0]["content"])
                return "blue"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What color does the user like?",
                        "answer": "blue",
                        "sessions": [[{"role": "user", "content": "I like blue notebooks."}]],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            case = load_longmemeval_cases(dataset)[0]
            model = CaptureModel()

            result = run_longmemeval_case(
                case=case,
                model=model,
                workspace=workspace,
                generation=GenerationConfig(),
                debug=False,
                enable_semantic=False,
            )

            self.assertEqual(result["prediction"], "blue")
            self.assertTrue(model.system_prompts)
            self.assertIn(LONGMEMEVAL_SYSTEM_PROMPT, model.system_prompts[0])
            self.assertNotIn("小咕嘎", model.system_prompts[0])

    def test_run_case_skips_question_memory_finalization_for_unreliable_api_summary(self) -> None:
        class BadSummaryModel:
            def generate_reply(self, messages, gen):
                _ = gen
                prompt = messages[-1]["content"]
                if "Memory route classifier" in prompt:
                    return "[{\"target\": \"discard\""
                return "blue"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What color does the user like?",
                        "answer": "blue",
                        "sessions": [[{"role": "user", "content": "I like blue notebooks."}]],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")
            case = load_longmemeval_cases(dataset)[0]

            result = run_longmemeval_case(
                case=case,
                model=BadSummaryModel(),
                workspace=workspace,
                generation=GenerationConfig(),
                debug=False,
                enable_semantic=False,
            )

            self.assertEqual(result["prediction"], "blue")
            self.assertTrue(result["finalize_skipped"])
            self.assertTrue(workspace.results_file.exists())

    def test_run_benchmark_respects_limit_and_writes_shared_results(self) -> None:
        class SimpleModel:
            def generate_reply(self, messages, gen):
                _ = gen
                prompt = messages[-1]["content"]
                if "Memory route classifier" in prompt:
                    return json.dumps([{"target": "discard", "label": "benchmark"}])
                return "answer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "longmemeval.jsonl"
            rows = [
                {
                    "question_id": "q1",
                    "question": "Question one?",
                    "answer": "answer",
                    "sessions": [[{"role": "user", "content": "History one."}]],
                },
                {
                    "question_id": "q2",
                    "question": "Question two?",
                    "answer": "answer",
                    "sessions": [[{"role": "user", "content": "History two."}]],
                },
            ]
            dataset.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            workspace = benchmark_workspace("longmemeval", root=root, run_id="run_001")

            results = run_longmemeval_benchmark(
                dataset_path=dataset,
                model=SimpleModel(),
                workspace=workspace,
                generation=GenerationConfig(),
                limit=1,
                debug=False,
                enable_semantic=False,
            )

            self.assertEqual([item["case_id"] for item in results], ["q1"])
            result_lines = workspace.results_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(result_lines), 1)
            self.assertTrue((workspace.case_memory_root("q1") / "session_memories.jsonl").exists())
            self.assertFalse((workspace.case_memory_root("q2") / "session_memories.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
