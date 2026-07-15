from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.manager import MemoryManager
from guga.memory.migration import repair_memory_file, repair_memory_root
from guga.rag.schemas import RetrievalHit


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MemoryMigrationTest(unittest.TestCase):
    def test_repair_file_backfills_temporal_fields_and_marks_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archival_memory.jsonl"
            _append_jsonl(
                path,
                {
                    "id": "mem_meeting",
                    "type": "episodic",
                    "summary": "用户计划在2026年6月20日与导师见面",
                    "created_at": "2026-06-12T17:09:43+08:00",
                    "source_message_ids": ["msg_meeting"],
                    "status": "active",
                },
            )
            _append_jsonl(
                path,
                {
                    "id": "mem_bug",
                    "type": "episodic",
                    "summary": "用户指出助手刚才没有输出，有 bug。",
                    "created_at": "2026-06-12T17:11:31+08:00",
                    "source_message_ids": ["msg_bug"],
                    "status": "active",
                },
            )
            _append_jsonl(
                path,
                {
                    "id": "mem_mojibake",
                    "type": "episodic",
                    "summary": "用户提到：???????2026?5?9???????",
                    "created_at": "2026-06-11T22:23:32+08:00",
                    "source_message_ids": ["msg_bad"],
                    "status": "active",
                },
            )

            stats = repair_memory_file(path)

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["checked"], 3)
            self.assertEqual(stats["updated"], 3)
            self.assertEqual(rows[0]["valid_at"], "2026-06-20T00:00:00+08:00")
            self.assertEqual(rows[0]["semantic_day"], "2026-06-20")
            self.assertEqual(rows[0]["time_source"], "semantic_explicit_date")
            self.assertEqual(rows[1]["type"], "system_feedback")
            self.assertTrue(rows[1]["exclude_from_retrieval"])
            self.assertEqual(rows[2]["status"], "decayed")
            self.assertEqual(rows[2]["noise_reason"], "mojibake")
            self.assertTrue(rows[2]["exclude_from_retrieval"])

    def test_repair_root_covers_memory_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("archival_memory.jsonl", "event_summaries.jsonl", "session_memories.jsonl"):
                _append_jsonl(
                    root / name,
                    {
                        "id": f"mem_{name}",
                        "type": "episodic",
                        "summary": "今天聊了记忆系统",
                        "created_at": "2026-06-12T17:09:43+08:00",
                        "status": "active",
                    },
                )

            stats = repair_memory_root(root)

            self.assertEqual(stats["files"], 3)
            self.assertEqual(stats["checked"], 3)
            self.assertEqual(stats["updated"], 3)
            for name in ("archival_memory.jsonl", "event_summaries.jsonl", "session_memories.jsonl"):
                payload = json.loads((root / name).read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(payload["time_source"], "semantic_relative_date")
                self.assertEqual(payload["semantic_day"], "2026-06-12")

    def test_manager_skips_feedback_and_excluded_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _append_jsonl(
                root / "archival_memory.jsonl",
                {
                    "id": "mem_bug",
                    "type": "system_feedback",
                    "summary": "用户指出助手刚才没有输出，有 bug。",
                    "created_at": "2099-01-01T00:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+08:00",
                    "source_message_ids": ["msg_bug"],
                    "exclude_from_retrieval": True,
                    "status": "active",
                },
            )
            manager = MemoryManager(memory_root=root, top_k=4, enable_semantic=False)

            context = manager.prepare_context("刚才没有输出 bug", session_id="sess_probe")

            self.assertEqual(context.hits, [])

    def test_manager_skips_stale_semantic_hits_for_excluded_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _append_jsonl(
                root / "archival_memory.jsonl",
                {
                    "id": "mem_bug",
                    "type": "system_feedback",
                    "summary": "用户指出助手刚才没有输出，有 bug。",
                    "created_at": "2099-01-01T00:00:00+08:00",
                    "last_recalled_at": "2099-01-01T00:00:00+08:00",
                    "source_message_ids": ["msg_bug"],
                    "exclude_from_retrieval": True,
                    "status": "active",
                },
            )
            manager = MemoryManager(memory_root=root, top_k=4, enable_semantic=False)
            stale_hit = RetrievalHit(
                chunk_id="chunk_mem_bug",
                text="用户指出助手刚才没有输出，有 bug。",
                score=0.9,
                source_type="memory",
                source_id="mem_bug",
                source_session_id="sess_old",
                source_message_id="msg_bug",
                created_at="2099-01-01T00:00:00+08:00",
            )

            hits = manager._merge_memory_hits(
                [stale_hit],
                manager._load_archival_records(),
                current_turn_ids=set(),
                time_hints={},
                session_id="sess_probe",
            )

            self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
