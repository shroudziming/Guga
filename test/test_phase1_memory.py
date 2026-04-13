from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.chat import ChatSession
from guga.memory.archival_store import ArchivalStore
from guga.memory.manager import MemoryManager
from guga.memory.schema import ArchivalMemoryRecord, MessageRecord
from guga.memory.session_store import SessionStore
from guga.memory.storage import read_json, read_jsonl
from guga.types import GenerationConfig


class FakeChatModel:
    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
        _ = gen
        last_user = ""
        for message in messages:
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
        return f"收到：{last_user}"


class TestPhase1Memory(unittest.TestCase):
    def test_session_store_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(Path(tmp_dir))
            session_id = "sess_store_case"
            store.append(
                MessageRecord(
                    id="msg_1",
                    session_id=session_id,
                    role="user",
                    content="hello",
                    created_at="2026-04-13T10:00:00+08:00",
                )
            )
            records = store.read_session(session_id)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].content, "hello")

    def test_archival_store_mark_accessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "archival_memory.jsonl"
            store = ArchivalStore(file_path)
            record = ArchivalMemoryRecord(
                id="mem_1",
                type="episodic",
                topic="career",
                summary="用户提到换工作",
                raw_excerpt="我最近在考虑换工作",
                created_at="2026-04-13T10:00:00+08:00",
                event_time_start="2026-04-13T10:00:00+08:00",
            )
            store.append(record)

            changed = store.mark_accessed("mem_1")
            self.assertTrue(changed)

            rows = read_jsonl(file_path)
            self.assertEqual(rows[0]["access_count"], 1)
            self.assertTrue(bool(rows[0]["last_accessed_at"]))

    def test_chat_session_memory_persistence_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_manager = MemoryManager(memory_root=Path(tmp_dir))
            session = ChatSession(
                model=FakeChatModel(),
                system_prompt="你是测试助手",
                generation=GenerationConfig(),
                memory_manager=memory_manager,
                session_id="sess_phase1_case",
            )

            session.reply("我最近在考虑换工作而且有点焦虑")

            session_files = list((Path(tmp_dir) / "sessions").glob("**/*.jsonl"))
            self.assertEqual(len(session_files), 1)
            rows = read_jsonl(session_files[0])
            self.assertEqual(len(rows), 2)

            profile_payload = read_json(Path(tmp_dir) / "profile.json", default={})
            self.assertTrue("updated_at" in profile_payload)

            archival_rows = read_jsonl(Path(tmp_dir) / "archival_memory.jsonl")
            self.assertGreaterEqual(len(archival_rows), 1)
            self.assertIn("换工作", archival_rows[0]["raw_excerpt"])


if __name__ == "__main__":
    unittest.main()
