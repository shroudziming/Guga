from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.memory.clock import now_iso, parse_time
from guga.memory.manager import MemoryManager
from guga.memory.schema import MessageRecord, ProfileRecord
from guga.memory.storage import append_jsonl, read_json, read_jsonl, write_json
from guga.memory.profile_store import ProfileStore


class TestPhase0Memory(unittest.TestCase):
    def test_clock_iso_and_parse(self) -> None:
        timestamp = now_iso()
        parsed = parse_time(timestamp)
        self.assertIsNotNone(parsed.tzinfo)

    def test_storage_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            json_file = root / "profile.json"
            jsonl_file = root / "messages.jsonl"

            payload = {"name": "Guga", "version": 1}
            write_json(json_file, payload)
            loaded = read_json(json_file, default={})
            self.assertEqual(loaded["name"], "Guga")

            append_jsonl(jsonl_file, {"id": "m1", "content": "hello"})
            append_jsonl(jsonl_file, {"id": "m2", "content": "world"})
            rows = read_jsonl(jsonl_file)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["id"], "m2")

    def test_profile_store_typed_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ProfileStore(Path(tmp_dir) / "profile.json")
            profile = store.load()
            self.assertIsInstance(profile, ProfileRecord)

            profile.preferred_name = "小咕嘎"
            store.save(profile)

            loaded = store.load()
            self.assertEqual(loaded.preferred_name, "小咕嘎")
            self.assertTrue(bool(loaded.updated_at))

    def test_memory_manager_record_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = MemoryManager(memory_root=Path(tmp_dir))
            session_id = "sess_phase0_case"

            user_message_id = manager.record_user_message(session_id=session_id, text="你好")
            assistant_message_id = manager.record_assistant_message(session_id=session_id, text="你好呀")
            manager.finalize_turn(session_id)

            self.assertTrue(user_message_id.startswith("msg_"))
            self.assertTrue(assistant_message_id.startswith("msg_"))

            session_files = list((Path(tmp_dir) / "sessions").glob("**/*.jsonl"))
            self.assertEqual(len(session_files), 1)

            rows = read_jsonl(session_files[0])
            self.assertEqual(len(rows), 2)

            user_row = MessageRecord.from_dict(rows[0])
            assistant_row = MessageRecord.from_dict(rows[1])
            self.assertEqual(user_row.role, "user")
            self.assertEqual(assistant_row.role, "assistant")


if __name__ == "__main__":
    unittest.main()
