from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.user_model import GugaUserModelStore


class GugaUserModelStoreTest(unittest.TestCase):
    def test_user_model_keeps_event_provenance_without_raw_message_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = GugaUserModelStore(Path(tmp_dir) / "guga_user_model.json")
            written = store.apply_operations(
                [
                    {
                        "operation": "upsert",
                        "statement": "Guga 认为用户会认真处理重要安排。",
                        "kind": "reliable_pattern",
                        "confidence": 0.82,
                        "stability": "recurring",
                        "source_event_ids": ["evt_a", "evt_b"],
                        "source_message_ids": ["msg_should_not_persist"],
                    }
                ]
            )

            self.assertEqual(len(written), 1)
            payload = json.loads((Path(tmp_dir) / "guga_user_model.json").read_text(encoding="utf-8"))
            insight = payload["insights"][0]
            self.assertEqual(insight["source_event_ids"], ["evt_a", "evt_b"])
            self.assertNotIn("source_message_ids", insight)
            self.assertEqual(insight["status"], "active")


if __name__ == "__main__":
    unittest.main()
