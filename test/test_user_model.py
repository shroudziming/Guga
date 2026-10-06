from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.user_model import GugaUserModelStore


class GugaUserModelStoreTest(unittest.TestCase):
    def test_user_model_allows_an_insight_without_evidence_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = GugaUserModelStore(Path(tmp_dir) / "guga_user_model.json")

            written = store.apply_operations(
                [
                    {
                        "operation": "upsert",
                        "statement": "用户希望记忆保留被修正的余地。",
                        "kind": "preference",
                        "confidence": 0.8,
                        "stability": "moderate",
                    }
                ]
            )

            self.assertEqual(written[0]["sources"], [])
            self.assertNotIn("source_event_ids", written[0])

    def test_user_model_persists_structured_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = GugaUserModelStore(Path(tmp_dir) / "guga_user_model.json")

            written = store.apply_operations(
                [
                    {
                        "operation": "upsert",
                        "statement": "用户希望记忆保留被修正的余地。",
                        "kind": "preference",
                        "confidence": 0.8,
                        "stability": "moderate",
                        "sources": [
                            {"type": "event_summary", "id": "summary_1"},
                            {"type": "message", "id": "msg_1"},
                        ],
                    }
                ]
            )

            self.assertEqual(
                written[0]["sources"],
                [
                    {"type": "event_summary", "id": "summary_1"},
                    {"type": "message", "id": "msg_1"},
                ],
            )

    def test_user_model_load_rejects_legacy_event_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "guga_user_model.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "insights": [
                            {
                                "id": "gum_legacy",
                                "statement": "旧画像",
                                "source_event_ids": ["evt_a"],
                                "status": "active",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source_event_ids is no longer supported"):
                GugaUserModelStore(path).load()

    def test_user_model_rejects_legacy_event_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = GugaUserModelStore(Path(tmp_dir) / "guga_user_model.json")

            with self.assertRaisesRegex(ValueError, "source_event_ids is no longer supported"):
                store.apply_operations(
                    [
                        {
                            "operation": "upsert",
                            "statement": "Guga 认为用户会认真处理重要安排。",
                            "kind": "reliable_pattern",
                            "confidence": 0.82,
                            "stability": "recurring",
                            "source_event_ids": ["evt_a", "evt_b"],
                        }
                    ]
                )


if __name__ == "__main__":
    unittest.main()
