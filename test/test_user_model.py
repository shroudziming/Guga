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

    def test_user_model_load_normalizes_legacy_event_sources(self) -> None:
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

            insight = GugaUserModelStore(path).load()["insights"][0]

            self.assertEqual(insight["sources"], [{"type": "semantic_event", "id": "evt_a"}])

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
            self.assertEqual(
                insight["sources"],
                [
                    {"type": "semantic_event", "id": "evt_a"},
                    {"type": "semantic_event", "id": "evt_b"},
                ],
            )
            self.assertNotIn("source_event_ids", insight)
            self.assertNotIn("source_message_ids", insight)
            self.assertEqual(insight["status"], "active")


if __name__ == "__main__":
    unittest.main()
