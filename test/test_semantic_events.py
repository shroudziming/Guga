from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.memory.semantic_events import SemanticEventStore
from guga.memory.time_utils import resolve_event_time


class SemanticEventTimeTest(unittest.TestCase):
    def test_resolves_relative_dates_ranges_and_unknown_time(self) -> None:
        reference = "2026-07-09T09:30:00+08:00"

        sunday = resolve_event_time("周日", reference, end_unknown=False)
        self.assertEqual(sunday.start_at, "2026-07-12T00:00:00+08:00")
        self.assertEqual(sunday.end_at, "2026-07-12T23:59:59+08:00")
        self.assertEqual(sunday.time_source, "relative_weekday")
        self.assertEqual(sunday.time_granularity, "date")

        next_tuesday = resolve_event_time("下周二", reference, end_unknown=False)
        self.assertEqual(next_tuesday.start_at, "2026-07-14T00:00:00+08:00")

        date_range = resolve_event_time("2026-07-14 到 2026-07-16", reference, end_unknown=False)
        self.assertEqual(date_range.start_at, "2026-07-14T00:00:00+08:00")
        self.assertEqual(date_range.end_at, "2026-07-16T23:59:59+08:00")
        self.assertEqual(date_range.time_source, "explicit_date")

        unknown = resolve_event_time("过几天再说", reference, end_unknown=True)
        self.assertIsNone(unknown.start_at)
        self.assertIsNone(unknown.end_at)
        self.assertEqual(unknown.time_source, "unknown")
        self.assertEqual(unknown.time_granularity, "unknown")
        self.assertTrue(unknown.end_unknown)


class SemanticEventStoreTest(unittest.TestCase):
    def test_store_ignores_llm_absolute_time_and_keeps_only_event_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            outcome = store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "appointment",
                        "subject": "user",
                        "entity": "dental appointment",
                        "description": "用户将在周日看牙",
                        "time_expression": "周日",
                        "end_unknown": False,
                        "reference_created_at": "2026-07-09T09:30:00+08:00",
                        "source_message_ids": ["msg_1"],
                        "start_at": "1999-01-01T00:00:00+08:00",
                        "end_at": "1999-01-01T23:59:59+08:00",
                        "time_source": "llm_claim",
                        "confidence": 0.9,
                        "guga_reflection": {
                            "appraisal": "我会在意这件事。",
                            "felt_response": "有点挂心。",
                            "relational_intent": "之后自然关心。",
                            "interpretation_confidence": 0.7,
                        },
                    }
                ],
                session_id="sess_1",
                include_guga_reflection=False,
            )

            self.assertEqual(len(outcome.created_event_ids), 1)
            event = store.load_active()[0]
            self.assertEqual(event["start_at"], "2026-07-12T00:00:00+08:00")
            self.assertEqual(event["time_source"], "relative_weekday")
            self.assertNotIn("guga_reflection", event)
            self.assertNotIn("summary", event)
            self.assertNotIn("object", event)
            self.assertNotIn("raw_excerpt", event)
            self.assertEqual(event["source_session_id"], "sess_1")

            persisted = json.loads((Path(tmp_dir) / "semantic_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(persisted["id"], event["id"])

    def test_replace_and_cancel_preserve_lifecycle_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            first = store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "state_change",
                        "subject": "user",
                        "entity": "mortgage preapproval",
                        "description": "用户的房贷预批额度为 $350,000",
                        "time_expression": "",
                        "end_unknown": True,
                        "reference_created_at": "2023-08-11T00:01:00+08:00",
                        "source_message_ids": ["msg_old"],
                    }
                ],
                session_id="s_old",
                include_guga_reflection=True,
            )
            old_id = first.created_event_ids[0]

            replacement = store.apply_operations(
                operations=[
                    {
                        "operation": "replace",
                        "target_event_id": old_id,
                        "event_kind": "state_change",
                        "subject": "user",
                        "entity": "mortgage preapproval",
                        "description": "用户的房贷预批额度为 $400,000",
                        "time_expression": "",
                        "end_unknown": True,
                        "reference_created_at": "2023-11-30T00:36:00+08:00",
                        "source_message_ids": ["msg_new"],
                        "guga_reflection": {
                            "appraisal": "这是一项重要的财务状态更新。",
                            "felt_response": "我会认真对待。",
                            "relational_intent": "之后以新额度为准。",
                            "interpretation_confidence": 0.8,
                        },
                    }
                ],
                session_id="s_new",
                include_guga_reflection=True,
            )
            new_id = replacement.created_event_ids[0]

            rows = {row["id"]: row for row in store.load_all()}
            self.assertEqual(rows[old_id]["status"], "inactive")
            self.assertEqual(rows[old_id]["inactive_reason"], "replaced")
            self.assertEqual(rows[new_id]["replaces_event_id"], old_id)
            self.assertEqual(store.load_active()[0]["description"], "用户的房贷预批额度为 $400,000")
            self.assertIn("guga_reflection", rows[new_id])

            cancellation = store.apply_operations(
                operations=[
                    {
                        "operation": "cancel",
                        "target_event_id": new_id,
                        "source_message_ids": ["msg_cancel"],
                    }
                ],
                session_id="s_cancel",
                include_guga_reflection=True,
            )
            self.assertEqual(cancellation.deactivated_event_ids, [new_id])
            cancelled = {row["id"]: row for row in store.load_all()}[new_id]
            self.assertEqual(cancelled["status"], "inactive")
            self.assertEqual(cancelled["inactive_reason"], "cancelled")
            self.assertIn("msg_cancel", cancelled["source_message_ids"])
            self.assertEqual(store.load_active(), [])


if __name__ == "__main__":
    unittest.main()
