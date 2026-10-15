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
    def test_store_rejects_create_without_complete_non_empty_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            base = {
                "operation": "create",
                "event_kind": "appointment",
                "subject": "user",
                "entity": "面试",
                "description": "用户参加面试",
                "time_expression": "8月20号",
                "time_granularity": "day",
            }
            for start_at, end_at, error in (
                ("2026-08-20T00:00:00+08:00", None, "end_at is required"),
                (
                    "2026-08-20T00:00:00+08:00",
                    "2026-08-20T00:00:00+08:00",
                    "start_at must be earlier than end_at",
                ),
                ("2026-08-20T00:00:00", "2026-08-21T00:00:00+08:00", "start_at must include timezone"),
            ):
                with self.subTest(start_at=start_at, end_at=end_at):
                    with self.assertRaisesRegex(ValueError, error):
                        store.apply_operations(
                            operations=[{**base, "start_at": start_at, "end_at": end_at}],
                            session_id="sess_strict",
                            include_guga_reflection=False,
                        )

    def test_store_persists_end_exclusive_interval_and_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            outcome = store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "appointment",
                        "subject": "user",
                        "entity": "面试",
                        "description": "用户参加面试",
                        "time_expression": "8月20号",
                        "start_at": "2026-08-20T00:00:00+08:00",
                        "end_at": "2026-08-21T00:00:00+08:00",
                        "time_granularity": "day",
                        "confidence": 0.9,
                    }
                ],
                session_id="sess_strict",
                include_guga_reflection=False,
            )

            self.assertEqual(len(outcome.created_event_ids), 1)
            event = store.load_active()[0]
            self.assertEqual(event["end_at"], "2026-08-21T00:00:00+08:00")
            self.assertEqual(event["time_granularity"], "day")
            self.assertFalse(event["end_unknown"])
            self.assertEqual(event["time_source"], "llm_resolved")

    def test_store_rejects_non_user_semantic_event_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            with self.assertRaisesRegex(ValueError, "subject must be 'user'"):
                store.apply_operations(
                    operations=[
                        {
                            "operation": "create",
                            "event_kind": "appointment",
                            "subject": "Guga",
                            "entity": "dentist",
                            "description": "Guga sees a dentist.",
                            "start_at": "2026-07-14T15:00:00+08:00",
                            "end_at": "2026-07-14T16:00:00+08:00",
                            "time_granularity": "hour",
                        }
                    ],
                    session_id="sess_non_user",
                    include_guga_reflection=False,
                )

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
                        "start_at": "2023-08-11T00:01:00+08:00",
                        "end_at": "2023-08-11T00:02:00+08:00",
                        "time_granularity": "minute",
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
                        "start_at": "2023-11-30T00:36:00+08:00",
                        "end_at": "2023-11-30T00:37:00+08:00",
                        "time_granularity": "minute",
                        "reference_created_at": "2023-11-30T00:36:00+08:00",
                        "source_message_ids": ["msg_new"],
                        "guga_reflection": {
                            "appraisal": "这是一项重要的财务状态更新。",
                            "felt_response": "我会认真对待。",
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

    def test_new_write_persists_exactly_two_reflection_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            store.apply_operations(
                operations=[
                    {
                        "operation": "create",
                        "event_kind": "task",
                        "subject": "user",
                        "entity": "report",
                        "description": "用户计划提交报告。",
                        "start_at": "2026-07-03T00:00:00+08:00",
                        "end_at": "2026-07-04T00:00:00+08:00",
                        "time_granularity": "day",
                        "source_message_ids": ["msg_user"],
                        "guga_reflection": {
                            "appraisal": "这件事很重要。",
                            "felt_response": "我会挂心。",
                        },
                    }
                ],
                session_id="sess_reflection",
                include_guga_reflection=True,
            )

            reflection = store.load_all()[0]["guga_reflection"]
            self.assertEqual(set(reflection), {"appraisal", "felt_response"})
            self.assertTrue(all(isinstance(value, str) and value for value in reflection.values()))

    def test_load_all_preserves_legacy_four_field_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "semantic_events.jsonl"
            legacy = {
                "id": "evt_legacy",
                "type": "semantic_event",
                "status": "active",
                "guga_reflection": {
                    "appraisal": "重要",
                    "felt_response": "在意",
                    "relational_intent": "继续关心",
                    "interpretation_confidence": 0.8,
                },
            }
            path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")

            self.assertEqual(SemanticEventStore(path).load_all(), [legacy])


if __name__ == "__main__":
    unittest.main()
