from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from guga.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, PROJECT_ROOT
from guga.memory.summarizer import MemoryBankSummarizer
from guga.models import create_chat_model
from guga.persona import PersonaManager
from guga.utils.paths import personas_dir


REQUIRED_OPERATION_FIELDS = {
    "operation",
    "event_kind",
    "subject",
    "entity",
    "description",
    "time_expression",
    "start_at",
    "end_at",
    "end_unknown",
    "source_message_ids",
    "confidence",
    "guga_reflection",
}


@unittest.skipUnless(os.environ.get("GUGA_RUN_LIVE_API_TESTS") == "1", "live API test disabled")
class PersonaSkillLiveApiTest(unittest.TestCase):
    def test_low_level_fields_are_covered_by_real_api(self) -> None:
        self._load_project_env_without_overwriting_process_values()
        self.assertEqual(os.environ.get("Guga_MODEL_PROVIDER", "").lower(), "api")
        model = create_chat_model(
            model_id=os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID),
            cache_dir=os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        )
        persona = PersonaManager(personas_dir()).load("default")
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
        result = summarizer.consolidate_low_level_memory(
            {
                "new_turns": [
                    {
                        "user_message_id": "msg_live_user",
                        "assistant_message_id": "msg_live_assistant",
                        "user_text": "我计划在2026年7月20日上午10点去医院复查。",
                        "assistant_text": "[sided_worried]记住了，这件事要认真对待。",
                        "created_at": "2026-07-15T12:00:00+08:00",
                    }
                ],
                "recent_active_events": [],
                "relevant_active_events": [],
                "retrieved_context": [],
            },
            include_guga_reflection=True,
            reflection_context=persona.reflection_context,
        )

        operation = next(
            (
                item
                for item in result.get("semantic_event_operations", [])
                if isinstance(item, dict) and item.get("operation") == "create"
            ),
            {},
        )
        coverage = self._field_coverage(operation)
        try:
            self.assertTrue(coverage["has_create_operation"], "missing create operation")
            for field in sorted(REQUIRED_OPERATION_FIELDS):
                self.assertTrue(coverage[f"has_{field}"], f"missing required field: {field}")
            self.assertTrue(coverage["subject_is_user"], "subject must be user")
            self.assertTrue(coverage["has_live_user_source"], "source message id must cover the user turn")
            self.assertTrue(coverage["reflection_has_exact_fields"], "reflection must have exactly two fields")
            self.assertTrue(coverage["reflection_fields_non_empty"], "reflection fields must be non-empty")
            self.assertTrue(
                coverage["reflection_absent_from_description"],
                "subjective reflection must not be copied into description",
            )
        except AssertionError as error:
            artifact_path = self._write_failure_artifact(result, coverage)
            error.add_note(f"failure artifact: {artifact_path}")
            raise

    @staticmethod
    def _load_project_env_without_overwriting_process_values() -> None:
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    @staticmethod
    def _field_coverage(operation: dict) -> dict[str, bool]:
        reflection = operation.get("guga_reflection")
        reflection_is_dict = isinstance(reflection, dict)
        reflection_values = list(reflection.values()) if reflection_is_dict else []
        description = operation.get("description")
        description_is_string = isinstance(description, str)
        source_message_ids = operation.get("source_message_ids")

        coverage = {f"has_{field}": field in operation for field in REQUIRED_OPERATION_FIELDS}
        coverage.update(
            {
                "has_create_operation": operation.get("operation") == "create",
                "subject_is_user": operation.get("subject") == "user",
                "has_live_user_source": isinstance(source_message_ids, list)
                and "msg_live_user" in source_message_ids,
                "reflection_has_exact_fields": reflection_is_dict
                and set(reflection) == {"appraisal", "felt_response"},
                "reflection_fields_non_empty": len(reflection_values) == 2
                and all(isinstance(value, str) and value.strip() for value in reflection_values),
                "reflection_absent_from_description": description_is_string
                and all(value not in description for value in reflection_values if isinstance(value, str)),
            }
        )
        return coverage

    @staticmethod
    def _write_failure_artifact(result: dict, coverage: dict[str, bool]) -> Path:
        artifact_dir = Path(tempfile.mkdtemp(prefix="guga_persona_skill_live_api_"))
        artifact_path = artifact_dir / "failure.json"
        artifact_path.write_text(
            json.dumps(
                {"structured_result": result, "field_coverage": coverage},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return artifact_path


if __name__ == "__main__":
    unittest.main()
