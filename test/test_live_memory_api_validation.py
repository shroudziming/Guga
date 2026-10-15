from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "live_memory_api_validation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("live_memory_api_validation", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveMemoryApiValidationUnitTest(unittest.TestCase):
    def test_time_validator_accepts_exclusive_end_contract(self) -> None:
        module = _load_script_module()
        module.validate_time_event(
            {
                "start_at": "2026-08-20T00:00:00+08:00",
                "end_at": "2026-08-21T00:00:00+08:00",
                "end_unknown": False,
            }
        )

    def test_time_validator_rejects_equal_or_missing_end(self) -> None:
        module = _load_script_module()
        for end_at in ("2026-08-20T00:00:00+08:00", None):
            with self.subTest(end_at=end_at), self.assertRaises(AssertionError):
                module.validate_time_event(
                    {
                        "start_at": "2026-08-20T00:00:00+08:00",
                        "end_at": end_at,
                        "end_unknown": end_at is None,
                    }
                )


if __name__ == "__main__":
    unittest.main()
