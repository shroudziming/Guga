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
    def test_time_validator_accepts_same_day_and_unknown_end_contracts(self) -> None:
        module = _load_script_module()
        module.validate_time_event(
            {
                "start_at": "2026-07-14T15:00:00+08:00",
                "end_at": "2026-07-14T15:00:00+08:00",
                "end_unknown": False,
            },
            expect_unknown_end=False,
        )
        module.validate_time_event(
            {
                "start_at": "2026-07-13T00:00:00+08:00",
                "end_at": None,
                "end_unknown": True,
            },
            expect_unknown_end=True,
        )


if __name__ == "__main__":
    unittest.main()
