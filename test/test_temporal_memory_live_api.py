from __future__ import annotations

import os
import unittest

from scripts.live_memory_api_validation import run_temporal_e2e


@unittest.skipUnless(os.environ.get("GUGA_RUN_LIVE_API_TESTS") == "1", "live API test disabled")
class TemporalMemoryLiveApiTest(unittest.TestCase):
    def test_real_api_consolidates_on_exit_and_retrieves_after_restart(self) -> None:
        result = run_temporal_e2e()
        self.assertEqual(result["event_start_at"], "2026-08-20T00:00:00+08:00")
        self.assertEqual(result["event_end_at"], "2026-08-21T00:00:00+08:00")
        self.assertIn(result["event_id"], result["retrieved_memory_ids"])
        self.assertTrue(result["first_turn_has_temporal_context"])
        self.assertTrue(result["second_turn_has_temporal_context"])
        self.assertTrue(result["debug_has_time_search"])
        self.assertTrue(result["debug_has_exit_consolidation"])


if __name__ == "__main__":
    unittest.main()
