from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.agent.runner import TaskRunEvent
from guga.benchmark.agent_tasks import default_agent_task_cases, run_agent_task_benchmark


class FakeTrace:
    def __init__(self) -> None:
        self.rows = []

    def load(self, task_id: str) -> list[dict]:
        return list(self.rows)


class FakeBenchmarkRunner:
    def __init__(self, case, workspace: Path) -> None:
        self.case = case
        self.workspace = workspace
        self.trace = FakeTrace()
        self.state = {"status": "planning", "plan_revision": 1, "attempt": 0}

    def start(self, request: str, session_id: str, *, task_id: str):
        self.state.update({"task_id": task_id, "status": "awaiting_approval"})
        yield TaskRunEvent("awaiting_approval", task_id, payload={"revision": 1, "plan": []})

    def resume(self, task_id: str, *, approved: bool):
        if not approved:
            raise AssertionError("benchmark should auto-approve")
        self._execute_case()
        self.state.update(
            {
                "status": "completed",
                "final_response": "done",
                "trace_ref": f"agent-run://benchmark/{task_id}/trace.jsonl",
            }
        )
        yield TaskRunEvent("terminal", task_id, "done", {"status": "completed"})

    def get_task(self, task_id: str) -> dict:
        return dict(self.state)

    def close(self) -> None:
        pass

    def _execute_case(self) -> None:
        case_id = self.case.case_id
        if case_id == "read_file_value":
            self._tool("guga_read_file", {"content": "answer=42", "ok": True})
        elif case_id == "discover_unknown_file":
            self._tool("guga_list_dir", {"entries": [{"name": "hidden-value.txt"}], "ok": True})
            self._tool("guga_read_file", {"content": "discovered=blue", "ok": True})
        elif case_id == "run_python_command":
            self._tool("guga_run_command", {"returncode": 0, "stdout": "42\n", "ok": True})
        elif case_id == "read_execute_verify":
            self._tool("guga_read_file", {"content": "value=7", "ok": True})
            self._tool("guga_run_command", {"returncode": 0, "stdout": "49\n", "ok": True})
            self.trace.rows.append(
                {"event": "verification_finished", "verification": {"matched": True}}
            )
        elif case_id == "recover_command_failure":
            self._tool("guga_run_command", {"returncode": 1, "stderr": "expected", "ok": True})
            self._tool("guga_run_command", {"returncode": 0, "stdout": "recovered\n", "ok": True})
        elif case_id == "write_and_reread":
            target = self.workspace / "output.txt"
            target.write_text("verified-write", encoding="utf-8")
            self._tool("guga_write_file", {"path": str(target), "ok": True})
            self._tool("guga_read_file", {"content": "verified-write", "ok": True})

    def _tool(self, name: str, result: dict) -> None:
        attempt = 1 + sum(row.get("tool_name") == name for row in self.trace.rows)
        execution_id = f"exec-{len(self.trace.rows)}"
        self.trace.rows.extend(
            [
                {
                    "event": "tool_call_started",
                    "tool_name": name,
                    "attempt": attempt,
                    "execution_id": execution_id,
                },
                {
                    "event": "tool_call_finished",
                    "result": result,
                    "execution_id": execution_id,
                },
            ]
        )


class AgentTaskBenchmarkTest(unittest.TestCase):
    def test_six_cases_write_deterministic_results_metrics_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)

            def factory(case, workspace, runs_root):
                return FakeBenchmarkRunner(case, workspace)

            run_dir = run_agent_task_benchmark(
                factory,
                output_root,
                run_id="test-run",
            )

            results = [
                json.loads(line)
                for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

            self.assertEqual(6, len(default_agent_task_cases()))
            self.assertEqual(6, len(results))
            self.assertTrue(all(row["passed"] for row in results))
            self.assertEqual(1.0, metrics["pass_rate"])
            self.assertEqual(6, metrics["passed"])
            for case in default_agent_task_cases():
                trace_path = run_dir / "cases" / case.case_id / "trace.jsonl"
                self.assertTrue(trace_path.exists(), case.case_id)
                self.assertTrue(trace_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
