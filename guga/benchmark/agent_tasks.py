from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable


CaseSetup = Callable[[Path], None]
CaseVerifier = Callable[[Path, list[dict]], tuple[bool, str]]


@dataclass(frozen=True)
class AgentTaskCase:
    case_id: str
    task: str
    setup: CaseSetup
    verify: CaseVerifier


def default_agent_task_cases() -> list[AgentTaskCase]:
    return [
        AgentTaskCase(
            "read_file_value",
            "读取 config.txt，找出 answer 的值并验证读取结果。",
            lambda root: (root / "config.txt").write_text("answer=42\n", encoding="utf-8"),
            _verify_read_value,
        ),
        AgentTaskCase(
            "discover_unknown_file",
            "列出 inbox 目录，找到其中未知名称的文本文件，读取并报告 discovered 的值。",
            _setup_discovery,
            _verify_discovery,
        ),
        AgentTaskCase(
            "run_python_command",
            "运行确定性 Python 命令计算 6*7，并验证标准输出为 42。",
            _no_setup,
            _verify_python,
        ),
        AgentTaskCase(
            "read_execute_verify",
            "读取 source.txt 的 value，用 Python 计算它的平方，并验证结果为 49。",
            lambda root: (root / "source.txt").write_text("value=7\n", encoding="utf-8"),
            _verify_multistep,
        ),
        AgentTaskCase(
            "recover_command_failure",
            "先运行一个确定会以非零状态退出的 Python 命令；识别预期失败后，改用正确命令输出 recovered 并验证。",
            _no_setup,
            _verify_recovery,
        ),
        AgentTaskCase(
            "write_and_reread",
            "将 verified-write 写入 output.txt，再重新读取文件验证内容完全一致。",
            _no_setup,
            _verify_write,
        ),
    ]


def run_agent_task_benchmark(
    runner_factory,
    output_root: Path,
    *,
    run_id: str | None = None,
    cases: list[AgentTaskCase] | None = None,
) -> Path:
    """Run real task loops; only external deterministic verifiers assign pass/fail."""

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root).resolve() / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_cases = cases or default_agent_task_cases()
    results: list[dict] = []
    previous_flags = {
        "Guga_ENABLE_COMMAND_TOOL": os.environ.get("Guga_ENABLE_COMMAND_TOOL"),
        "Guga_ENABLE_WRITE_TOOL": os.environ.get("Guga_ENABLE_WRITE_TOOL"),
    }
    os.environ["Guga_ENABLE_COMMAND_TOOL"] = "1"
    os.environ["Guga_ENABLE_WRITE_TOOL"] = "1"
    try:
        for case in selected_cases:
            results.append(_run_case(case, runner_factory, run_dir))
            _write_jsonl(run_dir / "results.jsonl", results)
    finally:
        _restore_env(previous_flags)

    passed = sum(bool(row["passed"]) for row in results)
    metrics = {
        "run_id": resolved_run_id,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "tool_calls": sum(int(row["tool_calls"]) for row in results),
        "retries": sum(int(row["retries"]) for row in results),
        "plan_revisions": sum(int(row["plan_revisions"]) for row in results),
        "duration_ms": sum(int(row["duration_ms"]) for row in results),
        "failure_reasons": [row["failure_reason"] for row in results if row["failure_reason"]],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _run_case(case: AgentTaskCase, runner_factory, run_dir: Path) -> dict:
    started = perf_counter()
    case_dir = run_dir / "cases" / case.case_id
    workspace = case_dir / "workspace"
    runtime_root = case_dir / "runtime"
    workspace.mkdir(parents=True, exist_ok=True)
    case.setup(workspace)
    task_id = f"bench-{case.case_id}"
    trace_rows: list[dict] = []
    state: dict = {"status": "failed"}
    error = ""
    runner = None
    try:
        runner = runner_factory(case, workspace, runtime_root)
        list(runner.start(case.task, f"benchmark-{case.case_id}", task_id=task_id))
        for _ in range(10):
            state = runner.get_task(task_id)
            if state.get("status") in {"completed", "failed", "blocked"}:
                break
            if state.get("status") != "awaiting_approval":
                raise RuntimeError(f"benchmark graph stopped in unexpected state: {state.get('status')}")
            list(runner.resume(task_id, approved=True))
        else:
            raise RuntimeError("benchmark exceeded approval/revision limit")
        trace_rows = runner.trace.load(task_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if runner is not None:
            try:
                trace_rows = runner.trace.load(task_id)
            except Exception:
                pass
    finally:
        if runner is not None:
            runner.close()

    if not trace_rows:
        trace_rows = [{"event": "benchmark_error", "error": error or "empty trace"}]
    _write_jsonl(case_dir / "trace.jsonl", trace_rows)
    verifier_passed, verifier_reason = case.verify(workspace, trace_rows)
    runtime_status = str(state.get("status", "failed"))
    passed = runtime_status == "completed" and verifier_passed and not error
    failure_reason = ""
    if error:
        failure_reason = error
    elif runtime_status != "completed":
        failure_reason = f"runtime status: {runtime_status}"
    elif not verifier_passed:
        failure_reason = verifier_reason

    tool_starts = [row for row in trace_rows if row.get("event") == "tool_call_started"]
    return {
        "case_id": case.case_id,
        "task": case.task,
        "passed": passed,
        "runtime_status": runtime_status,
        "verifier_reason": verifier_reason,
        "failure_reason": failure_reason,
        "tool_calls": len(tool_starts),
        "retries": sum(int(row.get("attempt", 1)) > 1 for row in tool_starts),
        "plan_revisions": sum(
            row.get("event") == "plan_revision_requested" for row in trace_rows
        ),
        "duration_ms": int((perf_counter() - started) * 1000),
        "trace": f"cases/{case.case_id}/trace.jsonl",
    }


def _setup_discovery(root: Path) -> None:
    inbox = root / "inbox"
    inbox.mkdir()
    (inbox / "hidden-value.txt").write_text("discovered=blue\n", encoding="utf-8")


def _no_setup(root: Path) -> None:
    return None


def _verify_read_value(root: Path, rows: list[dict]) -> tuple[bool, str]:
    passed = any(
        "answer=42" in str(result.get("content", ""))
        for result in _tool_results(rows, "guga_read_file")
    )
    return passed, "read_file did not return answer=42"


def _verify_discovery(root: Path, rows: list[dict]) -> tuple[bool, str]:
    listed = bool(_tool_results(rows, "guga_list_dir"))
    read = any(
        "discovered=blue" in str(result.get("content", ""))
        for result in _tool_results(rows, "guga_read_file")
    )
    return listed and read, "agent did not list the directory and read the discovered value"


def _verify_python(root: Path, rows: list[dict]) -> tuple[bool, str]:
    passed = any(
        result.get("returncode") == 0 and str(result.get("stdout", "")).strip() == "42"
        for result in _tool_results(rows, "guga_run_command")
    )
    return passed, "Python command did not produce deterministic output 42"


def _verify_multistep(root: Path, rows: list[dict]) -> tuple[bool, str]:
    read = any(
        "value=7" in str(result.get("content", ""))
        for result in _tool_results(rows, "guga_read_file")
    )
    executed = any(
        result.get("returncode") == 0 and str(result.get("stdout", "")).strip() == "49"
        for result in _tool_results(rows, "guga_run_command")
    )
    verified = any(
        row.get("event") == "verification_finished"
        and row.get("verification", {}).get("matched") is True
        for row in rows
    )
    return read and executed and verified, "read, execute, and verification evidence was incomplete"


def _verify_recovery(root: Path, rows: list[dict]) -> tuple[bool, str]:
    results = _tool_results(rows, "guga_run_command")
    failed = any(result.get("returncode") not in {None, 0} for result in results)
    recovered = any(
        result.get("returncode") == 0 and "recovered" in str(result.get("stdout", ""))
        for result in results
    )
    return failed and recovered, "expected command failure was not followed by a successful recovery"


def _verify_write(root: Path, rows: list[dict]) -> tuple[bool, str]:
    target = root / "output.txt"
    disk_matches = target.exists() and target.read_text(encoding="utf-8") == "verified-write"
    wrote = bool(_tool_results(rows, "guga_write_file"))
    reread = any(
        result.get("content") == "verified-write"
        for result in _tool_results(rows, "guga_read_file")
    )
    return disk_matches and wrote and reread, "written file or reread evidence did not match"


def _tool_results(rows: list[dict], tool_name: str) -> list[dict]:
    execution_ids = {
        row.get("execution_id")
        for row in rows
        if row.get("event") == "tool_call_started" and row.get("tool_name") == tool_name
    }
    return [
        row.get("result", {})
        for row in rows
        if row.get("event") == "tool_call_finished"
        and row.get("execution_id") in execution_ids
        and isinstance(row.get("result"), dict)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
