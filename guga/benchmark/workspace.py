from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from guga.config import PROJECT_ROOT


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class BenchmarkWorkspace:
    name: str
    run_id: str
    root: Path

    @property
    def memory_root(self) -> Path:
        return self.root / "memory"

    @property
    def debug_reports_dir(self) -> Path:
        return self.root / "debug_reports"

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def results_file(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def progress_file(self) -> Path:
        return self.root / "progress.jsonl"

    def case_root(self, case_id: str) -> Path:
        return self.root / "cases" / safe_case_id(case_id)

    def case_memory_root(self, case_id: str) -> Path:
        return self.case_root(case_id) / "memory"

    def case_debug_reports_dir(self, case_id: str) -> Path:
        return self.debug_reports_dir / safe_case_id(case_id)

    def case_checkpoint_file(self, case_id: str) -> Path:
        return self.case_root(case_id) / "checkpoint.json"


def benchmark_workspace(name: str, root: Path | None = None, run_id: str | None = None) -> BenchmarkWorkspace:
    safe_name = _safe_segment(name)
    safe_run_id = _safe_segment(run_id or _default_run_id())
    base = root or PROJECT_ROOT / "data" / "benchmarks"
    workspace = BenchmarkWorkspace(
        name=safe_name,
        run_id=safe_run_id,
        root=base / safe_name / "runs" / safe_run_id,
    )
    for path in (workspace.memory_root, workspace.debug_reports_dir, workspace.documents_dir, workspace.root):
        path.mkdir(parents=True, exist_ok=True)
    return workspace


def safe_case_id(value: str) -> str:
    return _safe_segment(value or "case")


def _safe_segment(value: str) -> str:
    normalized = _SAFE_NAME.sub("_", value.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("benchmark path segment must not be empty")
    return normalized


def _default_run_id() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y%m%d_%H%M%S")
