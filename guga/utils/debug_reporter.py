from __future__ import annotations

from pathlib import Path


class FileDebugSink:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._targets: dict[str, Path] = {}

    def __call__(self, message: str) -> None:
        session_id = self._extract_session_id(message)
        target = self._targets.get(session_id)
        if target is None:
            target = self.report_dir / f"{session_id}.log"
            self._targets[session_id] = target

        with target.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def _extract_session_id(self, message: str) -> str:
        parts = message.split("]")
        if len(parts) >= 3:
            value = parts[2].strip("[").strip()
            if value:
                return value
        return "unknown_session"
