from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock


class WorkspaceError(ValueError):
    pass


class WorkspaceContext:
    def __init__(self, default_root: Path, allow_create: bool = False) -> None:
        resolved = Path(default_root).resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"default workspace is not a directory: {resolved}")
        self._default_root = resolved
        self._current_root = resolved
        self._allow_create = bool(allow_create)
        self._confirmed = False
        self._lock = RLock()

    @property
    def default_root(self) -> Path:
        return self._default_root

    @property
    def current_root(self) -> Path:
        with self._lock:
            return self._current_root

    @property
    def confirmed(self) -> bool:
        with self._lock:
            return self._confirmed

    @property
    def allow_create(self) -> bool:
        return self._allow_create

    def inspect(self) -> dict[str, object]:
        with self._lock:
            self._require_directory(self._current_root)
            self._confirmed = True
            return self._state_payload()

    def set(self, path: str, create_if_missing: bool = False) -> dict[str, object]:
        with self._lock:
            if not self._confirmed:
                raise WorkspaceError("inspect the current workspace before changing it")
            raw_path = str(path).strip()
            if not raw_path:
                raise WorkspaceError("workspace path is required")
            candidate = Path(raw_path)
            target = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self._current_root / candidate).resolve()
            )
            created = False
            if not target.exists():
                if not (self._allow_create and create_if_missing):
                    raise WorkspaceError(
                        f"workspace does not exist and creation is disabled: {target}"
                    )
                target.mkdir(parents=True, exist_ok=False)
                created = True
            self._require_directory(target)

            previous = self._current_root
            self._current_root = target
            self._confirmed = False
            return {
                "previous_root": str(previous),
                "current_root": str(target),
                "default_root": str(self._default_root),
                "created": created,
                "confirmed": False,
            }

    def reset(self) -> dict[str, object]:
        with self._lock:
            self._require_directory(self._default_root)
            previous = self._current_root
            self._current_root = self._default_root
            self._confirmed = False
            return {
                "previous_root": str(previous),
                "current_root": str(self._default_root),
                "default_root": str(self._default_root),
                "created": False,
                "confirmed": False,
            }

    def require_confirmed(self) -> Path:
        with self._lock:
            if not self._confirmed:
                raise WorkspaceError(
                    "inspect the current workspace before using operational tools"
                )
            self._require_directory(self._current_root)
            return self._current_root

    def invalidate_confirmation(self) -> None:
        with self._lock:
            self._confirmed = False

    def _state_payload(self) -> dict[str, object]:
        return {
            "current_root": str(self._current_root),
            "default_root": str(self._default_root),
            "allow_create": self._allow_create,
            "confirmed": self._confirmed,
        }

    @staticmethod
    def _require_directory(path: Path) -> None:
        if not path.exists():
            raise WorkspaceError(f"workspace does not exist: {path}")
        if not path.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {path}")


def workspace_context_from_env(
    fallback_root: Path,
    env: Mapping[str, str] | None = None,
) -> WorkspaceContext:
    source = env if env is not None else os.environ
    configured_root = str(source.get("Guga_CLI_DEFAULT_WORKSPACE_PATH", "")).strip()
    default_root = Path(configured_root) if configured_root else Path(fallback_root)
    allow_create = str(source.get("Guga_CLI_ALLOW_CREATE_WORKSPACE", "0")).strip().lower()
    return WorkspaceContext(
        default_root,
        allow_create=allow_create in {"1", "true", "yes", "on"},
    )
