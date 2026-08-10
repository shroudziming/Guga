from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from guga.memory.time_utils import extract_semantic_time, now_beijing, parse_datetime
from guga.workspace import WorkspaceContext


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolModelResponse:
    content: str
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolStreamText:
    content: str


@dataclass(frozen=True)
class ToolStreamToolCalls:
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(
        self,
        tools: list[ToolSpec] | None = None,
        workspace: WorkspaceContext | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools or []}
        self._workspace = workspace

    @property
    def workspace(self) -> WorkspaceContext | None:
        return self._workspace

    def invalidate_workspace_confirmation(self) -> None:
        if self._workspace is not None:
            self._workspace.invalidate_confirmation()

    def add(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def has_tools(self) -> bool:
        return bool(self._tools)

    def names(self) -> set[str]:
        return set(self._tools)

    def openai_tools(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        selected = self._tools.values()
        if names is not None:
            selected = [tool for tool in selected if tool.name in names]
        return [tool.to_openai_tool() for tool in selected]

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        schema = tool.parameters
        required = schema.get("required", [])
        missing = [field for field in required if field not in arguments]
        if missing:
            raise ValueError(f"missing required tool argument: {missing[0]}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = [field for field in arguments if field not in properties]
            if unknown:
                raise ValueError(f"unknown tool argument: {unknown[0]}")
        for field, value in arguments.items():
            field_schema = properties.get(field)
            if not isinstance(field_schema, dict):
                continue
            expected = field_schema.get("type")
            if expected and not _matches_json_type(value, expected):
                raise ValueError(f"invalid type for tool argument {field}: expected {expected}")
            if "enum" in field_schema and value not in field_schema["enum"]:
                raise ValueError(f"invalid value for tool argument {field}")

    def execute(self, call: ToolCall) -> dict[str, Any]:
        tool = self._tools.get(call.name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {call.name}"}
        try:
            self.validate_arguments(call.name, call.arguments)
            result = tool.handler(call.arguments)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "tool": call.name}
        if not isinstance(result, dict):
            return {"ok": True, "result": result, "tool": call.name}
        result.setdefault("ok", True)
        result.setdefault("tool", call.name)
        return result


def default_tool_registry(
    project_root: Path | None = None,
    *,
    workspace: WorkspaceContext | None = None,
) -> ToolRegistry:
    active_workspace = workspace or WorkspaceContext(
        (project_root or Path(__file__).resolve().parents[1]).resolve()
    )
    registry = ToolRegistry(workspace=active_workspace)
    registry.add(_time_parse_tool())
    registry.add(_workspace_tool(active_workspace))
    registry.add(_list_dir_tool(active_workspace))
    registry.add(_read_file_tool(active_workspace))
    registry.add(_write_file_tool(active_workspace))
    registry.add(_run_command_tool(active_workspace))
    return registry


def conversation_tool_registry() -> ToolRegistry:
    return ToolRegistry([_time_parse_tool()])


def _workspace_tool(workspace: WorkspaceContext) -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action", ""))
        if action == "inspect":
            return workspace.inspect()
        if action == "set":
            return workspace.set(
                str(args.get("path", "")),
                create_if_missing=bool(args.get("create_if_missing", False)),
            )
        if action == "reset":
            return workspace.reset()
        raise ValueError(f"unsupported workspace action: {action}")

    return ToolSpec(
        name="guga_workspace",
        description=(
            "Inspect, change, or reset the task workspace for this process. "
            "Inspect before using file or command tools, and inspect again after changing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["inspect", "set", "reset"]},
                "path": {
                    "type": "string",
                    "description": "Absolute path or a path relative to the current workspace.",
                },
                "create_if_missing": {
                    "type": "boolean",
                    "description": "Create a missing workspace only when session policy allows it.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _time_parse_tool() -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        reference_time = str(args.get("reference_time", "") or "")
        reference = parse_datetime(reference_time) if reference_time else now_beijing()
        extracted = extract_semantic_time(text, reference_time=reference)
        if extracted is None:
            return {
                "text": text,
                "reference_time": reference.isoformat(timespec="seconds"),
                "matched": False,
            }
        valid_at, source, granularity = extracted
        return {
            "text": text,
            "reference_time": reference.isoformat(timespec="seconds"),
            "matched": True,
            "valid_at": valid_at.isoformat(timespec="seconds"),
            "semantic_day": valid_at.date().isoformat(),
            "time_source": source,
            "time_granularity": granularity,
        }

    return ToolSpec(
        name="guga_parse_time",
        description="Parse explicit or relative time expressions in user text using Beijing time.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text containing a time expression."},
                "reference_time": {
                    "type": "string",
                    "description": "Optional ISO datetime used as the reference time.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _list_dir_tool(workspace: WorkspaceContext) -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        root = workspace.require_confirmed()
        target = _resolve_safe_path(root, str(args.get("path", ".")))
        if not target.exists():
            return {"ok": False, "error": "path does not exist", "path": str(target)}
        if not target.is_dir():
            return {"ok": False, "error": "path is not a directory", "path": str(target)}
        limit = _clamp_int(args.get("limit"), default=100, minimum=1, maximum=500)
        rows = []
        for child in sorted(target.iterdir(), key=lambda item: item.name.lower())[:limit]:
            rows.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
        return {"path": str(target), "entries": rows}

    return ToolSpec(
        name="guga_list_dir",
        description="List files and folders under the confirmed session workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path under the session workspace."},
                "limit": {"type": "integer", "description": "Maximum number of entries."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _read_file_tool(workspace: WorkspaceContext) -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        root = workspace.require_confirmed()
        target = _resolve_safe_path(root, str(args.get("path", "")))
        if not target.exists():
            return {"ok": False, "error": "path does not exist", "path": str(target)}
        if not target.is_file():
            return {"ok": False, "error": "path is not a file", "path": str(target)}
        max_chars = _clamp_int(args.get("max_chars"), default=12000, minimum=1, maximum=50000)
        text = target.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(target),
            "truncated": len(text) > max_chars,
            "content": text[:max_chars],
        }

    return ToolSpec(
        name="guga_read_file",
        description="Read a UTF-8 text file under the confirmed session workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path under the session workspace."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _write_file_tool(workspace: WorkspaceContext) -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        if os.environ.get("Guga_ENABLE_WRITE_TOOL", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return {"ok": False, "error": "write tool disabled; set Guga_ENABLE_WRITE_TOOL=1 to enable"}
        root = workspace.require_confirmed()
        target = _resolve_safe_path(root, str(args.get("path", "")))
        content = str(args.get("content", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes": len(content.encode("utf-8"))}

    return ToolSpec(
        name="guga_write_file",
        description="Write a UTF-8 text file under the confirmed session workspace. Disabled unless explicitly enabled.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path under the session workspace."},
                "content": {"type": "string", "description": "File content to write."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _run_command_tool(workspace: WorkspaceContext) -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        if os.environ.get("Guga_ENABLE_COMMAND_TOOL", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return {"ok": False, "error": "command tool disabled; set Guga_ENABLE_COMMAND_TOOL=1 to enable"}
        root = workspace.require_confirmed()
        command = str(args.get("command", "")).strip()
        if not command:
            return {"ok": False, "error": "empty command"}
        timeout = _clamp_int(args.get("timeout_seconds"), default=10, minimum=1, maximum=60)
        completed = subprocess.run(
            command,
            cwd=root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }

    return ToolSpec(
        name="guga_run_command",
        description="Run a shell command in the confirmed session workspace. Disabled unless explicitly enabled.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_seconds": {"type": "integer", "description": "Timeout between 1 and 60 seconds."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def encode_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def parse_tool_arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _resolve_safe_path(root: Path, user_path: str) -> Path:
    candidate = Path(user_path)
    if candidate.is_absolute():
        target = candidate.resolve()
    else:
        target = (root / candidate).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"path escapes project root: {user_path}")
    return target


def _clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _matches_json_type(value: object, expected: str | list[str]) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    return any(checks.get(item, lambda _: False)(value) for item in expected_types)
