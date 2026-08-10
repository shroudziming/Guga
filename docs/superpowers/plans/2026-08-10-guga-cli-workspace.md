# Guga CLI Session Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command `guga_cli.ps1` launcher with Git-tracked startup configuration and a session-only workspace tool that all approved file and command operations must confirm before use.

**Architecture:** A thread-safe `WorkspaceContext` owns the default root, current root, creation policy, and confirmation flag for one Python process. The task tool registry shares that context so workspace changes immediately affect list/read/write/command tools; LangGraph invalidates confirmation at task start and plan revision. A root PowerShell launcher loads sensitive API connection values from ignored `.env`, loads non-sensitive startup behavior from tracked `config/guga_cli.env`, creates `Desktop/Guga`, and launches the existing CLI.

**Tech Stack:** Python 3.11, LangGraph 1.x, `unittest`, PowerShell 5+/7, existing Guga tool registry and CLI.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md`.
- The launcher filename is exactly `guga_cli.ps1` at the repository root.
- Git-tracked startup configuration is exactly `config/guga_cli.env`; API keys and base URLs remain only in ignored `.env`.
- `Guga_CLI_MODEL_ROUTE=api` is the committed default; valid values are exactly `api` and `local`.
- The default workspace is the current Windows user's `Desktop/Guga` directory and is created at launcher startup.
- Workspace changes are process-local and are never written to config, checkpoint, or long-term memory.
- `Guga_ENABLE_WRITE_TOOL=1` and `Guga_ENABLE_COMMAND_TOOL=1` are committed defaults, with comments stating `1 = allowed` and `0 = disabled`.
- No tool, including workspace inspection, runs before plan approval.
- Operational tools reject execution until the current workspace has been inspected; `set`, `reset`, new task start, and plan revision invalidate confirmation.
- Ordinary chat continues to expose only `guga_parse_time`.
- Preserve existing output truncation behavior and do not add full artifact archiving.
- Never print, trace, commit, or copy an API key.
- Keep every Git commit to one file. Red contract tests and production changes are committed separately and pushed to `feat/langgraph-agent-runtime`.

## File Structure

- Create `guga/workspace.py`: process-local workspace state, validation, switching, reset, confirmation, and environment construction.
- Modify `guga/tools.py`: register `guga_workspace`, share `WorkspaceContext`, and resolve every operational call against the live root.
- Modify `guga/agent/model_adapter.py`: tell the planner to include workspace confirmation before operational steps.
- Modify `guga/agent/graph.py`: invalidate workspace confirmation when a plan is revised.
- Modify `guga/agent/runner.py`: invalidate workspace confirmation at each new task start.
- Modify `src/basic_cli_chat.py`: construct the task registry from launcher-provided workspace environment.
- Create `config/guga_cli.env`: tracked, commented startup defaults without secrets.
- Create `guga_cli.ps1`: load both config sources, derive model variables, create the default workspace, and launch the CLI.
- Modify `README.md`: document one-command startup, route switching, config separation, and workspace behavior.
- Create `test/test_workspace_context.py`: unit contracts for process-local workspace state.
- Modify `test/test_tool_calling.py`: contracts for the workspace tool and operational gating.
- Modify `test/test_agent_model_adapter.py`: planner prompt contract.
- Modify `test/test_agent_task_graph.py`: revision invalidation contract.
- Modify `test/test_agent_runner.py`: new-task invalidation contract.
- Modify `test/test_basic_cli_agent_wiring.py`: CLI environment wiring contract.
- Create `test/test_guga_cli_launcher.py`: tracked config and PowerShell syntax contracts.

---

### Task 1: Process-local `WorkspaceContext`

**Files:**
- Create: `test/test_workspace_context.py`
- Create: `guga/workspace.py`

**Interfaces:**
- Produces: `WorkspaceError(ValueError)`.
- Produces: `WorkspaceContext(default_root: Path, allow_create: bool = False)`.
- Produces methods `inspect() -> dict`, `set(path: str, create_if_missing: bool = False) -> dict`, `reset() -> dict`, `require_confirmed() -> Path`, `invalidate_confirmation() -> None`.
- Produces properties `default_root: Path`, `current_root: Path`, `confirmed: bool`, `allow_create: bool`.
- Produces: `workspace_context_from_env(fallback_root: Path, env: Mapping[str, str] | None = None) -> WorkspaceContext` using `Guga_CLI_DEFAULT_WORKSPACE_PATH` and `Guga_CLI_ALLOW_CREATE_WORKSPACE`.

- [ ] **Step 1: Write the failing workspace state contract**

Create `test/test_workspace_context.py` with real temporary directories:

```python
class WorkspaceContextTest(unittest.TestCase):
    def test_inspect_set_reinspect_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default = root / "default"
            other = root / "other"
            default.mkdir()
            context = WorkspaceContext(default, allow_create=True)

            with self.assertRaisesRegex(WorkspaceError, "inspect"):
                context.require_confirmed()
            self.assertEqual(default.resolve(), Path(context.inspect()["current_root"]))
            self.assertTrue(context.confirmed)

            changed = context.set(str(other), create_if_missing=True)
            self.assertEqual(default.resolve(), Path(changed["previous_root"]))
            self.assertEqual(other.resolve(), Path(changed["current_root"]))
            self.assertFalse(context.confirmed)
            self.assertTrue(other.is_dir())

            context.inspect()
            context.reset()
            self.assertEqual(default.resolve(), context.current_root)
            self.assertFalse(context.confirmed)

    def test_failed_switch_preserves_root_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default = Path(temp_dir) / "default"
            default.mkdir()
            context = WorkspaceContext(default, allow_create=False)
            context.inspect()

            with self.assertRaisesRegex(WorkspaceError, "does not exist"):
                context.set(str(Path(temp_dir) / "missing"), create_if_missing=True)

            self.assertEqual(default.resolve(), context.current_root)
            self.assertTrue(context.confirmed)

    def test_relative_switch_and_environment_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default = Path(temp_dir) / "default"
            child = default / "child"
            child.mkdir(parents=True)
            context = workspace_context_from_env(
                Path(temp_dir) / "fallback",
                {
                    "Guga_CLI_DEFAULT_WORKSPACE_PATH": str(default),
                    "Guga_CLI_ALLOW_CREATE_WORKSPACE": "1",
                },
            )
            context.inspect()
            context.set("child")
            self.assertEqual(child.resolve(), context.current_root)
            self.assertTrue(context.allow_create)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m unittest discover -s test -p "test_workspace_context.py"
```

Expected: import error for missing `guga.workspace`.

- [ ] **Step 3: Commit the red contract as one file**

```powershell
git add -- test/test_workspace_context.py
git commit -m "to(cli):定义会话工作区契约"
git push
```

- [ ] **Step 4: Implement `guga/workspace.py`**

Implement a locked state object. Resolve all paths with `Path.resolve()`, require `default_root` to exist and be a directory, require prior confirmation before `set`, and mutate state only after every validation and optional directory creation succeeds.

Core shape:

```python
class WorkspaceError(ValueError):
    pass


class WorkspaceContext:
    def __init__(self, default_root: Path, allow_create: bool = False) -> None:
        resolved = Path(default_root).resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"default workspace is not a directory: {resolved}")
        self.default_root = resolved
        self.current_root = resolved
        self.allow_create = bool(allow_create)
        self._confirmed = False
        self._lock = RLock()

    @property
    def confirmed(self) -> bool:
        with self._lock:
            return self._confirmed

    def inspect(self) -> dict:
        with self._lock:
            self._confirmed = True
            return {
                "current_root": str(self.current_root),
                "default_root": str(self.default_root),
                "confirmed": True,
            }

    def require_confirmed(self) -> Path:
        with self._lock:
            if not self._confirmed:
                raise WorkspaceError("inspect the current workspace before operational tools")
            return self.current_root
```

`set()` resolves relative paths under `current_root`; it creates a missing directory only when both `allow_create` and `create_if_missing` are true. `reset()` assigns `default_root`. Successful `set()`/`reset()` call `invalidate_confirmation()`.

`workspace_context_from_env()` treats `1`, `true`, `yes`, and `on` as true and otherwise false. An empty workspace path uses `fallback_root`.

- [ ] **Step 5: Run workspace tests and verify GREEN**

```powershell
python -m unittest discover -s test -p "test_workspace_context.py"
```

Expected: all workspace tests pass.

- [ ] **Step 6: Commit and push the single production file**

```powershell
git add -- guga/workspace.py
git commit -m "feat(cli):增加会话工作区状态"
git push
```

---

### Task 2: Dynamic workspace tool registry

**Files:**
- Modify: `test/test_tool_calling.py`
- Modify: `guga/tools.py`

**Interfaces:**
- Consumes: `WorkspaceContext` from Task 1.
- Changes: `ToolRegistry.__init__(tools=None, workspace: WorkspaceContext | None = None)`.
- Produces: `ToolRegistry.workspace -> WorkspaceContext | None` and `invalidate_workspace_confirmation() -> None`.
- Changes: `default_tool_registry(project_root: Path | None = None, workspace: WorkspaceContext | None = None) -> ToolRegistry`.
- Adds tool: `guga_workspace(action, path?, create_if_missing?)`.

- [ ] **Step 1: Add failing registry and gating tests**

Extend `ToolCallingTest` in `test/test_tool_calling.py`:

```python
def test_operational_tools_require_workspace_inspection_and_follow_switch(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        default = Path(temp_dir) / "default"
        other = Path(temp_dir) / "other"
        default.mkdir()
        other.mkdir()
        (default / "value.txt").write_text("default", encoding="utf-8")
        (other / "value.txt").write_text("other", encoding="utf-8")
        workspace = WorkspaceContext(default, allow_create=True)
        registry = default_tool_registry(workspace=workspace)

        denied = registry.execute(ToolCall("1", "guga_read_file", {"path": "value.txt"}))
        self.assertFalse(denied["ok"])
        self.assertIn("inspect", denied["error"])

        inspected = registry.execute(ToolCall("2", "guga_workspace", {"action": "inspect"}))
        self.assertTrue(inspected["confirmed"])
        self.assertEqual("default", registry.execute(
            ToolCall("3", "guga_read_file", {"path": "value.txt"})
        )["content"])

        registry.execute(ToolCall(
            "4", "guga_workspace", {"action": "set", "path": str(other)}
        ))
        denied_after_set = registry.execute(
            ToolCall("5", "guga_read_file", {"path": "value.txt"})
        )
        self.assertFalse(denied_after_set["ok"])
        registry.execute(ToolCall("6", "guga_workspace", {"action": "inspect"}))
        self.assertEqual("other", registry.execute(
            ToolCall("7", "guga_read_file", {"path": "value.txt"})
        )["content"])

def test_task_registry_contains_workspace_but_conversation_registry_does_not(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        task_names = default_tool_registry(Path(temp_dir)).names()
    self.assertIn("guga_workspace", task_names)
    self.assertEqual({"guga_parse_time"}, conversation_tool_registry().names())
```

Update existing direct operational-tool tests to call `guga_workspace inspect` before read/write/command execution. Do not weaken the new default gate.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest discover -s test -p "test_tool_calling.py"
```

Expected: missing `workspace` argument/tool and operational calls are not gated.

- [ ] **Step 3: Commit the test file only**

```powershell
git add -- test/test_tool_calling.py
git commit -m "to(tools):定义动态工作区工具契约"
git push
```

- [ ] **Step 4: Implement dynamic registry behavior in `guga/tools.py`**

Store the shared context on `ToolRegistry`. In `default_tool_registry`, create `WorkspaceContext(project_root or PROJECT_ROOT)` only when no context is injected. Register `guga_workspace` before list/read/write/command.

The workspace handler must use exact actions:

```python
def handler(args: dict[str, Any]) -> dict[str, Any]:
    action = str(args.get("action", "")).strip().lower()
    if action == "inspect":
        return workspace.inspect()
    if action == "set":
        return workspace.set(
            str(args.get("path", "")),
            create_if_missing=bool(args.get("create_if_missing", False)),
        )
    if action == "reset":
        return workspace.reset()
    raise WorkspaceError(f"unknown workspace action: {action}")
```

Use an OpenAI schema with `action` enum `inspect`, `set`, `reset`, a string `path`, a boolean `create_if_missing`, and `additionalProperties: false`.

At the start of every list/read/write/command handler, call `root = workspace.require_confirmed()`. Resolve paths against that returned root. Run commands with `cwd=root`. Preserve current output limits and environment gates.

`invalidate_workspace_confirmation()` is a no-op when the registry has no workspace and otherwise calls the context method.

- [ ] **Step 5: Run tool and model-adapter regressions**

```powershell
python -m unittest discover -s test -p "test_tool_calling.py"
python -m unittest discover -s test -p "test_agent_model_adapter.py"
```

Expected: both modules pass; strict argument schema validation also covers `guga_workspace`.

- [ ] **Step 6: Commit and push `guga/tools.py` only**

```powershell
git add -- guga/tools.py
git commit -m "feat(tools):支持会话级动态工作区"
git push
```

---

### Task 3: Planner and LangGraph confirmation lifecycle

**Files:**
- Modify: `test/test_agent_model_adapter.py`
- Modify: `guga/agent/model_adapter.py`
- Modify: `test/test_agent_task_graph.py`
- Modify: `guga/agent/graph.py`
- Modify: `test/test_agent_runner.py`
- Modify: `guga/agent/runner.py`

**Interfaces:**
- Consumes: `ToolRegistry.invalidate_workspace_confirmation()` from Task 2.
- Behavior: planner requests workspace inspection before any operational tool and another inspection after workspace changes.
- Behavior: new tasks and plan revisions invalidate the shared context without resetting `current_root`.

- [ ] **Step 1: Add a failing planner prompt contract**

In `test/test_agent_model_adapter.py`, create a valid plan response and call `create_plan()`, then assert the first model request contains all mandatory workspace rules:

```python
prompt = json.dumps(model.messages_seen[0], ensure_ascii=False)
self.assertIn("guga_workspace", prompt)
self.assertIn("inspect", prompt)
self.assertIn("文件或命令", prompt)
self.assertIn("切换工作区后再次", prompt)
```

- [ ] **Step 2: Run RED and commit only the test file**

```powershell
python -m unittest discover -s test -p "test_agent_model_adapter.py"
git add -- test/test_agent_model_adapter.py
git commit -m "to(agent):定义工作区规划规则"
git push
```

Expected before commit: assertion failure because the control prompt lacks these rules.

- [ ] **Step 3: Add the control rule to `AgentModelAdapter.create_plan()`**

Extend only the internal planning system prompt. State that plans using `guga_list_dir`, `guga_read_file`, `guga_write_file`, or `guga_run_command` must first dedicate a step to `guga_workspace inspect`; a `set`/`reset` step must be followed by another inspect step. Do not inject persona language into this rule.

- [ ] **Step 4: Run GREEN and commit only the adapter file**

```powershell
python -m unittest discover -s test -p "test_agent_model_adapter.py"
git add -- guga/agent/model_adapter.py
git commit -m "feat(agent):要求计划确认工作区"
git push
```

- [ ] **Step 5: Add a failing plan-revision invalidation contract**

In `test/test_agent_task_graph.py`, build the graph with a registry backed by a real `WorkspaceContext`. Inspect it before running a fake plan whose first verification requests replanning. After the graph returns to the second approval interrupt, assert `workspace.confirmed is False` while `workspace.current_root` is unchanged.

- [ ] **Step 6: Run RED and commit only the graph test file**

```powershell
python -m unittest discover -s test -p "test_agent_task_graph.py"
git add -- test/test_agent_task_graph.py
git commit -m "to(agent):覆盖修订计划工作区失效"
git push
```

Expected before commit: the workspace remains confirmed after revision.

- [ ] **Step 7: Invalidate confirmation in the graph revision node**

At the start of `revise_plan(state)`, call:

```python
tools.invalidate_workspace_confirmation()
```

Keep the existing trace and approval flow unchanged.

- [ ] **Step 8: Run GREEN and commit only `guga/agent/graph.py`**

```powershell
python -m unittest discover -s test -p "test_agent_task_graph.py"
git add -- guga/agent/graph.py
git commit -m "fix(agent):修订计划时重验工作区"
git push
```

- [ ] **Step 9: Add a failing new-task invalidation contract**

In `test/test_agent_runner.py`, inject a registry with a real workspace, call `workspace.inspect()`, then consume `runner.start()` only until the approval interrupt. Assert the current root is unchanged and `workspace.confirmed is False`.

- [ ] **Step 10: Run RED and commit only the runner test file**

```powershell
python -m unittest discover -s test -p "test_agent_runner.py"
git add -- test/test_agent_runner.py
git commit -m "to(agent):覆盖新任务工作区失效"
git push
```

- [ ] **Step 11: Invalidate confirmation at task start**

In `AgentTaskRunner.start()`, after unfinished-task validation and before memory retrieval, call `self.tools.invalidate_workspace_confirmation()`. Do not reset the workspace root.

- [ ] **Step 12: Run GREEN and commit only the runner file**

```powershell
python -m unittest discover -s test -p "test_agent_runner.py"
git add -- guga/agent/runner.py
git commit -m "fix(agent):新任务重新确认工作区"
git push
```

---

### Task 4: Wire the launcher workspace into the text CLI

**Files:**
- Modify: `test/test_basic_cli_agent_wiring.py`
- Modify: `src/basic_cli_chat.py`

**Interfaces:**
- Consumes: `workspace_context_from_env(PROJECT_ROOT, os.environ)` from Task 1.
- Consumes: `default_tool_registry(workspace=workspace)` from Task 2.
- Environment input: `Guga_CLI_DEFAULT_WORKSPACE_PATH` and `Guga_CLI_ALLOW_CREATE_WORKSPACE`.

- [ ] **Step 1: Add a failing CLI wiring test**

Extend `test/test_basic_cli_agent_wiring.py`. Use a temporary existing directory and patch the process environment:

```python
patch.dict(
    cli.os.environ,
    {
        "Guga_DEBUG": "0",
        "Guga_MODEL_ID": "fake-model",
        "Guga_CLI_DEFAULT_WORKSPACE_PATH": temp_dir,
        "Guga_CLI_ALLOW_CREATE_WORKSPACE": "1",
    },
    clear=False,
)
```

After `cli.main()` exits, assert the registry passed to `AgentTaskRunner` has a workspace whose `default_root` equals the temporary directory, `allow_create` is true, and `confirmed` is false. Retain the existing assertion that `ChatSession` receives only `guga_parse_time`.

- [ ] **Step 2: Run RED and commit only the test file**

```powershell
python -m unittest discover -s test -p "test_basic_cli_agent_wiring.py"
git add -- test/test_basic_cli_agent_wiring.py
git commit -m "to(cli):定义启动工作区接线"
git push
```

Expected: the task registry still captures `PROJECT_ROOT` and exposes no workspace property.

- [ ] **Step 3: Construct the task registry from environment**

In `src/basic_cli_chat.py`:

```python
from guga.workspace import workspace_context_from_env

workspace = workspace_context_from_env(PROJECT_ROOT, os.environ)
task_tools = default_tool_registry(workspace=workspace)
```

Keep model, memory manager, persona, runner, and command controller construction unchanged. Do not add workspace tools to ordinary chat.

- [ ] **Step 4: Run CLI and chat regressions**

```powershell
python -m unittest discover -s test -p "test_basic_cli_agent_wiring.py"
python -m unittest discover -s test -p "test_chat_session_rag_flow.py"
python -m unittest discover -s test -p "test_tool_calling.py"
```

Expected: all pass.

- [ ] **Step 5: Commit and push the CLI file only**

```powershell
git add -- src/basic_cli_chat.py
git commit -m "feat(cli):接入默认会话工作区"
git push
```

---

### Task 5: Tracked startup config and `guga_cli.ps1`

**Files:**
- Create: `test/test_guga_cli_launcher.py`
- Create: `config/guga_cli.env`
- Create: `guga_cli.ps1`

**Interfaces:**
- Config keys: `Guga_CLI_MODEL_ROUTE`, `Guga_CLI_API_MODEL_ID`, `Guga_CLI_LOCAL_MODEL_ID`, `Guga_CLI_LOCAL_CACHE_DIR`, `Guga_CLI_DEFAULT_WORKSPACE`, `Guga_CLI_ALLOW_CREATE_WORKSPACE`, `Guga_ENABLE_WRITE_TOOL`, `Guga_ENABLE_COMMAND_TOOL`, `Guga_DEBUG`.
- Derived environment: `Guga_MODEL_PROVIDER`, `Guga_MODEL_ID`, `Guga_CACHE_DIR`, `Guga_CLI_DEFAULT_WORKSPACE_PATH`.
- Sensitive environment remains sourced from `.env`: `Guga_API_KEY`/`OPENAI_API_KEY` and `Guga_API_BASE_URL`/`OPENAI_BASE_URL`.

- [ ] **Step 1: Write failing tracked-config and script syntax tests**

Create `test/test_guga_cli_launcher.py`:

```python
class GugaCliLauncherTest(unittest.TestCase):
    def test_tracked_config_defaults_are_explicit_and_non_sensitive(self) -> None:
        text = (PROJECT_ROOT / "config" / "guga_cli.env").read_text(encoding="utf-8")
        self.assertIn("Guga_CLI_MODEL_ROUTE=api", text)
        self.assertIn("Guga_ENABLE_WRITE_TOOL=1", text)
        self.assertIn("Guga_ENABLE_COMMAND_TOOL=1", text)
        self.assertIn("1 = 允许", text)
        self.assertIn("0 = 禁止", text)
        self.assertNotIn("API_KEY=", text)
        self.assertNotIn("OPENAI_API_KEY=", text)

    def test_powershell_launcher_has_valid_syntax(self) -> None:
        launcher = PROJECT_ROOT / "guga_cli.ps1"
        escaped = str(launcher).replace("'", "''")
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped}',"
            "[ref]$tokens,[ref]$errors) > $null; "
            "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
```

- [ ] **Step 2: Run RED and commit only the test file**

```powershell
python -m unittest discover -s test -p "test_guga_cli_launcher.py"
git add -- test/test_guga_cli_launcher.py
git commit -m "to(cli):定义启动配置与脚本契约"
git push
```

Expected: missing config and launcher files.

- [ ] **Step 3: Create the tracked config with the approved comments**

Create `config/guga_cli.env` exactly from the approved spec. It must default to API, allow write and command execution, disable debug output, select `desktop`, and explain every boolean value. Do not include placeholders resembling a real key.

- [ ] **Step 4: Run only the config test and commit the config file**

```powershell
python test/test_guga_cli_launcher.py GugaCliLauncherTest.test_tracked_config_defaults_are_explicit_and_non_sensitive
git add -- config/guga_cli.env
git commit -m "feat(cli):增加可提交启动配置"
git push
```

Expected: config test passes; full module still fails because the launcher is absent.

- [ ] **Step 5: Implement `guga_cli.ps1`**

Use `$PSScriptRoot` as repository root and `$ErrorActionPreference = 'Stop'`. Implement two parsers:

1. Import all non-comment `KEY=VALUE` lines from optional root `.env` into process environment without printing values.
2. Import only the approved key allowlist from required `config/guga_cli.env`; reject unknown keys so a committed file cannot inject `Guga_API_KEY`.

Derive route values:

```powershell
$route = $env:Guga_CLI_MODEL_ROUTE.Trim().ToLowerInvariant()
switch ($route) {
    'api' {
        $env:Guga_MODEL_PROVIDER = 'api'
        $env:Guga_MODEL_ID = $env:Guga_CLI_API_MODEL_ID
        if (-not ($env:Guga_API_KEY -or $env:OPENAI_API_KEY)) {
            throw 'API route requires Guga_API_KEY or OPENAI_API_KEY in .env'
        }
        if (-not ($env:Guga_API_BASE_URL -or $env:OPENAI_BASE_URL)) {
            throw 'API route requires Guga_API_BASE_URL or OPENAI_BASE_URL in .env'
        }
    }
    'local' {
        $env:Guga_MODEL_PROVIDER = 'local'
        $env:Guga_MODEL_ID = $env:Guga_CLI_LOCAL_MODEL_ID
        $cache = [IO.Path]::GetFullPath((Join-Path $repoRoot $env:Guga_CLI_LOCAL_CACHE_DIR))
        $env:Guga_CACHE_DIR = $cache
    }
    default { throw "Unsupported Guga_CLI_MODEL_ROUTE: $route" }
}
```

Require `Guga_CLI_DEFAULT_WORKSPACE=desktop`; resolve `[Environment]::GetFolderPath('Desktop')`, append `Guga`, create it with `New-Item -ItemType Directory -Force`, and set `Guga_CLI_DEFAULT_WORKSPACE_PATH` to the resolved directory. Keep the user-facing comments in the script concise and explain the two config sources, derived model variables, tool gates, and session-only workspace.

Launch in the same console:

```powershell
Push-Location $repoRoot
try {
    & python -u src\basic_cli_chat.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
```

- [ ] **Step 6: Run launcher syntax and focused CLI tests**

```powershell
python -m unittest discover -s test -p "test_guga_cli_launcher.py"
python -m unittest discover -s test -p "test_basic_cli_agent_wiring.py"
```

Expected: all pass without starting a model or reading a real API key.

- [ ] **Step 7: Commit and push only the launcher file**

```powershell
git add -- guga_cli.ps1
git commit -m "feat(cli):增加一键启动脚本"
git push
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: `.\guga_cli.ps1`, config separation, API/local route switch, default `Desktop/Guga`, `guga_workspace` actions, and session-only switching.

- [ ] **Step 1: Update README in one focused edit**

Replace the primary text CLI command with:

```powershell
.\guga_cli.ps1
```

Document:

- `.env` contains only private API connection data and remains ignored.
- `config/guga_cli.env` is tracked and contains route/tool/workspace defaults.
- Change only `Guga_CLI_MODEL_ROUTE=local` to choose the configured local model.
- Default workspace is the actual Windows desktop `Guga` folder.
- `guga_workspace inspect`, `set`, and `reset` are task-only tools.
- Workspace switching lasts only until process exit.
- Write and command defaults are allowed, but plan approval and workspace inspection remain mandatory.

- [ ] **Step 2: Check Markdown and commit only README**

```powershell
git diff --check -- README.md
git add -- README.md
git commit -m "docs(cli):说明启动与工作区配置"
git push
```

- [ ] **Step 3: Run focused acceptance tests**

```powershell
python -m unittest discover -s test -p "test_workspace_context.py"
python -m unittest discover -s test -p "test_tool_calling.py"
python -m unittest discover -s test -p "test_agent_model_adapter.py"
python -m unittest discover -s test -p "test_agent_task_graph.py"
python -m unittest discover -s test -p "test_agent_runner.py"
python -m unittest discover -s test -p "test_basic_cli_agent_wiring.py"
python -m unittest discover -s test -p "test_guga_cli_launcher.py"
```

Expected: all pass.

- [ ] **Step 4: Run the complete repository verification**

```powershell
python -m unittest discover -s test
python -m compileall -q guga src
git diff --check origin/main...HEAD
git status --short
```

Expected: full suite passes, compileall exits zero, diff check is clean, and `git status --short` is empty.

- [ ] **Step 5: Audit branch and commit granularity**

```powershell
$multi = @()
git log --format="%H" origin/main..HEAD | ForEach-Object {
    $files = @(git diff-tree --no-commit-id --name-only -r $_)
    if ($files.Count -ne 1) { $multi += "${_}:$($files.Count)" }
}
if ($multi.Count) { throw "Multi-file commits: $($multi -join ', ')" }
if ((git rev-parse HEAD) -ne (git rev-parse origin/feat/langgraph-agent-runtime)) {
    throw "Remote feature branch is not synchronized"
}
```

Expected: no multi-file commits and local/remote feature SHA match.

- [ ] **Step 6: Preserve the branch for integration choice**

Do not merge `main`. Report the worktree path, branch name, latest SHA, test count, and any external API smoke test not performed. Use `superpowers:finishing-a-development-branch` to present merge/PR/keep options.
