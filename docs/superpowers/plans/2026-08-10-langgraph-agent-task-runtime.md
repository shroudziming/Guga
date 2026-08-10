# LangGraph Agent Task Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph-backed Guga task runtime that shows a plan for approval, executes approved tools until a verified terminal state, retries a mismatched step at most three times, records a developer-readable trace, and stores a long-term task outcome containing the trace reference.

**Architecture:** Keep `ChatSession`, the existing model implementations, `ToolRegistry`, and `MemoryManager` as reusable dependencies. Add a focused `guga.agent` package that owns structured task state, model adaptation, LangGraph routing, approval interrupts, trace recording, and terminal outcome delivery. Enter this runtime explicitly through `/task`; ordinary chat receives only the conversation tool registry so it cannot bypass task-plan approval.

**Tech Stack:** Python 3, LangGraph 1.x, OpenAI-compatible tool calling, JSONL persistence, `unittest`.

## Global Constraints

- Work from the repository root `D:\work\LLM\Guga`.
- At execution time, use `superpowers:using-git-worktrees` before changing implementation files.
- Create one feature branch for the entire implementation and push that branch, not `main`, after each commit.
- Each task changes exactly one file, and that file is committed exactly once during this implementation.
- Never stage another file with the current task's file.
- Test-first commits may be red and use the `to` commit type; the immediately following implementation tasks must make the relevant test group green.
- Use commit messages in `<type>(<scope>): <subject>` form with subjects under 50 characters.
- A plan revision always requires a new approval interrupt.
- Approved plans allow only the tools declared by the current step.
- A mismatched step gets at most three actual tool executions.
- Raw execution trace stays outside chat history and user semantic events.
- `TaskOutcome` contains a trace reference but never copies raw trace rows into long-term memory.
- Current tool-output truncation remains unchanged and is documented as a known limitation.

---

## File Map

### New runtime files

- `guga/agent/__init__.py`: marks the package and exports the public runner types.
- `guga/agent/state.py`: owns plan, action, verification, and task-state contracts.
- `guga/agent/outcome.py`: owns the immutable terminal task summary passed to memory.
- `guga/agent/trace.py`: appends developer-readable JSONL execution events.
- `guga/agent/model_adapter.py`: converts existing Guga model APIs into planning, native tool selection, verification, and final-report operations.
- `guga/agent/graph.py`: owns LangGraph nodes, conditional routes, approval interrupts, and retry enforcement.
- `guga/agent/runner.py`: owns task IDs, checkpointer configuration, graph start/resume calls, and terminal outcome delivery.
- `guga/agent/cli.py`: converts `/task`, `/approve`, and `/reject` text into runner calls and renderable CLI results.
- `guga/memory/task_outcomes.py`: stores one idempotent terminal outcome per task.

### Modified runtime files

- `requirements.txt`: adds the LangGraph dependency.
- `guga/utils/paths.py`: provides the per-agent run directory.
- `guga/tools.py`: exposes registry names, filtered tool schemas, and a conversation-only registry.
- `guga/config.py`: defines the fixed three-attempt default.
- `guga/memory/manager.py`: constructs the outcome store and exposes `record_task_outcome()`.
- `src/basic_cli_chat.py`: constructs the task runner/controller and routes explicit task commands.
- `README.md`: documents task usage, approval behavior, trace location, and current limitations.

### Test files

- `test/test_agent_trace.py`: trace ordering and JSONL contents.
- `test/test_agent_task_memory.py`: idempotent outcome storage and memory bridge.
- `test/test_agent_model_adapter.py`: structured plan/verification parsing and native tool selection.
- `test/test_agent_task_graph.py`: approval, completion, retry, replan, blocked, and tool-range behavior.
- `test/test_agent_runner.py`: task start/resume and terminal-memory delivery.
- `test/test_agent_cli.py`: explicit command routing and revised-plan presentation.
- `test/test_tool_calling.py`: filtered schemas and conversation-only tool registry.

---

### Task 1: Add the LangGraph dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: the existing pip requirements file.
- Produces: an installable `langgraph>=1.0,<2.0` runtime dependency.

- [ ] **Step 1: Add the dependency**

Append this dependency beside the other runtime libraries:

```text
langgraph>=1.0,<2.0
```

- [ ] **Step 2: Verify dependency syntax**

Run:

```powershell
python -m pip install --dry-run -r requirements.txt
```

Expected: dependency resolution succeeds and includes a LangGraph 1.x release.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add requirements.txt
git commit -m "chore(agent):增加LangGraph依赖"
git push origin HEAD
```

Expected: the commit contains only `requirements.txt`.

### Task 2: Define agent task state contracts

**Files:**
- Create: `guga/agent/state.py`

**Interfaces:**
- Produces: `PlanStep`, `ToolAction`, `Verification`, `AgentTaskState`, and `TaskStatus`.
- Consumed by: `model_adapter.py`, `graph.py`, `runner.py`, and task tests.

- [ ] **Step 1: Create the state types**

Use `TypedDict` so LangGraph can checkpoint JSON-compatible state:

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict


TaskStatus = Literal[
    "planning",
    "awaiting_approval",
    "executing",
    "completed",
    "failed",
    "blocked",
]


class PlanStep(TypedDict):
    id: str
    description: str
    expected_result: str
    allowed_tools: list[str]


class ToolAction(TypedDict):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str


class Verification(TypedDict):
    matched: bool
    reason: str
    requires_replan: bool
    blocked: bool


class AgentTaskState(TypedDict, total=False):
    task_id: str
    agent_id: str
    user_request: str
    status: TaskStatus
    plan: list[PlanStep]
    plan_revision: int
    approved_revision: int
    current_step_index: int
    current_action: ToolAction
    attempt: int
    max_attempts: int
    tool_result: dict[str, Any]
    verification: Verification
    evidence: list[dict[str, Any]]
    trace_ref: str
    final_response: str
```

- [ ] **Step 2: Verify the module compiles**

Run:

```powershell
python -m py_compile guga\agent\state.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/state.py
git commit -m "feat(agent):定义任务状态契约"
git push origin HEAD
```

### Task 3: Define the terminal outcome contract

**Files:**
- Create: `guga/agent/outcome.py`

**Interfaces:**
- Produces: `TaskOutcome` and `TaskOutcome.as_dict()`.
- Consumed by: `task_outcomes.py`, `manager.py`, and `runner.py`.

- [ ] **Step 1: Create the immutable outcome**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    goal: str
    status: str
    summary: str
    trace_ref: str
    completed_at: str
    tools_used: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tools_used"] = list(self.tools_used)
        return payload
```

- [ ] **Step 2: Verify the module compiles**

Run:

```powershell
python -m py_compile guga\agent\outcome.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/outcome.py
git commit -m "feat(agent):定义任务终态摘要"
git push origin HEAD
```

### Task 4: Specify trace behavior with a failing test

**Files:**
- Create: `test/test_agent_trace.py`

**Interfaces:**
- Consumes: `ExecutionTraceStore(root: Path)` from Task 6.
- Verifies: `trace_ref(task_id) -> str`, `append(task_id, event, payload) -> None`, and ordered JSONL rows.

- [ ] **Step 1: Write the trace contract test**

Create tests that use `TemporaryDirectory` and assert:

```python
store = ExecutionTraceStore(Path(tmp), agent_id="default")
store.append("task_1", "tool_call_started", {"call_id": "call_1"})
store.append("task_1", "tool_call_finished", {"call_id": "call_1", "ok": True})

trace_path = Path(store.trace_ref("task_1"))
rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

self.assertEqual([row["sequence"] for row in rows], [1, 2])
self.assertEqual(rows[0]["event"], "tool_call_started")
self.assertEqual(rows[1]["event"], "tool_call_finished")
self.assertEqual(rows[0]["task_id"], "task_1")
self.assertIn("created_at", rows[0])
```

Add a second test that creates two task IDs and proves they write to different files.

- [ ] **Step 2: Run the test and confirm the missing implementation**

Run:

```powershell
python -m unittest test.test_agent_trace
```

Expected: FAIL because `guga.agent.trace` does not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_trace.py
git commit -m "to(agent):定义执行轨迹测试"
git push origin HEAD
```

### Task 5: Add the agent-run path helper

**Files:**
- Modify: `guga/utils/paths.py`

**Interfaces:**
- Produces: `agent_runs_dir(agent_id: str = "default") -> Path`.
- Consumed by: `ExecutionTraceStore` construction in the CLI.

- [ ] **Step 1: Add the path helper**

```python
def agent_runs_dir(agent_id: str = "default") -> Path:
    return PROJECT_ROOT / "data" / "agent_runs" / agent_id
```

Keep it separate from `memory_data_dir()` because raw traces are not long-term memory records.

- [ ] **Step 2: Verify the helper**

Run:

```powershell
python -c "from guga.utils.paths import agent_runs_dir; print(agent_runs_dir('default'))"
```

Expected: output ends with `data\agent_runs\default`.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/utils/paths.py
git commit -m "feat(agent):增加执行轨迹路径"
git push origin HEAD
```

### Task 6: Implement the execution trace store

**Files:**
- Create: `guga/agent/trace.py`

**Interfaces:**
- Consumes: an explicit root `Path` and `agent_id`.
- Produces: `ExecutionTraceStore.append()` and `ExecutionTraceStore.trace_ref()`.

- [ ] **Step 1: Implement append-only per-task traces**

Use a lock and derive sequence numbers from the existing non-empty rows while holding the lock:

```python
class ExecutionTraceStore:
    def __init__(self, root: Path, agent_id: str) -> None:
        self.root = Path(root)
        self.agent_id = agent_id
        self._lock = RLock()

    def trace_ref(self, task_id: str) -> str:
        return str((self.root / task_id / "trace.jsonl").resolve())

    def append(self, task_id: str, event: str, payload: dict) -> None:
        target = Path(self.trace_ref(task_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            sequence = 1
            if target.exists():
                sequence += sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
            row = {
                "sequence": sequence,
                "event": event,
                "task_id": task_id,
                "agent_id": self.agent_id,
                "created_at": now_beijing_iso(),
                **payload,
            }
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Reject empty `task_id` and `event` values with `ValueError` before creating directories.

- [ ] **Step 2: Run the trace tests**

Run:

```powershell
python -m unittest test.test_agent_trace
```

Expected: PASS.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/trace.py
git commit -m "feat(agent):记录工具执行轨迹"
git push origin HEAD
```

### Task 7: Specify task-outcome memory behavior

**Files:**
- Create: `test/test_agent_task_memory.py`

**Interfaces:**
- Consumes: `TaskOutcomeStore.append(outcome) -> bool` and `MemoryManager.record_task_outcome(outcome) -> bool`.
- Verifies: idempotency by `task_id`, trace references, and separation from `semantic_events.jsonl`.

- [ ] **Step 1: Write the failing memory tests**

Cover these exact cases:

```python
outcome = TaskOutcome(
    task_id="task_1",
    goal="run tests",
    status="completed",
    summary="tests passed",
    trace_ref="D:/tmp/task_1/trace.jsonl",
    completed_at="2026-08-10T12:00:00+08:00",
    tools_used=("guga_run_command",),
)
self.assertTrue(store.append(outcome))
self.assertFalse(store.append(outcome))
```

Assert the JSONL contains exactly one row, `tools_used` is a JSON list, and `trace_ref` is unchanged. Construct a `MemoryManager(memory_root=Path(tmp), enable_semantic=False)`, call `record_task_outcome()` twice, and assert `task_outcomes.jsonl` has one row while `semantic_events.jsonl` is absent or contains no `task_id == "task_1"` row.

- [ ] **Step 2: Run the test and confirm missing APIs**

Run:

```powershell
python -m unittest test.test_agent_task_memory
```

Expected: FAIL because `TaskOutcomeStore` and `record_task_outcome()` do not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_task_memory.py
git commit -m "to(memory):定义任务终态记忆测试"
git push origin HEAD
```

### Task 8: Implement the task outcome store

**Files:**
- Create: `guga/memory/task_outcomes.py`

**Interfaces:**
- Consumes: `TaskOutcome`.
- Produces: `TaskOutcomeStore(file_path)` and `append(outcome) -> bool`.

- [ ] **Step 1: Implement idempotent JSONL append**

```python
class TaskOutcomeStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self._lock = RLock()

    def append(self, outcome: TaskOutcome) -> bool:
        with self._lock:
            rows = self.load_all()
            if any(row.get("task_id") == outcome.task_id for row in rows):
                return False
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome.as_dict(), ensure_ascii=False) + "\n")
            return True

    def load_all(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
```

Validate that `task_id`, `goal`, `status`, `summary`, `trace_ref`, and `completed_at` are non-empty before writing.

- [ ] **Step 2: Run the store-level test subset**

Run:

```powershell
python -m unittest test.test_agent_task_memory.TaskOutcomeStoreTest
```

Expected: PASS for direct store behavior; the `MemoryManager` bridge test remains failing.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/memory/task_outcomes.py
git commit -m "feat(memory):保存任务终态摘要"
git push origin HEAD
```

### Task 9: Connect terminal outcomes to MemoryManager

**Files:**
- Modify: `guga/memory/manager.py`

**Interfaces:**
- Consumes: `TaskOutcome` and `TaskOutcomeStore`.
- Produces: `MemoryManager.record_task_outcome(outcome: TaskOutcome) -> bool`.

- [ ] **Step 1: Construct the store and expose the bridge**

Add imports for `TaskOutcome` and `TaskOutcomeStore`. In `__init__`, after the existing store construction, add:

```python
self.task_outcome_store = TaskOutcomeStore(
    self.memory_root / "task_outcomes.jsonl"
)
```

Add this public method near the other record methods:

```python
def record_task_outcome(self, outcome: TaskOutcome) -> bool:
    return self.task_outcome_store.append(outcome)
```

Do not call `SemanticEventStore`, the summarizer, or user-model refresh from this method.

- [ ] **Step 2: Run the full outcome-memory test**

Run:

```powershell
python -m unittest test.test_agent_task_memory
```

Expected: PASS.

- [ ] **Step 3: Run existing memory smoke tests**

Run:

```powershell
python -m unittest test.test_memory_manager test.test_semantic_events
```

Expected: PASS.

- [ ] **Step 4: Commit and push only this file**

```powershell
git add guga/memory/manager.py
git commit -m "feat(memory):接收智能体任务终态"
git push origin HEAD
```

### Task 10: Specify registry filtering and conversation tools

**Files:**
- Modify: `test/test_tool_calling.py`

**Interfaces:**
- Consumes: `ToolRegistry.names()`, `ToolRegistry.openai_tools(names=None)`, and `conversation_tool_registry()`.
- Verifies: filtering preserves the existing default behavior and removes file/command tools from ordinary chat.

- [ ] **Step 1: Add registry tests to the existing test class**

Add assertions equivalent to:

```python
registry = default_tool_registry(project_root=Path(tmp))
self.assertIn("guga_run_command", registry.names())

schemas = registry.openai_tools(names={"guga_read_file"})
self.assertEqual(
    [item["function"]["name"] for item in schemas],
    ["guga_read_file"],
)

chat_registry = conversation_tool_registry()
self.assertEqual(chat_registry.names(), {"guga_parse_time"})
```

Keep every existing test unchanged.

- [ ] **Step 2: Run and confirm the new contract fails**

Run:

```powershell
python -m unittest test.test_tool_calling
```

Expected: FAIL because the new registry APIs do not exist.

- [ ] **Step 3: Commit and push only this red test modification**

```powershell
git add test/test_tool_calling.py
git commit -m "to(tools):定义工具范围测试"
git push origin HEAD
```

### Task 11: Add filtered schemas and conversation-only tools

**Files:**
- Modify: `guga/tools.py`

**Interfaces:**
- Produces: `ToolRegistry.names() -> set[str]`, filtered `openai_tools(names: set[str] | None = None)`, and `conversation_tool_registry()`.
- Preserves: existing unfiltered `openai_tools()` behavior.

- [ ] **Step 1: Add registry queries**

```python
def names(self) -> set[str]:
    return set(self._tools)

def openai_tools(self, names: set[str] | None = None) -> list[dict[str, Any]]:
    selected = self._tools.values()
    if names is not None:
        selected = [tool for tool in selected if tool.name in names]
    return [tool.to_openai_tool() for tool in selected]
```

Add the conversation registry:

```python
def conversation_tool_registry() -> ToolRegistry:
    return ToolRegistry([_time_parse_tool()])
```

Do not change `default_tool_registry()` or the handlers.

- [ ] **Step 2: Run all tool tests**

Run:

```powershell
python -m unittest test.test_tool_calling test.test_openai_compatible_chat_model
```

Expected: PASS.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/tools.py
git commit -m "feat(tools):区分任务与对话工具"
git push origin HEAD
```

### Task 12: Add the fixed task-attempt configuration

**Files:**
- Modify: `guga/config.py`

**Interfaces:**
- Produces: `DEFAULT_AGENT_MAX_ATTEMPTS = 3`.
- Consumed by: `AgentTaskRunner` initial state.

- [ ] **Step 1: Add the constant**

```python
DEFAULT_AGENT_MAX_ATTEMPTS = 3
```

Do not add an environment override in the first version; the agreed execution protocol fixes the value at three.

- [ ] **Step 2: Verify the value**

Run:

```powershell
python -c "from guga.config import DEFAULT_AGENT_MAX_ATTEMPTS; assert DEFAULT_AGENT_MAX_ATTEMPTS == 3"
```

Expected: exit code 0.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/config.py
git commit -m "feat(agent):固定任务重试上限"
git push origin HEAD
```

### Task 13: Specify model-adapter behavior

**Files:**
- Create: `test/test_agent_model_adapter.py`

**Interfaces:**
- Consumes: `AgentModelAdapter` and `AgentProtocolError`.
- Verifies: JSON plan parsing, allowed-tool validation, native tool-call selection, verification parsing, and fallback structured generation.

- [ ] **Step 1: Write fake model tests**

Create a fake model with recorded messages and these methods:

```python
def generate_structured_reply(self, messages, gen):
    return StructuredReply(content=self.structured_outputs.pop(0))

def generate_reply_with_tools(self, messages, gen, tools):
    return self.tool_outputs.pop(0)

def generate_reply(self, messages, gen):
    return self.text_outputs.pop(0)
```

Cover these cases:

- A valid plan returns one `PlanStep` and preserves `expected_result`.
- A plan naming an unknown tool raises `AgentProtocolError`.
- `choose_action()` passes only the current step's allowed schemas and returns the single native `ToolCall` as `ToolAction`.
- Zero or multiple tool calls raise `AgentProtocolError`.
- Verification requires all four boolean/string fields.
- When `generate_structured_reply` is absent, the adapter calls `generate_reply` and parses a JSON object from its full response.

- [ ] **Step 2: Run and confirm the adapter is missing**

Run:

```powershell
python -m unittest test.test_agent_model_adapter
```

Expected: FAIL because `guga.agent.model_adapter` does not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_model_adapter.py
git commit -m "to(agent):定义模型适配测试"
git push origin HEAD
```

### Task 14: Implement the model adapter

**Files:**
- Create: `guga/agent/model_adapter.py`

**Interfaces:**
- Consumes: existing model generation methods, `GenerationConfig`, and `ToolRegistry`.
- Produces: `create_plan()`, `choose_action()`, `verify_result()`, `render_final()`, and `AgentProtocolError`.

- [ ] **Step 1: Implement structured generation and validation**

Use this public shape:

```python
class AgentProtocolError(RuntimeError):
    pass


class AgentModelAdapter:
    def __init__(self, model, generation: GenerationConfig, system_prompt: str, tools: ToolRegistry) -> None:
        self.model = model
        self.generation = generation
        self.system_prompt = system_prompt
        self.tools = tools

    def create_plan(self, state: AgentTaskState) -> list[PlanStep]:
        payload = self._json_object(self._plan_messages(state))
        steps = payload.get("steps")
        return self._validate_steps(steps)

    def choose_action(self, state: AgentTaskState) -> ToolAction:
        step = state["plan"][state["current_step_index"]]
        allowed = set(step["allowed_tools"])
        response = self.model.generate_reply_with_tools(
            self._action_messages(state),
            self.generation,
            self.tools.openai_tools(names=allowed),
        )
        if len(response.tool_calls) != 1:
            raise AgentProtocolError("agent action must contain exactly one tool call")
        call = response.tool_calls[0]
        if call.name not in allowed:
            raise AgentProtocolError(f"tool outside approved step: {call.name}")
        return {
            "call_id": call.id,
            "tool_name": call.name,
            "arguments": call.arguments,
            "reason": response.content.strip(),
        }
```

Implement `_json_object(messages)` by preferring callable `generate_structured_reply`; otherwise call `generate_reply`. Parse the entire stripped response with `json.loads`, require a dictionary, and raise `AgentProtocolError` on parse or schema failure.

`_validate_steps()` must require a non-empty list, unique non-empty IDs, non-empty descriptions and expected results, and non-empty allowed-tool lists whose names are all present in `self.tools.names()`.

`verify_result()` must return exactly:

```python
{
    "matched": bool(payload["matched"]),
    "reason": required_text(payload, "reason"),
    "requires_replan": bool(payload["requires_replan"]),
    "blocked": bool(payload["blocked"]),
}
```

Reject contradictory verification where `matched` is true together with `requires_replan` or `blocked`.

`render_final()` uses ordinary `generate_reply()` with goal, status, plan, evidence, verification, and trace reference. It returns stripped text and raises `AgentProtocolError` for an empty answer.

- [ ] **Step 2: Run the adapter tests**

Run:

```powershell
python -m unittest test.test_agent_model_adapter
```

Expected: PASS.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/model_adapter.py
git commit -m "feat(agent):适配规划与工具决策"
git push origin HEAD
```

### Task 15: Specify the LangGraph state machine

**Files:**
- Create: `test/test_agent_task_graph.py`

**Interfaces:**
- Consumes: `build_agent_task_graph(adapter: AgentModelAdapter, tools: ToolRegistry, trace: ExecutionTraceStore, checkpointer)`.
- Verifies: interrupt payloads and all deterministic routes.

- [ ] **Step 1: Write deterministic fake-adapter and fake-tool tests**

Use `InMemorySaver`, a fake adapter with queued plans/actions/verifications, and a `ToolSpec` handler that increments a counter. Cover:

```text
start -> create_plan -> request_approval interrupt
approved -> choose_action -> execute_tool -> verify -> complete
rejected -> blocked without tool execution
mismatch attempts 1 and 2 -> retry
mismatch attempt 3 -> failed
requires_replan -> new plan revision -> new approval interrupt
blocked verification -> blocked without further retries
tool outside approved step -> replan without tool execution
```

For the first case, assert the initial result contains `__interrupt__`, its plan revision is 1, and the fake tool count is zero. Resume with:

```python
graph.invoke(
    Command(resume={"approved": True}),
    config={"configurable": {"thread_id": "task_1"}},
)
```

For retry failure, assert the handler count equals exactly 3 and final `status == "failed"`.

- [ ] **Step 2: Run and confirm the graph is missing**

Run:

```powershell
python -m unittest test.test_agent_task_graph
```

Expected: FAIL because `guga.agent.graph` does not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_task_graph.py
git commit -m "to(agent):定义任务状态机测试"
git push origin HEAD
```

### Task 16: Implement the LangGraph state machine

**Files:**
- Create: `guga/agent/graph.py`

**Interfaces:**
- Consumes: `AgentModelAdapter`, `ToolRegistry`, `ExecutionTraceStore`, and a LangGraph checkpointer.
- Produces: `build_agent_task_graph(adapter: AgentModelAdapter, tools: ToolRegistry, trace: ExecutionTraceStore, checkpointer)` returning a compiled graph.

- [ ] **Step 1: Implement nodes with no tool side effect before approval**

Implement nodes named exactly:

```text
create_plan
request_approval
choose_action
execute_tool
verify_result
advance_step
revise_plan
complete_task
fail_task
block_task
```

`request_approval` calls `interrupt()` before any execution and returns `approved_revision == plan_revision` only for an approved response. `execute_tool` asserts that equality again before calling `ToolRegistry.execute()`.

`choose_action` catches `AgentProtocolError`, stores a verification result with `requires_replan=True`, and routes to `revise_plan` without calling a tool. This includes a model selecting a tool outside the approved step.

Before `ToolRegistry.execute()`, append `tool_call_started`. After it returns, append `tool_call_finished`. Add one evidence row containing `step_id`, `attempt`, `action`, and `result`.

`verify_result` appends `step_verified` after receiving the adapter result.

- [ ] **Step 2: Implement deterministic routes**

Use these rules:

```python
def route_after_verification(state: AgentTaskState) -> str:
    result = state["verification"]
    if result["matched"]:
        return "advance_step"
    if result["blocked"]:
        return "block_task"
    if result["requires_replan"]:
        return "revise_plan"
    if state["attempt"] < state["max_attempts"]:
        return "choose_action"
    return "fail_task"
```

`advance_step` increments `current_step_index` and resets `attempt` to zero. Route to `complete_task` only when the index equals the plan length.

`revise_plan` calls the adapter with existing evidence, increments `plan_revision`, resets current step and attempt, appends `plan_revised`, and routes back to `request_approval`.

`complete_task`, `fail_task`, and `block_task` set the terminal status, ask the adapter for `final_response`, append one terminal trace event, and route to `END`.

- [ ] **Step 3: Run state-machine tests**

Run:

```powershell
python -m unittest test.test_agent_task_graph
```

Expected: PASS with exactly three calls in the retry-exhaustion test.

- [ ] **Step 4: Commit and push only this file**

```powershell
git add guga/agent/graph.py
git commit -m "feat(agent):实现LangGraph任务循环"
git push origin HEAD
```

### Task 17: Specify runner start, resume, and memory delivery

**Files:**
- Create: `test/test_agent_runner.py`

**Interfaces:**
- Consumes: `AgentTaskRunner`, `TaskRunResult`, `MemoryManager`, and an in-memory checkpointer.
- Verifies: start interruption, resume with the same task ID, terminal outcome delivery, and idempotent memory writes.

- [ ] **Step 1: Write runner integration tests**

Use the same fake adapter and fake tool pattern as the graph tests. Cover:

- `start("inspect README")` returns `status == "awaiting_approval"`, a non-empty task ID, and the plan interrupt payload.
- `resume(task_id, True)` uses the same checkpoint thread and returns `completed` after the fake verification passes.
- A terminal run writes one row to `task_outcomes.jsonl` containing the same `task_id` and a trace path ending in `trace.jsonl`.
- Calling the runner's terminal-result conversion twice does not create a second outcome row.
- `start("  ")` raises `ValueError` without invoking the graph.

- [ ] **Step 2: Run and confirm the runner is missing**

Run:

```powershell
python -m unittest test.test_agent_runner
```

Expected: FAIL because `guga.agent.runner` does not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_runner.py
git commit -m "to(agent):定义任务运行器测试"
git push origin HEAD
```

### Task 18: Implement the task runner and terminal memory bridge

**Files:**
- Create: `guga/agent/runner.py`

**Interfaces:**
- Consumes: `AgentModelAdapter`, `ToolRegistry`, `MemoryManager`, `ExecutionTraceStore`, `build_agent_task_graph()`, and `DEFAULT_AGENT_MAX_ATTEMPTS`.
- Produces: `TaskRunResult`, `AgentTaskRunner.start()`, and `AgentTaskRunner.resume()`.

- [ ] **Step 1: Define the runner result**

```python
@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    status: str
    approval_request: dict | None = None
    final_response: str = ""
```

- [ ] **Step 2: Implement construction, start, and resume**

Use this constructor and compile the graph once:

```python
def __init__(
    self,
    adapter: AgentModelAdapter,
    tools: ToolRegistry,
    trace: ExecutionTraceStore,
    memory_manager: MemoryManager,
    agent_id: str,
    checkpointer=None,
) -> None:
    self.trace = trace
    self.memory_manager = memory_manager
    self.agent_id = agent_id
    self.graph = build_agent_task_graph(
        adapter=adapter,
        tools=tools,
        trace=trace,
        checkpointer=checkpointer or InMemorySaver(),
    )
```

`start(user_request)` creates `task_<12 hex>` and invokes the graph with:

```python
{
    "task_id": task_id,
    "agent_id": agent_id,
    "user_request": user_request,
    "status": "planning",
    "plan_revision": 0,
    "approved_revision": 0,
    "current_step_index": 0,
    "attempt": 0,
    "max_attempts": DEFAULT_AGENT_MAX_ATTEMPTS,
    "evidence": [],
    "trace_ref": trace.trace_ref(task_id),
}
```

Reject an empty request before invoking the graph. `resume(task_id, approved)` uses `Command(resume={"approved": approved})` with the same thread ID.

Both methods call one private `_result(task_id, graph_result)` method. If `__interrupt__` is present, read the first interrupt object's `.value` and return `status="awaiting_approval"`. If status is terminal, create `TaskOutcome` using the graph state, collect unique tool names from evidence in first-use order, call `memory_manager.record_task_outcome()`, and return the final response.

- [ ] **Step 3: Run the runner integration tests**

```powershell
python -m unittest test.test_agent_runner
```

Expected: PASS, including one terminal outcome row and no duplicate after converting the terminal result twice.

- [ ] **Step 4: Commit and push only this file**

```powershell
git add guga/agent/runner.py
git commit -m "feat(agent):封装任务启动与恢复"
git push origin HEAD
```

### Task 19: Export the public agent API

**Files:**
- Create: `guga/agent/__init__.py`

**Interfaces:**
- Produces: stable imports for CLI construction.

- [ ] **Step 1: Export only public runtime types**

```python
from guga.agent.model_adapter import AgentModelAdapter, AgentProtocolError
from guga.agent.outcome import TaskOutcome
from guga.agent.runner import AgentTaskRunner, TaskRunResult
from guga.agent.trace import ExecutionTraceStore

__all__ = [
    "AgentModelAdapter",
    "AgentProtocolError",
    "AgentTaskRunner",
    "ExecutionTraceStore",
    "TaskOutcome",
    "TaskRunResult",
]
```

- [ ] **Step 2: Verify public imports**

Run:

```powershell
python -c "from guga.agent import AgentTaskRunner, ExecutionTraceStore, TaskOutcome"
```

Expected: exit code 0.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/__init__.py
git commit -m "feat(agent):导出任务运行接口"
git push origin HEAD
```

### Task 20: Specify CLI task interaction

**Files:**
- Create: `test/test_agent_cli.py`

**Interfaces:**
- Consumes: `AgentCliController.handle(text) -> AgentCliResult | None`.
- Verifies: explicit task routing and approval lifecycle.

- [ ] **Step 1: Write a fake-runner CLI test**

Define a fake runner returning queued `TaskRunResult` objects. Assert:

- `/task inspect README` calls `start("inspect README")` and renders the plan.
- An empty `/task` returns a usage message and does not call the runner.
- `/approve` without a pending task returns a clear error.
- After a task starts, `/approve` calls `resume(task_id, True)`.
- `/reject` calls `resume(task_id, False)`.
- A revised-plan approval result keeps the task pending.
- A terminal result clears the pending task.
- Ordinary text returns `None`, allowing `ChatSession` to handle it.

Use this result shape:

```python
@dataclass(frozen=True)
class AgentCliResult:
    handled: bool
    text: str
```

- [ ] **Step 2: Run and confirm the controller is missing**

Run:

```powershell
python -m unittest test.test_agent_cli
```

Expected: FAIL because `guga.agent.cli` does not exist.

- [ ] **Step 3: Commit and push only this red test file**

```powershell
git add test/test_agent_cli.py
git commit -m "to(agent):定义任务命令交互测试"
git push origin HEAD
```

### Task 21: Implement the CLI controller

**Files:**
- Create: `guga/agent/cli.py`

**Interfaces:**
- Consumes: `AgentTaskRunner` and `TaskRunResult`.
- Produces: `AgentCliController`, `AgentCliResult`, and deterministic plan rendering.

- [ ] **Step 1: Implement command routing**

`AgentCliController` owns `pending_task_id: str | None`. Implement:

```python
def handle(self, text: str) -> AgentCliResult | None:
    if text.startswith("/task"):
        request = text.removeprefix("/task").strip()
        if not request:
            return AgentCliResult(True, "用法: /task <任务描述>")
        return self._render(self.runner.start(request))
    if text == "/approve":
        if self.pending_task_id is None:
            return AgentCliResult(True, "当前没有等待批准的任务。")
        return self._render(self.runner.resume(self.pending_task_id, True))
    if text == "/reject":
        if self.pending_task_id is None:
            return AgentCliResult(True, "当前没有等待批准的任务。")
        return self._render(self.runner.resume(self.pending_task_id, False))
    return None
```

`_render()` stores the task ID while status is `awaiting_approval`, formats revision, steps, expected results, and allowed tools, and clears the pending ID for terminal states.

- [ ] **Step 2: Run CLI controller tests**

Run:

```powershell
python -m unittest test.test_agent_cli
```

Expected: PASS.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add guga/agent/cli.py
git commit -m "feat(agent):实现任务命令控制器"
git push origin HEAD
```

### Task 22: Wire the task runtime into the basic CLI

**Files:**
- Modify: `src/basic_cli_chat.py`

**Interfaces:**
- Consumes: all public agent APIs, `conversation_tool_registry()`, and `agent_runs_dir()`.
- Produces: user-visible `/task`, `/approve`, and `/reject` behavior.

- [ ] **Step 1: Construct the task dependencies once**

After creating `model` and `memory_manager`, construct:

```python
task_tools = default_tool_registry(PROJECT_ROOT)
trace_store = ExecutionTraceStore(
    agent_runs_dir(agent_identity.agent_id),
    agent_identity.agent_id,
)
adapter = AgentModelAdapter(
    model=model,
    generation=default_generation_config(),
    system_prompt=persona.system_prompt,
    tools=task_tools,
)
task_runner = AgentTaskRunner(
    adapter=adapter,
    tools=task_tools,
    trace=trace_store,
    memory_manager=memory_manager,
    agent_id=agent_identity.agent_id,
)
task_controller = AgentCliController(task_runner)
```

Construct `ChatSession` with `tool_registry=conversation_tool_registry()` so file and command tools cannot execute through ordinary chat.

- [ ] **Step 2: Route task commands before ChatSession**

Inside the input loop, after built-in exit/clear/RAG commands and before `reply_stream`, add:

```python
agent_result = task_controller.handle(user_text)
if agent_result is not None:
    print(f"小咕嘎> {agent_result.text}\n")
    continue
```

Update the startup command line to include `/task`, `/approve`, and `/reject`.

- [ ] **Step 3: Verify CLI help and imports**

Run:

```powershell
python -B src\basic_cli_chat.py --help
```

If the current script initializes the model before interpreting `--help`, instead run:

```powershell
python -m py_compile src\basic_cli_chat.py
```

Expected: the applicable command exits successfully.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m unittest test.test_agent_cli test.test_agent_task_graph test.test_tool_calling
```

Expected: PASS.

- [ ] **Step 5: Commit and push only this file**

```powershell
git add src/basic_cli_chat.py
git commit -m "feat(cli):接入智能体任务流程"
git push origin HEAD
```

### Task 23: Document the task workflow

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: task commands, approval semantics, retry behavior, trace location, memory location, and limitations.

- [ ] **Step 1: Add a focused intelligent-task section**

Document this exact user flow:

```text
/task 读取 README 并运行相关测试
/approve
/reject
```

State explicitly:

- `/task` generates and displays a plan without executing task tools.
- `/approve` permits continuous execution of the current plan.
- A revised plan pauses again for approval.
- A step fails after three mismatched executions.
- Traces are stored under `data/agent_runs/<agent_id>/<task_id>/trace.jsonl`.
- Terminal summaries are stored in `data/memory/agents/<agent_id>/task_outcomes.jsonl`.
- Trace rows are for developer debugging and are not injected into user semantic memory.
- Tool results retain the current output-length limits; complete raw-output archival is not yet implemented.

- [ ] **Step 2: Check Markdown and wording**

Run:

```powershell
git diff --check -- README.md
rg -n "(/task|/approve|trace.jsonl|task_outcomes.jsonl|三次)" README.md
```

Expected: no whitespace errors and all required concepts are present.

- [ ] **Step 3: Commit and push only this file**

```powershell
git add README.md
git commit -m "docs(agent):说明任务执行流程"
git push origin HEAD
```

### Task 24: Run final verification without changing files

**Files:**
- No file changes.

**Interfaces:**
- Verifies: the complete requested behavior and regression safety.

- [ ] **Step 1: Run every new focused test**

```powershell
python -m unittest test.test_agent_trace test.test_agent_task_memory test.test_agent_model_adapter test.test_agent_task_graph test.test_agent_runner test.test_agent_cli test.test_tool_calling
```

Expected: PASS.

- [ ] **Step 2: Run the full unit-test suite**

```powershell
python -m unittest discover -s test
```

Expected: PASS. If a documented pre-existing environment failure recurs, preserve its output and prove the new focused tests pass independently.

- [ ] **Step 3: Audit the behavioral invariants**

Run:

```powershell
rg -n "interrupt\(|max_attempts|allowed_tools|trace_ref|record_task_outcome" guga test src
git diff main...HEAD --check
git status --short
git log --oneline main..HEAD
```

Expected:

- approval uses a LangGraph interrupt;
- the retry cap is represented in runtime state and routing;
- tool scope is checked before execution;
- trace references reach `TaskOutcome`;
- each implementation commit names exactly one changed file;
- the worktree is clean.

- [ ] **Step 4: Inspect every commit for the one-file invariant**

```powershell
git log --format="%H" main..HEAD | ForEach-Object {
    git diff-tree --no-commit-id --name-only -r $_
}
```

Expected: each commit prints exactly one path.

- [ ] **Step 5: Push the verified branch**

```powershell
git push origin HEAD
```

Expected: the remote feature branch contains all verified one-file commits.
