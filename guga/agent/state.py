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
    verification_method: str
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
    session_id: str
    user_request: str
    task_context: str
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
    recovery_required: bool
    recovery_execution_id: str
