from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.chat.session import ChatSession
from guga.config import default_generation_config
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.memory.semantic_events import SemanticEventStore
from guga.memory.summarizer import MemoryBankSummarizer
from guga.models.factory import create_chat_model
from guga.utils.debug_reporter import FileDebugSink


_REFLECTION_KEYS = {"appraisal", "felt_response"}


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise RuntimeError(f"missing API configuration: {env_path}")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ["Guga_MEMORY_USE_LLM_SUMMARY"] = "1"


class RecordingModel:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def generate_reply(self, messages, gen):
        self.prompts.append(str(messages[-1]["content"]))
        response = self.delegate.generate_reply(messages, gen)
        self.responses.append(str(response))
        return response

    def generate_json_reply(self, messages, gen):
        self.prompts.append(str(messages[-1]["content"]))
        response = self.delegate.generate_json_reply(messages, gen)
        self.responses.append(str(response))
        return response

    def generate_structured_reply(self, messages, gen):
        self.prompts.append(str(messages[-1]["content"]))
        response = self.delegate.generate_structured_reply(messages, gen)
        self.responses.append(str(response.content))
        return response


def validate_time_event(event: dict) -> None:
    start = _parse_iso(str(event.get("start_at") or ""), "start_at")
    end = _parse_iso(str(event.get("end_at") or ""), "end_at")
    if start >= end:
        raise AssertionError(f"event interval must be non-empty and end-exclusive: {event}")
    if event.get("end_unknown") is not False:
        raise AssertionError(f"new event must persist end_unknown=false: {event}")


def _parse_iso(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AssertionError(f"invalid {field} ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AssertionError(f"{field} must include timezone: {value!r}")
    return parsed


def _low_level_packet(user_text: str, created_at: str) -> dict:
    return {
        "new_turns": [
            {
                "user_message_id": "msg_live_user",
                "assistant_message_id": "msg_live_assistant",
                "created_at": created_at,
                "user_text": user_text,
                "assistant_text": "好的，我记下了。",
            }
        ],
        "recent_active_events": [],
        "relevant_active_events": [],
        "retrieved_context": [],
    }


def _single_operation(summarizer: MemoryBankSummarizer, packet: dict, *, reflection: bool = False) -> dict:
    result = summarizer.consolidate_low_level_memory(packet, include_guga_reflection=reflection)
    operations = result.get("semantic_event_operations", [])
    if len(operations) != 1:
        raise AssertionError(f"expected one semantic event operation, got: {operations}")
    operation = dict(operations[0])
    if operation.get("operation") in {"create", "update", "replace"} and operation.get("subject") != "user":
        raise AssertionError(f"semantic event subject must be user: {operation}")
    return operation


def run_same_day_time(summarizer: MemoryBankSummarizer) -> dict:
    operation = _single_operation(
        summarizer,
        _low_level_packet("我下周二下午三点去看牙。", "2026-07-09T09:30:00+08:00"),
    )
    validate_time_event(operation)
    if not str(operation["start_at"]).startswith("2026-07-14T15:00:00+08:00"):
        raise AssertionError(f"unexpected relative-time resolution: {operation}")
    return operation


def run_reflection_schema(summarizer: MemoryBankSummarizer) -> dict:
    operation = _single_operation(
        summarizer,
        _low_level_packet("我下周二下午三点去看牙。", "2026-07-09T09:30:00+08:00"),
        reflection=True,
    )
    reflection = operation.get("guga_reflection")
    if not isinstance(reflection, dict) or set(reflection) != _REFLECTION_KEYS:
        raise AssertionError(f"invalid guga_reflection schema: {operation}")
    return operation


def run_replace_lifecycle(summarizer: MemoryBankSummarizer) -> dict:
    with tempfile.TemporaryDirectory(prefix="guga-live-replace-") as tmp:
        store = SemanticEventStore(Path(tmp) / "semantic_events.jsonl")
        old_id = store.apply_operations(
            operations=[
                {
                    "operation": "create",
                    "event_kind": "appointment",
                    "subject": "user",
                    "entity": "看牙",
                    "description": "用户周日看牙。",
                    "time_expression": "周日",
                    "start_at": "2026-07-12T09:00:00+08:00",
                    "end_at": "2026-07-12T10:00:00+08:00",
                    "time_granularity": "hour",
                }
            ],
            session_id="seed",
            include_guga_reflection=False,
        ).created_event_ids[0]
        packet = _low_level_packet("周日的看牙改到下周二下午三点。", "2026-07-09T09:30:00+08:00")
        packet["recent_active_events"] = store.load_active()
        packet["relevant_active_events"] = store.load_active()
        operation = _single_operation(summarizer, packet)
        if operation.get("operation") not in {"update", "replace"} or operation.get("target_event_id") != old_id:
            raise AssertionError(f"expected mutation of {old_id}, got: {operation}")
        validate_time_event(operation)
        store.apply_operations(operations=[operation], session_id="replace", include_guga_reflection=False)
        rows = {row["id"]: row for row in store.load_all()}
        active = [row for row in rows.values() if row.get("status") == "active"]
        if len(active) != 1 or active[0]["start_at"] != operation["start_at"]:
            raise AssertionError(f"invalid mutation lifecycle: {rows}")
        return operation


def run_cancel_lifecycle(summarizer: MemoryBankSummarizer) -> dict:
    with tempfile.TemporaryDirectory(prefix="guga-live-cancel-") as tmp:
        store = SemanticEventStore(Path(tmp) / "semantic_events.jsonl")
        old_id = store.apply_operations(
            operations=[
                {
                    "operation": "create",
                    "event_kind": "appointment",
                    "subject": "user",
                    "entity": "看牙",
                    "description": "用户周日看牙。",
                    "start_at": "2026-07-12T09:00:00+08:00",
                    "end_at": "2026-07-12T10:00:00+08:00",
                    "time_granularity": "hour",
                }
            ],
            session_id="seed",
            include_guga_reflection=False,
        ).created_event_ids[0]
        packet = _low_level_packet("周日的看牙取消了，不去了。", "2026-07-09T09:30:00+08:00")
        packet["recent_active_events"] = store.load_active()
        packet["relevant_active_events"] = store.load_active()
        operation = _single_operation(summarizer, packet)
        if operation.get("operation") != "cancel" or operation.get("target_event_id") != old_id:
            raise AssertionError(f"expected cancellation of {old_id}, got: {operation}")
        store.apply_operations(operations=[operation], session_id="cancel", include_guga_reflection=False)
        cancelled = {row["id"]: row for row in store.load_all()}[old_id]
        if cancelled["status"] != "inactive" or cancelled["inactive_reason"] != "cancelled":
            raise AssertionError(f"invalid cancellation lifecycle: {cancelled}")
        return operation


def _packet_from_prompt(prompt: str) -> dict:
    prefix = "Input packet:\n"
    if prefix not in prompt:
        raise AssertionError("missing consolidation input packet")
    packet, _ = json.JSONDecoder().raw_decode(prompt.split(prefix, 1)[1])
    return packet


def run_two_stage_persistence(model: RecordingModel) -> dict:
    with tempfile.TemporaryDirectory(prefix="guga-live-pipeline-") as tmp:
        prompt_start = len(model.prompts)
        manager = MemoryManager(
            memory_root=Path(tmp),
            model=model,
            enable_semantic=False,
            consolidation_config=MemoryConsolidationConfig(batch_turns=1, include_guga_reflection=False),
        )
        user_id = manager.record_user_message(
            "live_pipeline",
            "我下周二下午三点去看牙。",
            created_at="2026-07-09T09:30:00+08:00",
        )
        manager.record_assistant_message("live_pipeline", "好的，我记下了。")
        manager.finalize_turn("live_pipeline")
        events = manager.semantic_event_store.load_active()
        if len(events) != 1:
            raise AssertionError(f"expected one persisted event, got: {events}")
        persisted = events[0]
        validate_time_event(persisted)
        scenario_prompts = model.prompts[prompt_start:]
        low_prompts = [prompt for prompt in scenario_prompts if "Low-level memory consolidation" in prompt]
        high_prompts = [prompt for prompt in scenario_prompts if "High-level memory consolidation" in prompt]
        if not low_prompts or len(high_prompts) != 1:
            raise AssertionError(f"unexpected API call count: low={len(low_prompts)}, high={len(high_prompts)}")
        high_packet = _packet_from_prompt(high_prompts[0])
        high_event_ids = {str(event.get("id", "")) for event in high_packet.get("semantic_events", [])}
        if persisted["id"] not in high_event_ids:
            raise AssertionError("Stage 2 did not receive the persisted Stage 1 event")
        if user_id not in persisted.get("source_message_ids", []):
            raise AssertionError("persisted event lost its raw user-message evidence")
        return persisted


def _session_rows(memory_root: Path, session_id: str) -> list[dict]:
    path = memory_root / "sessions" / f"{session_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has_temporal_context(rows: list[dict]) -> bool:
    user = next((row for row in rows if row.get("role") == "user"), {})
    turn = ((user.get("metadata") or {}).get("turn") or {})
    return bool(turn.get("temporal_context"))


def run_temporal_e2e() -> dict:
    load_env_file()
    model_id = os.environ.get("Guga_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError("Guga_MODEL_ID is required for live API validation")
    model = create_chat_model(model_id=model_id)

    with (
        tempfile.TemporaryDirectory(prefix="guga-live-temporal-memory-") as memory_tmp,
        tempfile.TemporaryDirectory(prefix="guga-live-temporal-debug-") as debug_tmp,
    ):
        memory_root = Path(memory_tmp)
        debug_root = Path(debug_tmp)
        debug_lines: list[str] = []
        file_sink = FileDebugSink(debug_root)

        def debug_sink(message: str) -> None:
            debug_lines.append(message)
            file_sink(message)

        first_manager = MemoryManager(
            memory_root=memory_root,
            model=model,
            enable_semantic=True,
            debug=True,
            debug_sink=debug_sink,
            consolidation_config=MemoryConsolidationConfig(
                batch_turns=10,
                include_guga_reflection=False,
            ),
        )
        first_session = ChatSession(
            model=model,
            system_prompt="你是一个准确使用记忆工具的助手。",
            generation=default_generation_config(),
            memory_manager=first_manager,
            session_id="temporal_e2e_first",
            debug=True,
            debug_sink=debug_sink,
        )
        first_session.reply(
            "我8月20号有个面试",
            created_at="2026-08-18T10:00:00+08:00",
        )
        settled = first_session.settle_memory_for_shutdown()
        if settled.get("status") != "complete":
            raise AssertionError(f"exit consolidation did not complete: {settled}")

        events = first_manager.semantic_event_store.load_active()
        interview_events = [
            event
            for event in events
            if "面试" in f"{event.get('entity', '')} {event.get('description', '')}"
        ]
        if len(events) != 1 or len(interview_events) != 1:
            raise AssertionError(f"expected exactly one active interview event: {events}")
        event = interview_events[0]
        validate_time_event(event)
        if event["start_at"] != "2026-08-20T00:00:00+08:00" or event["end_at"] != "2026-08-21T00:00:00+08:00":
            raise AssertionError(f"unexpected interview interval: {event}")

        second_manager = MemoryManager(
            memory_root=memory_root,
            model=model,
            enable_semantic=True,
            debug=True,
            debug_sink=debug_sink,
            consolidation_config=MemoryConsolidationConfig(
                batch_turns=10,
                include_guga_reflection=False,
            ),
        )
        second_session = ChatSession(
            model=model,
            system_prompt="你是一个准确使用记忆工具的助手。",
            generation=default_generation_config(),
            memory_manager=second_manager,
            session_id="temporal_e2e_second",
            debug=True,
            debug_sink=debug_sink,
        )
        answer = second_session.reply(
            "8月20号有什么事情",
            finalize_memory=False,
            created_at="2026-08-19T10:00:00+08:00",
        )

        first_rows = _session_rows(memory_root, first_session.session_id)
        second_rows = _session_rows(memory_root, second_session.session_id)
        second_user = next(row for row in second_rows if row.get("role") == "user")
        second_turn = ((second_user.get("metadata") or {}).get("turn") or {})
        time_tools = [
            item
            for item in (second_turn.get("tool_interactions") or [])
            if item.get("tool") == "memory_time_search"
        ]
        returned_memory_ids = [
            str(memory_id)
            for item in time_tools
            for memory_id in (item.get("returned_memory_ids") or [])
        ]
        debug_text = "\n".join(debug_lines)
        debug_has_time_search = "time_search " in debug_text and '"status":"success"' in debug_text
        debug_has_exit_consolidation = (
            "consolidation_trigger" in debug_text and '"trigger_reason":"exit"' in debug_text
        )
        result = {
            "event_id": str(event["id"]),
            "event_start_at": str(event["start_at"]),
            "event_end_at": str(event["end_at"]),
            "retrieved_memory_ids": returned_memory_ids,
            "first_turn_has_temporal_context": _has_temporal_context(first_rows),
            "second_turn_has_temporal_context": _has_temporal_context(second_rows),
            "debug_has_time_search": debug_has_time_search,
            "debug_has_exit_consolidation": debug_has_exit_consolidation,
            "answer": str(answer),
        }
        if event["id"] not in returned_memory_ids:
            raise AssertionError(f"time search did not return the interview event: {result}")
        if not all(
            result[key]
            for key in (
                "first_turn_has_temporal_context",
                "second_turn_has_temporal_context",
                "debug_has_time_search",
                "debug_has_exit_consolidation",
            )
        ):
            raise AssertionError(f"temporal trace/debug acceptance failed: {result}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live API validation for LLM-dependent Guga memory behavior.")
    parser.add_argument(
        "--scenario",
        choices=("all", "same-day", "reflection", "replace", "cancel", "two-stage", "temporal-e2e"),
        default="all",
    )
    args = parser.parse_args()
    if args.scenario == "temporal-e2e":
        print(json.dumps(run_temporal_e2e(), ensure_ascii=False, indent=2))
        return 0
    load_env_file()
    model_id = os.environ.get("Guga_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError("Guga_MODEL_ID is required for live API validation")
    base_model = create_chat_model(model_id=model_id)
    model = RecordingModel(base_model)
    summarizer = MemoryBankSummarizer(model=model, use_llm=True)
    selected = ("same-day", "reflection", "replace", "cancel", "two-stage") if args.scenario == "all" else (args.scenario,)
    results: dict[str, dict] = {}
    try:
        for scenario in selected:
            if scenario == "same-day":
                results[scenario] = run_same_day_time(summarizer)
            elif scenario == "reflection":
                results[scenario] = run_reflection_schema(summarizer)
            elif scenario == "replace":
                results[scenario] = run_replace_lifecycle(summarizer)
            elif scenario == "cancel":
                results[scenario] = run_cancel_lifecycle(summarizer)
            else:
                results[scenario] = run_two_stage_persistence(model)
    except Exception as exc:
        diagnostics = [
            {"length": len(response), "excerpt": response[:1000]}
            for response in model.responses[-2:]
        ]
        print(json.dumps({"error": str(exc), "recent_model_responses": diagnostics}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
