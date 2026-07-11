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

from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.memory.semantic_events import SemanticEventStore
from guga.memory.summarizer import MemoryBankSummarizer
from guga.models.factory import create_chat_model


_REFLECTION_KEYS = {"appraisal", "felt_response", "relational_intent", "interpretation_confidence"}


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


def validate_time_event(event: dict, *, expect_unknown_end: bool) -> None:
    start_at = str(event.get("start_at") or "")
    if not start_at:
        raise AssertionError(f"missing start_at: {event}")
    start = _parse_iso(start_at, "start_at")
    end_at = event.get("end_at")
    end_unknown = bool(event.get("end_unknown"))
    if expect_unknown_end:
        if not end_unknown or end_at is not None:
            raise AssertionError(f"expected unknown end: {event}")
        return
    if end_unknown or not end_at:
        raise AssertionError(f"expected resolved same-day end: {event}")
    end = _parse_iso(str(end_at), "end_at")
    if end != start:
        raise AssertionError(f"single-day event must use identical start/end: {event}")


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
    validate_time_event(operation, expect_unknown_end=False)
    if not str(operation["start_at"]).startswith("2026-07-14T15:00:00+08:00"):
        raise AssertionError(f"unexpected relative-time resolution: {operation}")
    return operation


def run_multiday_unknown_end(summarizer: MemoryBankSummarizer) -> dict:
    operation = _single_operation(
        summarizer,
        _low_level_packet("我下周开始出差，结束日期还不确定。", "2026-07-09T09:30:00+08:00"),
    )
    validate_time_event(operation, expect_unknown_end=True)
    if not str(operation["start_at"]).startswith("2026-07-13T00:00:00+08:00"):
        raise AssertionError(f"unexpected multi-day start: {operation}")
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
                    "end_at": "2026-07-12T09:00:00+08:00",
                    "end_unknown": False,
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
        validate_time_event(operation, expect_unknown_end=False)
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
                    "end_at": "2026-07-12T09:00:00+08:00",
                    "end_unknown": False,
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
        validate_time_event(persisted, expect_unknown_end=False)
        scenario_prompts = model.prompts[prompt_start:]
        low_prompts = [prompt for prompt in scenario_prompts if "Low-level memory consolidation" in prompt]
        high_prompts = [prompt for prompt in scenario_prompts if "High-level memory consolidation" in prompt]
        if len(low_prompts) != 1 or len(high_prompts) != 1:
            raise AssertionError(f"unexpected API call count: low={len(low_prompts)}, high={len(high_prompts)}")
        high_packet = _packet_from_prompt(high_prompts[0])
        high_event_ids = {str(event.get("id", "")) for event in high_packet.get("semantic_events", [])}
        if persisted["id"] not in high_event_ids:
            raise AssertionError("Stage 2 did not receive the persisted Stage 1 event")
        if user_id not in persisted.get("source_message_ids", []):
            raise AssertionError("persisted event lost its raw user-message evidence")
        return persisted


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live API validation for LLM-dependent Guga memory behavior.")
    parser.add_argument(
        "--scenario",
        choices=("all", "same-day", "multi-day", "reflection", "replace", "cancel", "two-stage"),
        default="all",
    )
    args = parser.parse_args()
    load_env_file()
    model_id = os.environ.get("Guga_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError("Guga_MODEL_ID is required for live API validation")
    base_model = create_chat_model(model_id=model_id)
    model = RecordingModel(base_model)
    summarizer = MemoryBankSummarizer(model=model, use_llm=True)
    selected = ("same-day", "multi-day", "reflection", "replace", "cancel", "two-stage") if args.scenario == "all" else (args.scenario,)
    results: dict[str, dict] = {}
    try:
        for scenario in selected:
            if scenario == "same-day":
                results[scenario] = run_same_day_time(summarizer)
            elif scenario == "multi-day":
                results[scenario] = run_multiday_unknown_end(summarizer)
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
