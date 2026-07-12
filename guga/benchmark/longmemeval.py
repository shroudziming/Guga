from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from guga.chat import ChatSession
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.manager import MemoryManager
from guga.benchmark.workspace import BenchmarkWorkspace, safe_case_id
from guga.types import GenerationConfig
from guga.utils.debug_reporter import FileDebugSink


LONGMEMEVAL_SYSTEM_PROMPT = (
    "You are an evaluation assistant for the LongMemEval benchmark. "
    "Answer the user question using only the provided conversation memory and retrieved evidence. "
    "Give a concise answer. If the evidence is insufficient or the question has a false premise, "
    "state that the answer is unknown instead of guessing. Do not role-play a daily companion persona."
)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class LongMemEvalMessage:
    role: str
    content: str
    created_at: str = ""


@dataclass(frozen=True)
class LongMemEvalCase:
    case_id: str
    question: str
    answer: str
    question_type: str
    sessions: list[list[LongMemEvalMessage]]
    session_ids: list[str]
    session_dates: list[str]
    question_date: str
    answer_session_ids: list[str]
    raw: dict[str, Any]


def load_longmemeval_cases(path: Path) -> list[LongMemEvalCase]:
    rows = _load_rows(path)
    cases: list[LongMemEvalCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = _first_text(row, "question", "query", "input")
        if not question:
            continue
        case_id = _first_text(row, "question_id", "id", "qid", "case_id") or f"case_{index}"
        sessions = _extract_sessions(row)
        session_ids = _string_list(row.get("haystack_session_ids"))
        session_dates = _string_list(row.get("haystack_dates"))
        if session_dates:
            sessions = [
                [
                    LongMemEvalMessage(
                        message.role,
                        message.content,
                        message.created_at or (session_dates[index] if index < len(session_dates) else ""),
                    )
                    for message in session
                ]
                for index, session in enumerate(sessions)
            ]
        cases.append(
            LongMemEvalCase(
                case_id=safe_case_id(case_id),
                question=question,
                answer=_first_text(row, "answer", "reference_answer", "gold", "target"),
                question_type=_first_text(row, "question_type", "category", "type"),
                sessions=sessions,
                session_ids=session_ids,
                session_dates=session_dates,
                question_date=_first_text(row, "question_date"),
                answer_session_ids=_string_list(row.get("answer_session_ids")),
                raw=row,
            )
        )
    return cases


def ingest_longmemeval_case(
    case: LongMemEvalCase,
    manager: MemoryManager,
    *,
    start_session_index: int = 0,
    initial_stats: dict[str, int] | None = None,
    on_session_completed: Callable[[int, dict[str, int]], None] | None = None,
    on_message_processed: Callable[[int, dict[str, int]], None] | None = None,
) -> dict[str, int]:
    stats = _initial_ingest_stats("raw", initial_stats)
    for session_index, session in enumerate(case.sessions[start_session_index:], start=start_session_index):
        session_id = _case_session_id(case, session_index)
        for message in session:
            role = _normalize_role(message.role)
            if role == "assistant":
                manager.record_assistant_message(
                    session_id=session_id,
                    text=message.content,
                    source="benchmark:longmemeval",
                    created_at=message.created_at or None,
                    store_as_memory=True,
                )
            else:
                manager.record_user_message(
                    session_id=session_id,
                    text=message.content,
                    source="benchmark:longmemeval",
                    created_at=message.created_at or None,
                )
            stats["messages"] += 1
            if on_message_processed is not None:
                on_message_processed(session_index + 1, dict(stats))
        stats["sessions"] += 1
        if on_session_completed is not None:
            on_session_completed(session_index + 1, dict(stats))
    return stats


def ingest_longmemeval_case_replay(
    case: LongMemEvalCase,
    manager: MemoryManager,
    *,
    start_session_index: int = 0,
    initial_stats: dict[str, int] | None = None,
    on_session_completed: Callable[[int, dict[str, int]], None] | None = None,
    on_message_processed: Callable[[int, dict[str, int]], None] | None = None,
) -> dict[str, int]:
    stats = _initial_ingest_stats("replay", initial_stats)
    for session_index, session in enumerate(case.sessions[start_session_index:], start=start_session_index):
        session_id = _case_session_id(case, session_index)
        has_open_user_turn = False
        for message in session:
            role = _normalize_role(message.role)
            if role == "assistant":
                manager.record_assistant_message(
                    session_id=session_id,
                    text=message.content,
                    source="benchmark:longmemeval",
                    created_at=message.created_at or None,
                )
                if has_open_user_turn:
                    manager.finalize_turn(session_id)
                    stats["finalized_turns"] += 1
                    stats["completed_turns"] += 1
                    has_open_user_turn = False
            else:
                if has_open_user_turn:
                    manager.finalize_turn(session_id)
                    stats["finalized_turns"] += 1
                    stats["completed_turns"] += 1
                manager.record_user_message(
                    session_id=session_id,
                    text=message.content,
                    source="benchmark:longmemeval",
                    created_at=message.created_at or None,
                )
                has_open_user_turn = True
            stats["messages"] += 1
            if on_message_processed is not None:
                on_message_processed(session_index + 1, dict(stats))
        if has_open_user_turn:
            manager.finalize_turn(session_id)
            stats["finalized_turns"] += 1
            stats["completed_turns"] += 1
        flush_stats = manager.flush_session_memory(session_id)
        for key in ("consolidation_batches", "low_level_updates", "high_level_updates", "high_level_noops"):
            stats[key] += int(flush_stats.get(key, 0))
        stats["sessions"] += 1
        if on_session_completed is not None:
            on_session_completed(session_index + 1, dict(stats))
    return stats


def run_longmemeval_case(
    case: LongMemEvalCase,
    model,
    workspace: BenchmarkWorkspace,
    generation: GenerationConfig,
    debug: bool = False,
    enable_semantic: bool = True,
    ingest_mode: str = "raw",
    replay_finalize_every: int = 10,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    total_started = perf_counter()
    memory_root = workspace.case_memory_root(case.case_id)
    debug_sink = FileDebugSink(workspace.case_debug_reports_dir(case.case_id)) if debug else None
    manager = MemoryManager(
        memory_root=memory_root,
        model=model,
        debug=debug,
        debug_sink=debug_sink,
        documents_dir=workspace.documents_dir,
        enable_semantic=enable_semantic,
        consolidation_config=MemoryConsolidationConfig(
            batch_turns=replay_finalize_every,
            include_guga_reflection=False,
            enable_archival_updates=True,
            enable_user_model_updates=False,
        )
        if ingest_mode == "replay"
        else None,
    )
    checkpoint_file = workspace.case_checkpoint_file(case.case_id)
    checkpoint = _load_checkpoint(checkpoint_file) if resume else None
    start_session_index = int(checkpoint.get("next_session_index", 0)) if checkpoint else 0
    initial_stats = checkpoint.get("ingest", {}) if checkpoint else None
    if start_session_index:
        _emit_progress(
            progress,
            {
                "phase": "case_resumed",
                "case_id": case.case_id,
                "next_session_index": start_session_index,
                "session_total": len(case.sessions),
            },
        )

    def checkpoint_session(next_session_index: int, stats: dict[str, int]) -> None:
        _write_checkpoint(
            checkpoint_file,
            {
                "schema_version": 1,
                "case_id": case.case_id,
                "ingest_mode": ingest_mode,
                "next_session_index": next_session_index,
                "ingest": stats,
            },
        )
        _emit_progress(
            progress,
            {
                "phase": "ingest_session_completed",
                "case_id": case.case_id,
                "session_index": next_session_index,
                "session_total": len(case.sessions),
                "ingest": stats,
            },
        )

    def report_message(session_index: int, stats: dict[str, int]) -> None:
        _emit_progress(
            progress,
            {
                "phase": "ingest_message_progress",
                "case_id": case.case_id,
                "session_index": session_index,
                "session_total": len(case.sessions),
                "ingest": stats,
            },
        )

    ingest_started = perf_counter()
    if ingest_mode == "replay":
        ingest_stats = ingest_longmemeval_case_replay(
            case,
            manager,
            start_session_index=start_session_index,
            initial_stats=initial_stats,
            on_session_completed=checkpoint_session,
            on_message_processed=report_message,
        )
    elif ingest_mode == "raw":
        ingest_stats = ingest_longmemeval_case(
            case,
            manager,
            start_session_index=start_session_index,
            initial_stats=initial_stats,
            on_session_completed=checkpoint_session,
            on_message_processed=report_message,
        )
    else:
        raise ValueError(f"Unsupported LongMemEval ingest_mode: {ingest_mode}")
    ingest_latency_ms = int((perf_counter() - ingest_started) * 1000)
    if manager.rag_pipeline is not None:
        _write_checkpoint(
            checkpoint_file,
            {
                "schema_version": 1,
                "case_id": case.case_id,
                "ingest_mode": ingest_mode,
                "next_session_index": len(case.sessions),
                "ingest": ingest_stats,
                "phase": "rebuild_indexes",
            },
        )
        _emit_progress(progress, {"phase": "rebuild_indexes", "case_id": case.case_id})
        manager.rebuild_rag_indexes(session_id=f"{case.case_id}_benchmark")

    session = ChatSession(
        model=model,
        system_prompt=LONGMEMEVAL_SYSTEM_PROMPT,
        generation=generation,
        memory_manager=manager,
        session_id=f"{case.case_id}_question",
        debug=debug,
        debug_sink=debug_sink,
    )
    answer_started = perf_counter()
    _write_checkpoint(
        checkpoint_file,
        {
            "schema_version": 1,
            "case_id": case.case_id,
            "ingest_mode": ingest_mode,
            "next_session_index": len(case.sessions),
            "ingest": ingest_stats,
            "phase": "answering",
        },
    )
    _emit_progress(progress, {"phase": "answering", "case_id": case.case_id})
    prediction = session.reply(case.question, finalize_memory=False, created_at=case.question_date or None).strip()
    answer_latency_ms = int((perf_counter() - answer_started) * 1000)
    total_latency_ms = int((perf_counter() - total_started) * 1000)

    result = {
        "case_id": case.case_id,
        "question_type": case.question_type,
        "question": case.question,
        "answer": case.answer,
        "question_date": case.question_date,
        "answer_session_ids": case.answer_session_ids,
        "prediction": prediction,
        "ingest": ingest_stats,
        "ingest_mode": ingest_mode,
        "memory_root": str(memory_root),
        "finalize_skipped": True,
        "timing_ms": {
            "ingest": ingest_latency_ms,
            "answer": answer_latency_ms,
            "total": total_latency_ms,
        },
    }
    _append_result(workspace.results_file, result)
    checkpoint_file.unlink(missing_ok=True)
    _emit_progress(
        progress,
        {
            "phase": "case_completed",
            "case_id": case.case_id,
            "timing_ms": result["timing_ms"],
        },
    )
    return result


def run_longmemeval_benchmark(
    dataset_path: Path,
    model,
    workspace: BenchmarkWorkspace,
    generation: GenerationConfig,
    limit: int | None = None,
    debug: bool = False,
    enable_semantic: bool = True,
    ingest_mode: str = "raw",
    replay_finalize_every: int = 10,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    cases = load_longmemeval_cases(dataset_path)
    if limit is not None:
        cases = cases[: max(0, limit)]

    completed_results = _load_completed_results(workspace.results_file) if resume else {}
    results: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        case_progress = lambda event, case_index=case_index: _record_progress(
            workspace,
            {
                **event,
                "case_index": case_index,
                "case_total": len(cases),
            },
            progress,
        )
        completed = completed_results.get(case.case_id)
        if completed is not None:
            workspace.case_checkpoint_file(case.case_id).unlink(missing_ok=True)
            _emit_progress(
                case_progress,
                {"phase": "case_skipped_completed", "case_id": case.case_id},
            )
            results.append(completed)
            continue
        _emit_progress(case_progress, {"phase": "case_started", "case_id": case.case_id})
        results.append(
            run_longmemeval_case(
                case=case,
                model=model,
                workspace=workspace,
                generation=generation,
                debug=debug,
                enable_semantic=enable_semantic,
                ingest_mode=ingest_mode,
                replay_finalize_every=replay_finalize_every,
                resume=resume,
                progress=case_progress,
            )
        )
    return results


def _load_rows(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "cases", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _append_result(path: Path, result: dict[str, Any]) -> None:
    _append_json_line(path, result)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _initial_ingest_stats(mode: str, initial_stats: dict[str, int] | None) -> dict[str, int]:
    keys = ["sessions", "messages"]
    if mode == "replay":
        keys.extend(
            [
                "finalized_turns",
                "completed_turns",
                "consolidation_batches",
                "low_level_updates",
                "high_level_updates",
                "high_level_noops",
            ]
        )
    initial_stats = initial_stats or {}
    return {key: int(initial_stats.get(key, 0)) for key in keys}


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LongMemEval checkpoint: {path}")
    return payload


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_completed_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = payload.get("case_id") if isinstance(payload, dict) else None
        if isinstance(case_id, str) and case_id:
            completed[case_id] = payload
    return completed


def _emit_progress(progress: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress is not None:
        progress(event)


def _record_progress(
    workspace: BenchmarkWorkspace,
    event: dict[str, Any],
    callback: ProgressCallback | None,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": workspace.run_id,
        **event,
    }
    _append_json_line(workspace.progress_file, payload)
    _emit_progress(callback, payload)


def _extract_sessions(row: dict[str, Any]) -> list[list[LongMemEvalMessage]]:
    for key in ("sessions", "haystack_sessions", "conversations"):
        value = row.get(key)
        sessions = _normalize_sessions(value)
        if sessions:
            return sessions

    for key in ("conversation", "history", "messages", "context"):
        value = row.get(key)
        messages = _normalize_messages(value)
        if messages:
            return [messages]

    return []


def _normalize_sessions(value: Any) -> list[list[LongMemEvalMessage]]:
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, dict) and _message_content(item) for item in value):
        return [_normalize_messages(value)]

    sessions: list[list[LongMemEvalMessage]] = []
    for item in value:
        messages = _normalize_session(item)
        if messages:
            sessions.append(messages)
    return sessions


def _normalize_session(value: Any) -> list[LongMemEvalMessage]:
    if isinstance(value, dict):
        session_time = _first_text(value, "created_at", "timestamp", "time", "date")
        for key in ("messages", "conversation", "history", "turns"):
            messages = _normalize_messages(value.get(key))
            if messages:
                if not session_time:
                    return messages
                return [
                    LongMemEvalMessage(role=message.role, content=message.content, created_at=message.created_at or session_time)
                    for message in messages
                ]
    return _normalize_messages(value)


def _normalize_messages(value: Any) -> list[LongMemEvalMessage]:
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return []

    messages: list[LongMemEvalMessage] = []
    for item in value:
        if isinstance(item, str):
            content = item.strip()
            if content:
                messages.append(LongMemEvalMessage(role="user", content=content))
            continue
        if not isinstance(item, dict):
            continue
        content = _message_content(item)
        if not content:
            continue
        role = _first_text(item, "role", "speaker", "from") or "user"
        created_at = _first_text(item, "created_at", "timestamp", "time", "date")
        messages.append(LongMemEvalMessage(role=role, content=content, created_at=created_at))
    return messages


def _message_content(item: dict[str, Any]) -> str:
    return _first_text(item, "content", "text", "message", "utterance", "value")


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized in {"assistant", "ai", "bot", "gpt"}:
        return "assistant"
    return "user"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _case_session_id(case: LongMemEvalCase, session_index: int) -> str:
    if session_index < len(case.session_ids):
        return safe_case_id(case.session_ids[session_index])
    return f"{case.case_id}_s{session_index}"
