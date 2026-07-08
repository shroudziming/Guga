from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

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
        cases.append(
            LongMemEvalCase(
                case_id=safe_case_id(case_id),
                question=question,
                answer=_first_text(row, "answer", "reference_answer", "gold", "target"),
                question_type=_first_text(row, "question_type", "category", "type"),
                sessions=_extract_sessions(row),
                raw=row,
            )
        )
    return cases


def ingest_longmemeval_case(case: LongMemEvalCase, manager: MemoryManager) -> dict[str, int]:
    messages = 0
    for session_index, session in enumerate(case.sessions):
        session_id = f"{case.case_id}_s{session_index}"
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
            messages += 1
    return {"sessions": len(case.sessions), "messages": messages}


def ingest_longmemeval_case_replay(case: LongMemEvalCase, manager: MemoryManager) -> dict[str, int]:
    messages = 0
    finalized_turns = 0
    completed_turns = 0
    consolidation_batches = 0
    low_level_updates = 0
    high_level_updates = 0
    high_level_noops = 0
    for session_index, session in enumerate(case.sessions):
        session_id = f"{case.case_id}_s{session_index}"
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
                    finalized_turns += 1
                    completed_turns += 1
                    has_open_user_turn = False
            else:
                if has_open_user_turn:
                    manager.finalize_turn(session_id)
                    finalized_turns += 1
                    completed_turns += 1
                manager.record_user_message(
                    session_id=session_id,
                    text=message.content,
                    source="benchmark:longmemeval",
                    created_at=message.created_at or None,
                )
                has_open_user_turn = True
            messages += 1
        if has_open_user_turn:
            manager.finalize_turn(session_id)
            finalized_turns += 1
            completed_turns += 1
        stats = manager.flush_session_memory(session_id)
        consolidation_batches += stats.get("consolidation_batches", 0)
        low_level_updates += stats.get("low_level_updates", 0)
        high_level_updates += stats.get("high_level_updates", 0)
        high_level_noops += stats.get("high_level_noops", 0)
    return {
        "sessions": len(case.sessions),
        "messages": messages,
        "finalized_turns": finalized_turns,
        "completed_turns": completed_turns,
        "consolidation_batches": consolidation_batches,
        "low_level_updates": low_level_updates,
        "high_level_updates": high_level_updates,
        "high_level_noops": high_level_noops,
    }


def run_longmemeval_case(
    case: LongMemEvalCase,
    model,
    workspace: BenchmarkWorkspace,
    generation: GenerationConfig,
    debug: bool = False,
    enable_semantic: bool = True,
    ingest_mode: str = "raw",
    replay_finalize_every: int = 10,
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
            enable_profile_updates=False,
            enable_personality_updates=False,
        )
        if ingest_mode == "replay"
        else None,
    )
    ingest_started = perf_counter()
    if ingest_mode == "replay":
        ingest_stats = ingest_longmemeval_case_replay(case, manager)
    elif ingest_mode == "raw":
        ingest_stats = ingest_longmemeval_case(case, manager)
    else:
        raise ValueError(f"Unsupported LongMemEval ingest_mode: {ingest_mode}")
    ingest_latency_ms = int((perf_counter() - ingest_started) * 1000)
    if manager.rag_pipeline is not None:
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
    prediction = session.reply(case.question, finalize_memory=False).strip()
    answer_latency_ms = int((perf_counter() - answer_started) * 1000)
    total_latency_ms = int((perf_counter() - total_started) * 1000)

    result = {
        "case_id": case.case_id,
        "question_type": case.question_type,
        "question": case.question,
        "answer": case.answer,
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
) -> list[dict[str, Any]]:
    cases = load_longmemeval_cases(dataset_path)
    if limit is not None:
        cases = cases[: max(0, limit)]

    results: list[dict[str, Any]] = []
    for case in cases:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


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
