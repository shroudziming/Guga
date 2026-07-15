from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from time import perf_counter, sleep
from uuid import uuid4

from guga.config import (
    DEFAULT_CURRENT_TURN_SCORE_FACTOR,
    DEFAULT_DOCUMENT_TOP_K,
    DEFAULT_MEMORY_DECAY_ENABLED,
    DEFAULT_MEMORY_DECAY_MIN_AGE_DAYS,
    DEFAULT_MEMORY_DECAY_THRESHOLD,
    DEFAULT_MEMORY_MIN_SCORE,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_RAG_CHUNK_OVERLAP,
    DEFAULT_RAG_CHUNK_SIZE,
    DEFAULT_RAG_EMBEDDING_MODEL,
    DEFAULT_RAG_ENABLE_SEMANTIC,
)
from guga.memory.agent_identity import AgentIdentity, agent_memory_root
from guga.memory.consolidation import MemoryConsolidationConfig
from guga.memory.event_summary_store import EventSummaryStore
from guga.memory.forgetting import normalize_memorybank_fields, refresh_jsonl_retention, reinforce_jsonl_records, retention_score
from guga.memory.semantic_events import SemanticEventStore
from guga.memory_source_validity import active_event_ids, uses_only_active_event_sources
from guga.memory.summarizer import MemoryBankSummarizer, SummaryGenerationError
from guga.memory.time_utils import apply_temporal_fields, day_bucket as time_day_bucket, extract_semantic_time, now_beijing, now_beijing_iso, parse_datetime
from guga.memory.user_model import GugaUserModelStore
from guga.rag.faiss_store import IncompatibleIndexError
from guga.rag.pipeline import RagPipeline
from guga.rag.schemas import RetrievalHit
from guga.types import DocumentHit, MemoryContext, MemoryHit
from guga.utils.paths import memory_data_dir, rag_documents_dir


@dataclass(frozen=True)
class _QueryPlan:
    route: str
    time_hints: dict[str, str | bool]
    reason: str = ""
    day: str = ""
    preferred_session_id: str = ""

    def as_debug_payload(self) -> dict[str, str | bool]:
        payload: dict[str, str | bool] = {"route": self.route, "reason": self.reason}
        if self.day:
            payload["day"] = self.day
        if self.preferred_session_id:
            payload["preferred_session_id"] = self.preferred_session_id
        return payload


class _SessionStore:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def create_session_id(self) -> str:
        return f"sess_{uuid4().hex[:12]}"

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        source: str = "chat",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> str:
        message_id = f"msg_{uuid4().hex[:10]}"
        target = self.session_dir / f"{session_id}.jsonl"
        payload = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "source": source,
            "metadata": metadata or {},
            "created_at": created_at or now_beijing_iso(),
        }
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return message_id


class MemoryManager:
    """Manage memory retrieval/writeback around the chat turn lifecycle.

    This class is the center of the RAG flow used by ChatSession:
    - before generation: retrieve memory/doc context (prepare_context)
    - prompt assembly: inject retrieval hits into system prompt
    - after generation: write back user memory and update vector index
    """

    def __init__(
        self,
        memory_root: Path | None = None,
        agent_identity: AgentIdentity | None = None,
        model=None,
        debug: bool = False,
        debug_sink: Callable[[str], None] | None = None,
        top_k: int = DEFAULT_MEMORY_TOP_K,
        document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
        enable_semantic: bool = DEFAULT_RAG_ENABLE_SEMANTIC,
        documents_dir: Path | None = None,
        consolidation_config: MemoryConsolidationConfig | None = None,
    ) -> None:
        """Create manager with BGE-M3 semantic retrieval capabilities.

        Args:
            memory_root: Root directory for memory files and indexes.
            agent_identity: Optional persona/agent binding for isolated memory roots.
            model: Reserved for future use (kept for API compatibility).
            debug: Whether to emit debug traces.
            debug_sink: Optional debug output sink callback.
            top_k: Max memory hits returned to prompt context.
            document_top_k: Max document hits returned to prompt context.
            enable_semantic: Whether to enable vector-based retrieval pipeline.
            documents_dir: Root directory for document retrieval.
        """
        self.model = model
        self.agent_identity = agent_identity
        if memory_root is None and self.agent_identity is None:
            self.agent_identity = self._default_agent_identity()
        if memory_root is not None:
            self.memory_root = Path(memory_root)
        elif self.agent_identity is not None:
            self.memory_root = agent_memory_root(self.agent_identity.agent_id)
        else:
            self.memory_root = memory_data_dir()
        self.documents_dir = documents_dir or rag_documents_dir()
        self.debug = debug
        self.debug_sink = debug_sink
        self.top_k = max(1, top_k)
        self.document_top_k = max(1, document_top_k)
        self.decay_enabled = self._env_bool("Guga_MEMORY_DECAY_ENABLED", DEFAULT_MEMORY_DECAY_ENABLED)
        self.decay_threshold = self._env_float(
            "Guga_MEMORY_DECAY_THRESHOLD",
            DEFAULT_MEMORY_DECAY_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        )
        self.decay_min_age_days = self._env_float(
            "Guga_MEMORY_DECAY_MIN_AGE_DAYS",
            DEFAULT_MEMORY_DECAY_MIN_AGE_DAYS,
            minimum=0.0,
            maximum=36500.0,
        )
        self.reinforce_min_score = 0.55
        self.current_turn_score_factor = self._env_float(
            "Guga_CURRENT_TURN_SCORE_FACTOR",
            DEFAULT_CURRENT_TURN_SCORE_FACTOR,
            minimum=0.0,
            maximum=1.0,
        )
        self.memory_min_score = self._env_float(
            "Guga_MEMORY_MIN_SCORE",
            DEFAULT_MEMORY_MIN_SCORE,
            minimum=0.0,
            maximum=10.0,
        )

        self.memory_root.mkdir(parents=True, exist_ok=True)
        self._validate_or_create_agent_manifest()
        self.archival_file = self.memory_root / "archival_memory.jsonl"
        self.session_memory_file = self.memory_root / "session_memories.jsonl"
        self.semantic_event_file = self.memory_root / "semantic_events.jsonl"
        self.consolidation_state_file = self.memory_root / "consolidation_state.json"
        self.event_summary_store = EventSummaryStore(self.memory_root / "event_summaries.jsonl")
        self.semantic_event_store = SemanticEventStore(self.semantic_event_file)
        self.user_model_store = GugaUserModelStore(self.memory_root / "guga_user_model.json")
        self.summarizer = MemoryBankSummarizer(model=model)
        self.consolidation_config = (consolidation_config or MemoryConsolidationConfig()).normalized()
        self.session_store = _SessionStore(self.memory_root / "sessions")
        self._turn_state: dict[str, dict[str, str]] = {}
        self._date_context_by_session: dict[str, str] = {}
        self._turn_state_lock = RLock()
        self._finalize_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guga-memory")
        self._finalize_futures: list[Future] = []

        self.rag_pipeline: RagPipeline | None = None
        self._semantic_ready = False
        if enable_semantic:
            self.rag_pipeline = RagPipeline(
                index_dir=self.memory_root / "rag" / "index",
                documents_dir=self.documents_dir,
                embedding_model=DEFAULT_RAG_EMBEDDING_MODEL,
                chunk_size=DEFAULT_RAG_CHUNK_SIZE,
                chunk_overlap=DEFAULT_RAG_CHUNK_OVERLAP,
                debug_hook=self._debug_pipeline,
            )

    def _default_agent_identity(self) -> AgentIdentity:
        return AgentIdentity(
            agent_id="default",
            reflection_context="default persona",
            persona_source="builtin:default",
            persona_fingerprint="builtin:default",
        )

    def _validate_or_create_agent_manifest(self) -> None:
        if self.agent_identity is None:
            return
        manifest_path = self.memory_root / "agent_manifest.json"
        expected = {
            "schema_version": 1,
            "agent_id": self.agent_identity.agent_id,
            "persona_source": self.agent_identity.persona_source,
            "persona_fingerprint": self.agent_identity.persona_fingerprint,
        }
        allowed_keys = {*expected.keys(), "created_at"}
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if set(payload) != allowed_keys:
                raise ValueError(f"invalid agent manifest keys in {manifest_path}")
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f"agent manifest mismatch for {key}")
            return

        payload = dict(expected)
        payload["created_at"] = now_beijing_iso()
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def prepare_context(self, user_text: str, session_id: str) -> MemoryContext:
        """Retrieve context for current user input and return structured hits.

        Upstream:
            Called by ChatSession before model generation.

        Retrieval steps:
            1) Load active records used to resolve semantic hit metadata.
            2) Retrieve BGE-M3 chunk hits from RagPipeline (memory + documents).
            3) Apply deterministic lifecycle, time, and prompt-noise filters.

        Args:
            user_text: Current user query text.
            session_id: Active session id for trace/debug logs.

        Returns:
            MemoryContext containing:
            - hits: merged memory hits used in prompt
            - document_hits: semantic document hits
            - archival_memories: summaries of selected memory hits
        """
        started = perf_counter()
        self._apply_decay_policy(session_id)
        current_turn_ids = self._current_turn_ids(session_id)
        records = self._load_archival_records()
        query_plan = self._build_query_plan(user_text, session_id, records)
        if query_plan.route == "recent_turn":
            records.extend(self._load_session_message_records(session_id))
        records = self._records_for_query_plan(records, query_plan, session_id)
        time_hints = query_plan.time_hints
        self._debug(
            session_id,
            f"retrieve_start query={json.dumps(user_text, ensure_ascii=False)} top_k={self.top_k} doc_top_k={self.document_top_k} candidates={len(records)} min_score={self.memory_min_score:.2f} current_turn_factor={self.current_turn_score_factor:.2f} query_plan={json.dumps(query_plan.as_debug_payload(), ensure_ascii=False)} time_hints={json.dumps(time_hints, ensure_ascii=False)}",
        )

        semantic_memory_hits, semantic_document_hits = self._retrieve_semantic(user_text=user_text, session_id=session_id)

        merged_memory_hits = self._merge_memory_hits(
            semantic_memory_hits,
            records,
            current_turn_ids=current_turn_ids,
            time_hints=time_hints,
            session_id=session_id,
        )
        merged_memory_hits = self._dedupe_semantic_event_overlaps(merged_memory_hits, query_plan)
        self._reinforce_recalled_memories(merged_memory_hits, session_id=session_id, query=user_text, current_turn_ids=current_turn_ids)
        document_hits = self._to_document_hits(semantic_document_hits)
        current_event_ids = active_event_ids(self.semantic_event_store.load_all())
        user_portrait = "\n".join(
            str(insight.get("statement", "")).strip()
            for insight in self.user_model_store.load().get("insights", [])
            if uses_only_active_event_sources(insight, current_event_ids)
            and str(insight.get("statement", "")).strip()
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        memory_hit_ids = [hit.id for hit in merged_memory_hits]
        doc_hit_ids = [hit.chunk_id for hit in document_hits]
        source_ids = [hit.source_session_id for hit in merged_memory_hits]
        memory_raw_payload = [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "summary": hit.summary,
                "raw_excerpt": hit.raw_excerpt,
                "memory_type": hit.memory_type,
                "retention": hit.retention,
                "memory_strength": hit.memory_strength,
                "source_session_id": hit.source_session_id,
                "source_message_ids": hit.source_message_ids,
                "day": hit.day,
                "valid_at": hit.valid_at,
                "invalid_at": hit.invalid_at,
                "time_source": hit.time_source,
                "semantic_score": hit.semantic_score,
                "score_source": hit.score_source,
                "score_components": hit.score_components,
                "is_current_turn": hit.is_current_turn,
            }
            for hit in merged_memory_hits
        ]
        self._debug(
            session_id,
            f"retrieve_done query={json.dumps(user_text, ensure_ascii=False)} top_k={self.top_k} doc_top_k={self.document_top_k} selected_mem={len(merged_memory_hits)} selected_doc={len(document_hits)} hit_ids={memory_hit_ids} doc_hit_ids={doc_hit_ids} source_ids={source_ids} memory_raw={json.dumps(memory_raw_payload, ensure_ascii=False)} latency_ms={elapsed_ms}",
        )

        return MemoryContext(
            archival_memories=[hit.summary for hit in merged_memory_hits],
            hits=merged_memory_hits,
            document_hits=document_hits,
            user_portrait=user_portrait,
            query_route=query_plan.route,
            query_reason=query_plan.reason,
        )

    def compose_system_prompt(self, base_prompt: str, memory_context: MemoryContext) -> str:
        """Build final system prompt by combining persona + retrieval results.

        Args:
            base_prompt: Persona/system base instruction.
            memory_context: Retrieval result from prepare_context.

        Returns:
            A single system prompt string consumed by the chat model.
        """
        sections = ["[Base Persona]", base_prompt]
        semantic_events = [hit for hit in memory_context.hits if hit.memory_type == "semantic_event"]
        derived_summaries = [hit for hit in memory_context.hits if hit.memory_type == "event_summary"]
        raw_evidence = [
            hit
            for hit in memory_context.hits
            if hit.memory_type == "conversation_turn" and not hit.is_current_turn
        ]
        archival_memories = [
            hit
            for hit in memory_context.hits
            if hit.memory_type not in {"semantic_event", "event_summary", "conversation_turn"}
        ]

        if semantic_events:
            sections.append("\n[Semantic Events]")
            sections.extend(self._format_memory_hit(hit) for hit in semantic_events)

        if archival_memories:
            sections.append("\n[Archival Memory]")
            sections.extend(self._format_memory_hit(hit) for hit in archival_memories)

        if derived_summaries:
            sections.append("\n[Derived Event Summaries]")
            sections.extend(self._format_memory_hit(hit) for hit in derived_summaries)

        if raw_evidence:
            sections.append("\n[Raw Evidence]")
            sections.extend(self._format_memory_hit(hit) for hit in raw_evidence)

        if memory_context.user_portrait:
            sections.append("\n[Guga User Model]")
            sections.append(memory_context.user_portrait)

        if memory_context.document_hits:
            sections.append("\n[Relevant Documents]")
            for hit in memory_context.document_hits:
                source_ref = hit.source_id or hit.source_path
                sections.append(f"- ({hit.chunk_id} | score={hit.score:.2f} | src={source_ref}) {hit.text}")

        if semantic_events or archival_memories or derived_summaries or raw_evidence or memory_context.document_hits:
            sections.append("\n[Current Rule]")
            sections.append(
                "请按层级优先级使用证据：当前 Semantic Events 优先于 Archival Memory，"
                "再优先于 Derived Event Summaries 和 Raw Evidence。Guga User Model 只用于理解用户，"
                "不得覆盖客观事实；如果证据不足，请说明没有找到可靠信息，不要编造。"
            )
        return "\n".join(sections)

    def _format_memory_hit(self, hit: MemoryHit) -> str:
        source_message = hit.source_message_ids[0] if hit.source_message_ids else ""
        source_ref = f"{hit.source_session_id}/{source_message}".strip("/")
        timestamp = self._format_beijing_minute(hit.created_at)
        text = hit.summary
        if hit.memory_type == "conversation_turn" and not hit.chunk_id and len(text) > DEFAULT_RAG_CHUNK_SIZE:
            text = text[:DEFAULT_RAG_CHUNK_SIZE].rstrip() + "..."
        chunk_ref = f" | chunk={hit.chunk_id}" if hit.chunk_id else ""
        return (
            f"- ({hit.id} | score={hit.score:.2f} | retention={hit.retention:.2f} | "
            f"S={hit.memory_strength} | at={timestamp} | src={source_ref}{chunk_ref}) {text}"
        )

    def _format_beijing_minute(self, value: str) -> str:
        if not value:
            return "未知时间"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        beijing = parsed.astimezone(timezone(timedelta(hours=8)))
        return beijing.strftime("%Y-%m-%d %H:%M 北京时间")

    def record_user_message(self, session_id: str, text: str, source: str = "chat", created_at: str | None = None) -> str:
        """Persist current user message and cache it in per-turn state.

        Returns:
            message_id written to sessions/<session_id>.jsonl.
        """
        message_id = self.session_store.append_message(
            session_id=session_id,
            role="user",
            content=text,
            source=source,
            created_at=created_at,
        )
        turn_payload = self._build_session_memory(session_id=session_id, message_id=message_id, text=text, created_at=created_at)
        self._append_jsonl(self.session_memory_file, turn_payload)
        with self._turn_state_lock:
            state = self._turn_state.setdefault(session_id, {})
            state["user_text"] = text
            state["user_message_id"] = message_id
            state["session_memory_id"] = turn_payload["id"]
            parsed_created_at = parse_datetime(created_at)
            if parsed_created_at is not None:
                self._date_context_by_session[session_id] = parsed_created_at.date().isoformat()
        self._debug(session_id, f"ingest role=user message_id={message_id}")
        return message_id

    def record_assistant_message(
        self,
        session_id: str,
        text: str,
        source: str = "chat",
        created_at: str | None = None,
        store_as_memory: bool = False,
    ) -> str:
        """Persist assistant reply and cache it in per-turn state.

        Returns:
            message_id written to sessions/<session_id>.jsonl.
        """
        message_id = self.session_store.append_message(
            session_id=session_id,
            role="assistant",
            content=text,
            source=source,
            created_at=created_at,
        )
        with self._turn_state_lock:
            state = self._turn_state.setdefault(session_id, {})
            state["assistant_text"] = text
            state["assistant_message_id"] = message_id
            if store_as_memory:
                turn_payload = self._build_session_memory(
                    session_id=session_id,
                    message_id=message_id,
                    text=f"assistant: {text}",
                    created_at=created_at,
                )
                self._append_jsonl(self.session_memory_file, turn_payload)
                state["assistant_session_memory_id"] = turn_payload["id"]
        self._debug(session_id, f"ingest role=assistant message_id={message_id}")
        return message_id

    def finalize_turn(self, session_id: str) -> None:
        """Finalize one turn: optional archival writeback and index update.

        Upstream:
            Called by ChatSession after assistant response is generated.

        Behavior:
            - If user text passes archive policy, append one memory record.
            - If semantic retrieval is enabled, update vector index incrementally.
            - Clear cached per-turn state for this session.
        """
        state = self._pop_turn_state(session_id)
        self._finalize_turn_state(session_id=session_id, state=state)

    def finalize_turn_async(self, session_id: str) -> Future:
        """Queue turn finalization in the background and return immediately."""
        state = self._pop_turn_state(session_id)
        if not state:
            completed: Future = Future()
            completed.set_result(None)
            return completed

        self._debug(session_id, "finalize_background_queued")
        future = self._finalize_executor.submit(self._finalize_turn_state, session_id, state)
        self._finalize_futures.append(future)

        def _log_done(done: Future) -> None:
            try:
                done.result()
            except Exception as exc:
                self._debug(session_id, f"finalize_background_failed reason={exc}")
                return
            self._debug(session_id, "finalize_background_done")

        future.add_done_callback(_log_done)
        return future

    def wait_for_background_tasks(self, timeout: float | None = None) -> None:
        """Wait for queued background memory work; useful in tests or graceful shutdown."""
        futures = list(self._finalize_futures)
        for future in futures:
            future.result(timeout=timeout)

    def flush_session_memory(self, session_id: str) -> dict[str, int]:
        """Force consolidation for any pending completed turns in a session."""
        return self._consolidate_pending_turns(session_id=session_id, force=True)

    def consolidate_until_settled(
        self,
        session_id: str,
        *,
        max_retry_cycles: int = 3,
        retry_delays: tuple[float, ...] = (2.0, 5.0, 10.0),
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Synchronously settle one session's ordered memory chain."""
        result = self.flush_session_memory(session_id)
        while not bool(result.get("memory_complete", False)):
            cycle = int(result.get("retry_cycle", 0) or 0)
            stage = str(result.get("pending_stage", ""))
            if cycle >= max_retry_cycles:
                if progress is not None:
                    progress({"phase": "stage_retry_exhausted", "stage": stage, "cycle": cycle})
                result["status"] = "failed"
                return result
            if progress is not None:
                progress({"phase": "stage_retry_started", "stage": stage, "cycle": cycle + 1})
            if retry_delays:
                delay_index = min(max(cycle - 1, 0), len(retry_delays) - 1)
                sleep(max(float(retry_delays[delay_index]), 0.0))
            result = self._consolidate_pending_turns(session_id=session_id, force=True)
            if bool(result.get("memory_complete", False)) and progress is not None:
                progress({"phase": "stage_retry_succeeded", "stage": stage, "cycle": cycle + 1})
        return result

    def _pop_turn_state(self, session_id: str) -> dict[str, str]:
        with self._turn_state_lock:
            return dict(self._turn_state.pop(session_id, {}))

    def _finalize_turn_state(self, session_id: str, state: dict[str, str]) -> None:
        started = perf_counter()
        session_memory_payload = self._load_memory_by_id(self.session_memory_file, state.get("session_memory_id", ""))
        if session_memory_payload and self.rag_pipeline is not None:
            try:
                self.rag_pipeline.add_memory_record(session_memory_payload)
            except Exception as exc:
                self._debug(session_id, f"index_update status=session_memory_failed reason={exc}")
        if not state.get("user_message_id"):
            elapsed_ms = int((perf_counter() - started) * 1000)
            self._debug(session_id, f"writeback status=no_completed_turn latency_ms={elapsed_ms}")
            return

        pending_count = self._append_pending_turn(session_id=session_id, state=state)
        elapsed_ms = int((perf_counter() - started) * 1000)
        self._debug(session_id, f"writeback status=pending_turn_added pending={pending_count} latency_ms={elapsed_ms}")
        if pending_count >= self.consolidation_config.batch_turns:
            self._consolidate_pending_turns(session_id=session_id, force=False)

    def _append_pending_turn(self, *, session_id: str, state: dict[str, str]) -> int:
        if not state.get("user_message_id"):
            return 0
        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            session_state = self._session_consolidation_state(payload, session_id)
            target = "queued_turns" if isinstance(session_state.get("active_batch"), dict) else "pending_turns"
            session_state[target].append(
                {
                    "user_message_id": state.get("user_message_id", ""),
                    "assistant_message_id": state.get("assistant_message_id", ""),
                    "session_memory_id": state.get("session_memory_id", ""),
                    "created_at": str(self._load_memory_by_id(self.session_memory_file, state.get("session_memory_id", "")).get("created_at", "")),
                }
            )
            session_state["completed_turns"] = int(session_state.get("completed_turns", 0) or 0) + 1
            pending_count = len(session_state[target])
            self._write_consolidation_state(payload)
            return pending_count

    def _consolidate_pending_turns(self, *, session_id: str, force: bool) -> dict[str, Any]:
        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            session_state = self._session_consolidation_state(payload, session_id)
            active = session_state.get("active_batch")
            if not isinstance(active, dict):
                active = None
            legacy_high = session_state.get("pending_high_level")
            if active is None and isinstance(legacy_high, dict):
                active = {
                    "batch_seq": int(legacy_high.get("batch_seq", 0) or 0),
                    "stage": "high",
                    "status": "pending_retry",
                    "attempt_count": 0,
                    "retry_cycle": 0,
                    "turns": [],
                    "low_level_updates": int(legacy_high.get("low_level_updates", 0) or 0),
                }
                session_state["active_batch"] = active
                self._write_consolidation_state(payload)
            pending_turns = list(session_state.get("pending_turns", []) or [])
            if active is None and not pending_turns:
                return self._consolidation_result(session_state, status="idle")
            if active is None and not force and len(pending_turns) < self.consolidation_config.batch_turns:
                return self._consolidation_result(session_state, status="waiting_batch")
            if active is None and not self.summarizer.use_llm:
                self._debug(session_id, "consolidation_skipped reason=no_llm_model")
                return self._consolidation_result(session_state, status="no_llm")
            if active is None:
                active = {
                    "batch_seq": int(session_state.get("batch_seq", 0) or 0) + 1,
                    "stage": "low",
                    "status": "retrying",
                    "attempt_count": 0,
                    "retry_cycle": 0,
                    "turns": pending_turns,
                }
                session_state["active_batch"] = active
                session_state["pending_turns"] = []
                self._write_consolidation_state(payload)

        batch_seq = int(active.get("batch_seq", 0) or 0)
        if str(active.get("stage", "low")) == "high":
            return self._run_high_stage(session_id=session_id, batch_seq=batch_seq, force=force)

        pending_turns = list(active.get("turns", []) or [])

        try:
            low_packet = self._build_low_level_packet(session_id=session_id, pending_turns=pending_turns)
            low_result = self.summarizer.consolidate_low_level_memory(
                low_packet,
                include_guga_reflection=self.consolidation_config.include_guga_reflection,
            )
            self._debug_structured_attempts(session_id, batch_seq=batch_seq, stage="low")
            low_counts = self._apply_low_level_consolidation(
                session_id=session_id,
                batch_seq=batch_seq,
                result=low_result,
                pending_turns=pending_turns,
            )
        except SummaryGenerationError as exc:
            self._debug_structured_attempts(session_id, batch_seq=batch_seq, stage="low")
            self._mark_active_batch_failure(session_id=session_id, batch_seq=batch_seq, stage="low", error=exc)
            self._debug(session_id, f"consolidation_failed reason={exc}")
            return self._current_consolidation_result(session_id, status="pending_retry")

        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            session_state = self._session_consolidation_state(payload, session_id)
            active = session_state.get("active_batch")
            if isinstance(active, dict) and int(active.get("batch_seq", 0) or 0) == batch_seq:
                active.update(
                    {
                        "stage": "high",
                        "status": "retrying",
                        "attempt_count": 0,
                        "low_level_updates": int(low_counts.get("low_level_updates", 0) or 0),
                        "low_commit_key": f"{session_id}:{batch_seq}:low",
                    }
                )
            self._write_consolidation_state(payload)
        return self._run_high_stage(session_id=session_id, batch_seq=batch_seq, force=force)

    def _run_high_stage(self, *, session_id: str, batch_seq: int, force: bool) -> dict[str, Any]:
        try:
            high_result = self.summarizer.consolidate_high_level_memory(self._build_high_level_packet())
            self._debug_structured_attempts(session_id, batch_seq=batch_seq, stage="high")
            high_counts = self._apply_high_level_consolidation(
                session_id=session_id,
                batch_seq=batch_seq,
                result=high_result,
                low_source_message_ids=[],
            )
        except SummaryGenerationError as exc:
            self._debug_structured_attempts(session_id, batch_seq=batch_seq, stage="high")
            self._mark_active_batch_failure(session_id=session_id, batch_seq=batch_seq, stage="high", error=exc)
            self._debug(session_id, f"high_level_retry_failed batch_seq={batch_seq} reason={exc}")
            return self._current_consolidation_result(session_id, status="pending_retry")

        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            session_state = self._session_consolidation_state(payload, session_id)
            active = session_state.get("active_batch")
            if not isinstance(active, dict) or int(active.get("batch_seq", 0) or 0) != batch_seq:
                return self._consolidation_result(session_state, status="stale")
            low_updates = int(active.get("low_level_updates", 0) or 0)
            session_state["active_batch"] = None
            session_state["pending_high_level"] = None
            session_state["pending_turns"] = list(session_state.get("queued_turns", []) or [])
            session_state["queued_turns"] = []
            session_state["batch_seq"] = batch_seq
            session_state["consolidation_batches"] = int(session_state.get("consolidation_batches", 0) or 0) + 1
            session_state["low_level_updates"] = int(session_state.get("low_level_updates", 0) or 0) + low_updates
            session_state["high_level_updates"] = int(session_state.get("high_level_updates", 0) or 0) + high_counts["high_level_updates"]
            session_state["high_level_noops"] = int(session_state.get("high_level_noops", 0) or 0) + high_counts["high_level_noops"]
            self._write_consolidation_state(payload)
            should_continue = bool(session_state["pending_turns"]) and (
                force or len(session_state["pending_turns"]) >= self.consolidation_config.batch_turns
            )
            result = self._consolidation_result(session_state, status="complete")
        self._debug(
            session_id,
            f"consolidation_done batch_seq={batch_seq} low_updates={low_updates} high_updates={high_counts['high_level_updates']} high_noops={high_counts['high_level_noops']}",
        )
        if should_continue:
            return self._consolidate_pending_turns(session_id=session_id, force=force)
        return result

    def _mark_active_batch_failure(
        self,
        *,
        session_id: str,
        batch_seq: int,
        stage: str,
        error: SummaryGenerationError,
    ) -> None:
        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            state = self._session_consolidation_state(payload, session_id)
            active = state.get("active_batch")
            if not isinstance(active, dict) or int(active.get("batch_seq", 0) or 0) != batch_seq:
                return
            active.update(
                {
                    "stage": stage,
                    "status": "pending_retry",
                    "attempt_count": int(getattr(error, "attempts", 0) or 0),
                    "retry_cycle": int(active.get("retry_cycle", 0) or 0) + 1,
                    "next_retry_at": now_beijing_iso(),
                    "last_error_type": str(getattr(error, "error_type", "schema")),
                    "last_error": str(error),
                    "response_hash": str(getattr(error, "response_hash", "")),
                }
            )
            if stage == "high":
                state["pending_high_level"] = {
                    "batch_seq": batch_seq,
                    "low_level_updates": int(active.get("low_level_updates", 0) or 0),
                }
            self._write_consolidation_state(payload)

    def _current_consolidation_result(self, session_id: str, *, status: str) -> dict[str, Any]:
        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            state = self._session_consolidation_state(payload, session_id)
            return self._consolidation_result(state, status=status)

    def _consolidation_result(self, state: dict, *, status: str) -> dict[str, Any]:
        active = state.get("active_batch")
        return {
            "consolidation_batches": int(state.get("consolidation_batches", 0) or 0),
            "low_level_updates": int(state.get("low_level_updates", 0) or 0),
            "high_level_updates": int(state.get("high_level_updates", 0) or 0),
            "high_level_noops": int(state.get("high_level_noops", 0) or 0),
            "status": status,
            "memory_complete": not isinstance(active, dict),
            "pending_stage": str(active.get("stage", "")) if isinstance(active, dict) else "",
            "retry_cycle": int(active.get("retry_cycle", 0) or 0) if isinstance(active, dict) else 0,
        }

    def _debug_structured_attempts(self, session_id: str, *, batch_seq: int, stage: str) -> None:
        for diagnostic in self.summarizer.last_structured_attempts:
            self._debug(
                session_id,
                "structured_call "
                + json.dumps(
                    {"batch_seq": batch_seq, "stage": stage, **diagnostic},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    def consolidation_stats(self, session_id: str) -> dict[str, int]:
        payload = self._read_consolidation_state()
        session_state = self._session_consolidation_state(payload, session_id)
        return {
            "completed_turns": int(session_state.get("completed_turns", 0) or 0),
            "consolidation_batches": int(session_state.get("consolidation_batches", 0) or 0),
            "low_level_updates": int(session_state.get("low_level_updates", 0) or 0),
            "high_level_updates": int(session_state.get("high_level_updates", 0) or 0),
            "high_level_noops": int(session_state.get("high_level_noops", 0) or 0),
        }

    def has_active_consolidation(self, session_id: str) -> bool:
        """Return whether a persisted batch must be resumed before accepting replay input."""
        with self._turn_state_lock:
            payload = self._read_consolidation_state()
            session_state = (payload.get("sessions", {}) or {}).get(session_id, {})
            return isinstance(session_state.get("active_batch"), dict)

    def _build_low_level_packet(self, *, session_id: str, pending_turns: list[dict]) -> dict:
        new_turns = self._load_pending_turn_messages(session_id=session_id, pending_turns=pending_turns)
        query = "\n".join(
            str(turn.get(key, ""))
            for turn in new_turns
            for key in ("user_text", "assistant_text")
            if str(turn.get(key, "")).strip()
        )
        retrieved_context = self._consolidation_retrieved_context(query)
        event_context = self._select_low_level_event_context(retrieved_context)
        packet = {
            "new_turns": new_turns,
            "recent_active_events": event_context["recent_active_events"],
            "relevant_active_events": event_context["relevant_active_events"],
            "retrieved_context": retrieved_context,
        }
        return self._trim_packet(packet)

    def _select_low_level_event_context(self, retrieved_context: list[dict]) -> dict[str, list[dict]]:
        active_events = self.semantic_event_store.load_active()
        active_by_id = {str(event.get("id", "")): event for event in active_events}
        semantic_ids: list[str] = []
        for item in retrieved_context:
            event_id = str(item.get("id", ""))
            if event_id and event_id not in semantic_ids:
                semantic_ids.append(event_id)
        return {
            "recent_active_events": active_events[-5:],
            "relevant_active_events": [active_by_id[event_id] for event_id in semantic_ids if event_id in active_by_id],
        }

    def _build_high_level_packet(self) -> dict:
        events = self.semantic_event_store.load_all()
        current_event_ids = active_event_ids(events)
        user_model = self.user_model_store.load()
        user_model = {
            **user_model,
            "insights": [
                insight
                for insight in user_model.get("insights", [])
                if isinstance(insight, dict) and uses_only_active_event_sources(insight, current_event_ids)
            ],
        }
        packet = {
            "semantic_events": [event for event in events if event.get("id") in current_event_ids][-50:],
            "event_summaries": [
                item
                for item in self.event_summary_store.load_active()
                if uses_only_active_event_sources(item, current_event_ids)
            ][-50:],
            "archival_memory": [
                item
                for item in self._read_jsonl_records(self.archival_file)
                if uses_only_active_event_sources(item, current_event_ids)
            ][-50:],
            "guga_user_model": user_model,
        }
        return self._trim_packet(packet)

    def _apply_low_level_consolidation(
        self,
        *,
        session_id: str,
        batch_seq: int,
        result: dict,
        pending_turns: list[dict],
    ) -> dict[str, int]:
        updates = 0
        source_message_ids = self._pending_source_message_ids(pending_turns)
        operations = self._event_operations_with_references(
            session_id=session_id,
            batch_seq=batch_seq,
            pending_turns=pending_turns,
            operations=result.get("semantic_event_operations", []) or [],
            fallback_source_ids=source_message_ids,
        )
        outcome = self.semantic_event_store.apply_operations(
            operations=operations,
            session_id=session_id,
            include_guga_reflection=self.consolidation_config.include_guga_reflection,
        )
        changed_ids = set(outcome.created_event_ids + outcome.updated_event_ids + outcome.deactivated_event_ids)
        all_events = self.semantic_event_store.load_all()
        current_event_ids = active_event_ids(all_events)
        changed_events = [event for event in all_events if event.get("id") in changed_ids]
        updates += len(changed_ids)
        if self.rag_pipeline is not None:
            for payload in changed_events:
                try:
                    self.rag_pipeline.add_memory_record(payload, active_event_ids=current_event_ids)
                except Exception as exc:
                    self._debug(session_id, f"index_update status=semantic_event_failed reason={exc}")
        for item in result.get("event_summaries", []) or []:
            if not isinstance(item, dict):
                continue
            payload = self.event_summary_store.upsert_batch_summary(
                session_id=session_id,
                batch_seq=batch_seq,
                payload=item,
                source_message_ids=source_message_ids,
                event_result=outcome,
                covered_events=changed_events,
            )
            if payload:
                updates += 1
                if self.rag_pipeline is not None:
                    try:
                        self.rag_pipeline.add_memory_record(payload, active_event_ids=current_event_ids)
                    except Exception as exc:
                        self._debug(session_id, f"index_update status=event_summary_failed reason={exc}")
        if self.rag_pipeline is not None:
            try:
                self.rag_pipeline.prune_invalid_memory_records(self.memory_root)
            except Exception as exc:
                self._debug(session_id, f"index_update status=prune_failed reason={exc}")
        return {"low_level_updates": updates}

    def _event_operations_with_references(
        self,
        *,
        session_id: str,
        batch_seq: int,
        pending_turns: list[dict],
        operations: list[object],
        fallback_source_ids: list[str],
    ) -> list[dict]:
        rows = self._read_session_rows(self.memory_root / "sessions" / f"{session_id}.jsonl")
        created_at_by_id = {str(row.get("id", "")): str(row.get("created_at", "")) for row in rows}
        normalized: list[dict] = []
        for operation_index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise SummaryGenerationError("semantic event operation must be an object")
            item = dict(operation)
            if str(item.get("operation", "")) in {"create", "replace"}:
                digest = hashlib.sha256(
                    f"{session_id}:{batch_seq}:event:{operation_index}".encode("utf-8")
                ).hexdigest()[:24]
                item["event_id"] = f"evt_{digest}"
            source_ids = [str(value) for value in (item.get("source_message_ids") or fallback_source_ids) if str(value)]
            item["source_message_ids"] = source_ids
            if not item.get("reference_created_at"):
                item["reference_created_at"] = next(
                    (created_at_by_id.get(message_id, "") for message_id in source_ids if created_at_by_id.get(message_id)),
                    now_beijing_iso(),
                )
            normalized.append(item)
        return normalized

    def _apply_high_level_consolidation(
        self,
        *,
        session_id: str,
        batch_seq: int,
        result: dict,
        low_source_message_ids: list[str],
    ) -> dict[str, int]:
        decision = str(result.get("decision", "no_high_level_update"))
        if decision == "no_high_level_update":
            self._debug(session_id, f"high_level_noop reason={result.get('reason', '')}")
            return {"high_level_updates": 0, "high_level_noops": 1}

        updates = 0
        if self.consolidation_config.enable_archival_updates:
            for operation_index, item in enumerate(result.get("archival_operations", []) or []):
                if not isinstance(item, dict):
                    continue
                payload = self._build_archival_update(
                    item=item,
                    session_id=session_id,
                    batch_seq=batch_seq,
                    operation_index=operation_index,
                )
                if not payload:
                    continue
                self._upsert_jsonl(self.archival_file, payload)
                updates += 1
                if self.rag_pipeline is not None:
                    try:
                        self.rag_pipeline.add_memory_record(
                            payload,
                            active_event_ids=active_event_ids(self.semantic_event_store.load_all()),
                        )
                    except Exception as exc:
                        self._debug(session_id, f"index_update status=archival_failed reason={exc}")
        if self.consolidation_config.enable_user_model_updates:
            written = self.user_model_store.apply_operations(result.get("user_model_operations", []) or [])
            updates += len(written)
        return {"high_level_updates": updates, "high_level_noops": 0}

    def _build_archival_update(
        self,
        *,
        item: dict,
        session_id: str,
        batch_seq: int,
        operation_index: int,
    ) -> dict:
        summary = str(item.get("summary", "")).strip()
        if not summary:
            return {}
        now = now_beijing_iso()
        source_event_ids = [str(value) for value in (item.get("source_event_ids") or []) if str(value)]
        if not source_event_ids:
            return {}
        return normalize_memorybank_fields(
            apply_temporal_fields(
                {
                    "id": f"mem_{hashlib.sha256(f'{session_id}:{batch_seq}:archival:{operation_index}'.encode('utf-8')).hexdigest()[:24]}",
                    "type": "episodic",
                    "topic": str(item.get("topic") or "general").strip()[:64] or "general",
                    "summary": summary[:500],
                    "raw_excerpt": summary[:500],
                    "importance": self._clamp_float(item.get("importance"), 0.7),
                    "confidence": self._clamp_float(item.get("confidence"), 0.7),
                    "created_at": now,
                    "last_recalled_at": now,
                    "memory_strength": 1,
                    "retention": 1.0,
                    "source_session_id": session_id,
                    "source_event_ids": source_event_ids,
                    "status": "active",
                },
                text=summary,
                reference_time=now,
            )
        )

    def _load_pending_turn_messages(self, *, session_id: str, pending_turns: list[dict]) -> list[dict]:
        rows = self._read_session_rows(self.memory_root / "sessions" / f"{session_id}.jsonl")
        by_id = {str(row.get("id", "")): row for row in rows}
        new_turns: list[dict] = []
        for turn in pending_turns:
            user = by_id.get(str(turn.get("user_message_id", "")), {})
            assistant = by_id.get(str(turn.get("assistant_message_id", "")), {})
            if not user:
                continue
            new_turns.append(
                {
                    "user_message_id": str(user.get("id", "")),
                    "assistant_message_id": str(assistant.get("id", "")),
                    "created_at": str(user.get("created_at", "")),
                    "user_text": str(user.get("content", "")),
                    "assistant_text": str(assistant.get("content", "")),
                }
            )
        return new_turns

    def _pending_source_message_ids(self, pending_turns: list[dict]) -> list[str]:
        ids: list[str] = []
        for turn in pending_turns:
            for key in ("user_message_id", "assistant_message_id"):
                value = str(turn.get(key, "")).strip()
                if value:
                    ids.append(value)
        return ids

    def _consolidation_retrieved_context(self, query: str) -> list[dict]:
        if self.rag_pipeline is None or not query.strip():
            return []
        try:
            memory_hits, _ = self.rag_pipeline.retrieve(query, memory_top_k=8, document_top_k=0)
        except Exception:
            return []
        valid_source_ids = {
            str(record.get("id", ""))
            for record in self._load_archival_records()
            if str(record.get("id", ""))
        }
        return [
            {
                "id": str(hit.source_id),
                "type": str(hit.source_type),
                "summary": str(hit.text),
                "source_session_id": str(hit.source_session_id),
                "source_message_ids": [str(hit.source_message_id)] if hit.source_message_id else [],
                "semantic_score": float(hit.score),
            }
            for hit in memory_hits
            if str(hit.source_id) in valid_source_ids
        ]

    def _trim_packet(self, packet: dict) -> dict:
        text = json.dumps(packet, ensure_ascii=False)
        if len(text) <= self.consolidation_config.max_packet_chars:
            return packet
        trimmed = dict(packet)
        for key in (
            "retrieved_context",
            "recent_active_events",
            "relevant_active_events",
            "personality_insights",
        ):
            value = trimmed.get(key)
            if isinstance(value, list) and len(json.dumps(trimmed, ensure_ascii=False)) > self.consolidation_config.max_packet_chars:
                trimmed[key] = value[-5:]
        if len(json.dumps(trimmed, ensure_ascii=False)) > self.consolidation_config.max_packet_chars:
            for turn in trimmed.get("new_turns", []) or []:
                if isinstance(turn, dict):
                    turn["user_text"] = str(turn.get("user_text", ""))[-2000:]
                    turn["assistant_text"] = str(turn.get("assistant_text", ""))[-2000:]
        return trimmed

    def _read_consolidation_state(self) -> dict:
        if not self.consolidation_state_file.exists():
            return {"schema_version": 2, "sessions": {}}
        try:
            payload = json.loads(self.consolidation_state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"schema_version": 2, "sessions": {}}
        if not isinstance(payload, dict):
            return {"schema_version": 2, "sessions": {}}
        payload["schema_version"] = 2
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            payload["sessions"] = {}
        return payload

    def _write_consolidation_state(self, payload: dict) -> None:
        self.consolidation_state_file.parent.mkdir(parents=True, exist_ok=True)
        payload["schema_version"] = 2
        temporary = self.consolidation_state_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.consolidation_state_file)

    def _session_consolidation_state(self, payload: dict, session_id: str) -> dict:
        sessions = payload.setdefault("sessions", {})
        session_state = sessions.setdefault(
            session_id,
            {
                "batch_seq": 0,
                "pending_turns": [],
                "queued_turns": [],
                "active_batch": None,
                "pending_high_level": None,
                "completed_turns": 0,
                "consolidation_batches": 0,
                "low_level_updates": 0,
                "high_level_updates": 0,
                "high_level_noops": 0,
            },
        )
        session_state.setdefault("batch_seq", 0)
        session_state.setdefault("pending_turns", [])
        session_state.setdefault("queued_turns", [])
        session_state.setdefault("active_batch", None)
        session_state.setdefault("pending_high_level", None)
        session_state.setdefault("completed_turns", 0)
        session_state.setdefault("consolidation_batches", 0)
        session_state.setdefault("low_level_updates", 0)
        session_state.setdefault("high_level_updates", 0)
        session_state.setdefault("high_level_noops", 0)
        session_state["completed_turns"] = int(session_state.get("completed_turns", 0) or 0)
        return session_state

    def _read_jsonl_records(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _clamp_float(self, value: object, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = fallback
        return max(0.0, min(number, 1.0))

    def rebuild_rag_indexes(self, session_id: str = "manual") -> dict[str, int]:
        """Force full rebuild of memory/document vector indexes.

        Returns:
            Dict with counts: memory_chunks, document_chunks, total_chunks.
        """
        if self.rag_pipeline is None:
            return {"memory_chunks": 0, "document_chunks": 0, "total_chunks": 0}

        result = self.rag_pipeline.rebuild_indexes(memory_root=self.memory_root)
        self._semantic_ready = True
        self._debug(
            session_id,
            f"index_update memory_chunks={result['memory_chunks']} document_chunks={result['document_chunks']} total_chunks={result['total_chunks']}",
        )
        return result

    def _retrieve_semantic(self, user_text: str, session_id: str) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
        """Retrieve semantic hits from RagPipeline with safe failure fallback.

        Returns:
            (memory_hits, document_hits). Empty lists on disabled/failed cases.
        """
        if self.rag_pipeline is None:
            return [], []

        try:
            self._ensure_semantic_index(session_id)
            return self.rag_pipeline.retrieve(
                query=user_text,
                memory_top_k=max(24, self.top_k * 6),
                document_top_k=self.document_top_k,
            )
        except Exception as exc:
            self._debug(session_id, f"retrieve_semantic_failed reason={exc}")
            raise RuntimeError(f"semantic retrieval failed: {exc}") from exc

    def _ensure_semantic_index(self, session_id: str) -> None:
        """Ensure semantic index is loaded; build it once if persisted data is absent."""
        if self.rag_pipeline is None or self._semantic_ready:
            return

        try:
            self.rag_pipeline.ensure_loaded()
        except IncompatibleIndexError as exc:
            self._debug(session_id, f"index_rebuild reason={exc}")
            result = self.rag_pipeline.rebuild_indexes(memory_root=self.memory_root)
            self._debug(
                session_id,
                f"index_update memory_chunks={result['memory_chunks']} document_chunks={result['document_chunks']} total_chunks={result['total_chunks']}",
            )
            self._semantic_ready = True
            return
        if not self.rag_pipeline.store.has_persisted_index():
            result = self.rag_pipeline.rebuild_indexes(memory_root=self.memory_root)
            self._debug(
                session_id,
                f"index_update memory_chunks={result['memory_chunks']} document_chunks={result['document_chunks']} total_chunks={result['total_chunks']}",
            )
        else:
            removed = self.rag_pipeline.prune_invalid_memory_records(self.memory_root)
            if removed:
                self._debug(session_id, f"index_prune removed_chunks={removed}")
        self._semantic_ready = True

    def _merge_memory_hits(
        self,
        semantic_hits: list[RetrievalHit],
        records: list[dict],
        current_turn_ids: set[str],
        time_hints: dict[str, str | bool],
        session_id: str,
    ) -> list[MemoryHit]:
        """Map BGE-M3 chunk hits to memory records, then filter prompt noise."""
        candidates: dict[str, MemoryHit] = {}
        record_by_id = {str(record.get("id", "")): record for record in records}

        for hit in semantic_hits:
            key = hit.source_id or hit.chunk_id
            if not key:
                continue
            record = record_by_id.get(key, {})
            if (hit.source_type or "memory") == "memory" and not record:
                continue
            normalized = normalize_memorybank_fields(record) if record else {}
            source_message_ids = [hit.source_message_id] if hit.source_message_id else list(record.get("source_message_ids", []))
            retention = float(normalized.get("retention", record.get("retention", 1.0) if record else 1.0) or 1.0)
            score = hit.score * retention
            components: dict[str, float | str | bool] = {
                "route": "semantic",
                "semantic_raw_score": round(hit.score, 4),
                "retention": round(retention, 4),
                "semantic_retained_score": round(score, 4),
            }
            score, temporal_components = self._apply_time_score_components(
                score,
                record or self._record_from_semantic_hit(key, hit),
                time_hints,
                session_id,
            )
            components.update(temporal_components)
            self._store_memory_candidate(
                candidates,
                MemoryHit(
                    id=key,
                    summary=(
                        hit.text
                        if str(record.get("type", "")) == "conversation_turn"
                        else str(record.get("summary") or hit.text)
                    ),
                    raw_excerpt=(
                        hit.text
                        if str(record.get("type", "")) == "conversation_turn"
                        else str(record.get("raw_excerpt") or hit.text)
                    ),
                    chunk_id=hit.chunk_id,
                    score=round(score, 4),
                    memory_type=str(normalized.get("type") or hit.source_type or "episodic"),
                    source_session_id=hit.source_session_id or str(record.get("source_session_id", "")),
                    source_message_ids=[str(item) for item in source_message_ids if str(item)],
                    created_at=hit.created_at or str(record.get("created_at", "")),
                    last_recalled_at=str(normalized.get("last_recalled_at", "")),
                    memory_strength=int(normalized.get("memory_strength", 1) or 1),
                    retention=retention,
                    importance=float(normalized.get("importance", 0.0) or 0.0),
                    confidence=float(normalized.get("confidence", 0.0) or 0.0),
                    day=str(record.get("day", "")),
                    valid_at=str(record.get("valid_at", "")),
                    invalid_at=str(record.get("invalid_at", "")),
                    time_source=str(record.get("time_source", "")),
                    semantic_score=round(score, 4),
                    score_source="semantic",
                    score_components=components,
                ),
            )

        merged = []
        for hit in candidates.values():
            if self._is_current_turn_hit(hit, current_turn_ids):
                self._weaken_current_turn_hit(hit)
            merged.append(hit)

        merged.sort(key=lambda item: item.score, reverse=True)
        return self._filter_memory_hits(merged)

    def _dedupe_semantic_event_overlaps(self, hits: list[MemoryHit], query_plan: _QueryPlan) -> list[MemoryHit]:
        if query_plan.route != "date_window":
            return hits

        fact_message_ids: set[str] = set()
        for hit in hits:
            if hit.memory_type == "semantic_event":
                fact_message_ids.update(hit.source_message_ids)
        if not fact_message_ids:
            return hits

        deduped: list[MemoryHit] = []
        for hit in hits:
            if hit.memory_type in {"event_summary", "conversation_turn"} and fact_message_ids.intersection(hit.source_message_ids):
                continue
            deduped.append(hit)
        return deduped

    def _record_from_semantic_hit(self, key: str, hit: RetrievalHit) -> dict:
        return {
            "id": key,
            "type": hit.source_type or "episodic",
            "summary": hit.text,
            "raw_excerpt": hit.text,
            "source_session_id": hit.source_session_id,
            "source_message_ids": [hit.source_message_id] if hit.source_message_id else [],
            "created_at": hit.created_at,
            "day": self._day_bucket(hit.created_at) if hit.created_at else "",
            "valid_at": hit.created_at,
            "invalid_at": "",
            "time_source": "semantic_hit_created_at",
        }

    def _store_memory_candidate(self, candidates: dict[str, MemoryHit], hit: MemoryHit) -> None:
        existing = candidates.get(hit.id)
        if existing is None:
            candidates[hit.id] = hit
            return

        keep = hit if hit.score > existing.score else existing
        candidates[hit.id] = keep
        keep.semantic_score = max(existing.semantic_score, hit.semantic_score)
        keep.source_message_ids = list(dict.fromkeys(existing.source_message_ids + hit.source_message_ids))
        keep.day = keep.day or existing.day or hit.day
        keep.valid_at = keep.valid_at or existing.valid_at or hit.valid_at
        keep.invalid_at = keep.invalid_at or existing.invalid_at or hit.invalid_at
        keep.time_source = keep.time_source or existing.time_source or hit.time_source

    def _weaken_current_turn_hit(self, hit: MemoryHit) -> None:
        hit.is_current_turn = True
        original_score = hit.score
        hit.score = round(hit.score * self.current_turn_score_factor, 4)
        hit.semantic_score = round(hit.semantic_score * self.current_turn_score_factor, 4)
        hit.score_components = dict(hit.score_components)
        hit.score_components["current_turn_factor"] = round(self.current_turn_score_factor, 4)
        hit.score_components["score_before_current_turn_factor"] = round(original_score, 4)
        hit.score_components["final_score"] = hit.score

    def _filter_memory_hits(self, hits: list[MemoryHit]) -> list[MemoryHit]:
        historical = [hit for hit in hits if not hit.is_current_turn and hit.score >= self.memory_min_score]
        current_hits = [hit for hit in hits if hit.is_current_turn and hit.score >= self.memory_min_score]
        if historical:
            selected: list[MemoryHit] = []
            selected_ids: set[str] = set()
            for memory_type in ("semantic_event", "episodic", "event_summary", "conversation_turn"):
                candidate = next((hit for hit in historical if hit.memory_type == memory_type), None)
                if candidate is not None:
                    selected.append(candidate)
                    selected_ids.add(candidate.id)
                if len(selected) >= self.top_k:
                    return selected
            for hit in historical:
                if hit.id in selected_ids:
                    continue
                selected.append(hit)
                selected_ids.add(hit.id)
                if len(selected) >= self.top_k:
                    break
            for hit in current_hits:
                if hit.id in selected_ids or len(selected) >= self.top_k:
                    continue
                selected.append(hit)
                selected_ids.add(hit.id)
            return selected
        fallback_current = current_hits or [hit for hit in hits if hit.is_current_turn]
        return fallback_current[:1]

    def _current_turn_ids(self, session_id: str) -> set[str]:
        with self._turn_state_lock:
            state = dict(self._turn_state.get(session_id, {}))
        return {str(value) for key in ("user_message_id", "session_memory_id") if (value := state.get(key))}

    def _is_current_turn_hit(self, hit: MemoryHit, current_turn_ids: set[str]) -> bool:
        if not current_turn_ids:
            return False
        return hit.id in current_turn_ids or any(message_id in current_turn_ids for message_id in hit.source_message_ids)

    def _reinforce_recalled_memories(self, hits: list[MemoryHit], session_id: str, query: str, current_turn_ids: set[str]) -> None:
        if not self._should_reinforce_query(query):
            return
        recalled_ids = {
            hit.id
            for hit in hits
            if hit.id.startswith(("mem_", "evt_", "turn_", "fact_"))
            and not hit.is_current_turn
            and not self._is_current_turn_hit(hit, current_turn_ids)
            and hit.score >= self.reinforce_min_score
        }
        if not recalled_ids:
            return
        archival_changed = reinforce_jsonl_records(self.archival_file, recalled_ids)
        event_changed = reinforce_jsonl_records(self.event_summary_store.file_path, recalled_ids)
        session_changed = reinforce_jsonl_records(self.session_memory_file, recalled_ids)
        semantic_changed = reinforce_jsonl_records(self.semantic_event_file, recalled_ids)
        changed = archival_changed + event_changed + session_changed + semantic_changed
        if changed:
            self._debug(session_id, f"memory_update recalled_ids={changed} action=reinforce")

    def _to_document_hits(self, semantic_document_hits: list[RetrievalHit]) -> list[DocumentHit]:
        """Convert RetrievalHit rows into prompt-ready DocumentHit objects."""
        rows: list[DocumentHit] = []
        for hit in semantic_document_hits:
            rows.append(
                DocumentHit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    score=hit.score,
                    source_id=hit.source_id,
                    source_path=hit.source_path,
                    created_at=hit.created_at,
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[: self.document_top_k]

    def _load_archival_records(self) -> list[dict]:
        """Load and normalize active archival memory records from JSONL file."""
        records: list[dict] = []
        current_event_ids = active_event_ids(self.semantic_event_store.load_all())
        for path in (self.archival_file, self.event_summary_store.file_path, self.session_memory_file, self.semantic_event_file):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not uses_only_active_event_sources(payload, current_event_ids):
                    continue
                normalized = self._normalize_archival_record(payload)
                if normalized and normalized["status"] == "active":
                    records.append(normalized)
        return records

    def _load_session_message_records(self, session_id: str, limit: int = 8) -> list[dict]:
        """Expose recent raw session messages only for explicit recent-turn queries."""
        session_file = self.memory_root / "sessions" / f"{session_id}.jsonl"
        rows = self._read_session_rows(session_file)[-limit:]
        records: list[dict] = []
        for row in rows:
            message_id = str(row.get("id", ""))
            content = str(row.get("content", "")).strip()
            if not message_id or not content:
                continue
            role = str(row.get("role", "")).strip() or "message"
            created_at = str(row.get("created_at", ""))
            records.append(
                apply_temporal_fields(
                    {
                        "id": f"chat_{message_id}",
                        "type": "conversation_turn",
                        "summary": f"{role}: {content}",
                        "raw_excerpt": content,
                        "source_session_id": session_id,
                        "source_message_ids": [message_id],
                        "created_at": created_at,
                        "day": self._day_bucket(created_at) if created_at else "",
                        "last_recalled_at": created_at,
                        "memory_strength": 1,
                        "retention": 1.0,
                        "importance": 0.4,
                        "confidence": 1.0,
                        "status": "active",
                    },
                    text=content,
                    reference_time=created_at,
                )
            )
        return records

    def _build_query_plan(self, query: str, session_id: str, records: list[dict]) -> _QueryPlan:
        normalized = query.strip()
        day, source = self._extract_query_day_with_source(normalized, session_id)
        if day:
            return _QueryPlan(
                route="date_window",
                time_hints={"day": day},
                reason=source,
                day=day,
            )

        if self._mentions_recent_current(normalized):
            return _QueryPlan(
                route="recent_turn",
                time_hints={"recent_current_session": True},
                reason="recent_current_session_reference",
            )

        if self._mentions_last_session(normalized):
            preferred_session_id = self._latest_non_current_session_id(records, session_id)
            hints: dict[str, str | bool] = {}
            if preferred_session_id:
                hints["preferred_session_id"] = preferred_session_id
            return _QueryPlan(
                route="last_session",
                time_hints=hints,
                reason="last_session_reference",
                preferred_session_id=preferred_session_id,
            )

        if self._mentions_user_portrait(normalized):
            return _QueryPlan(route="portrait", time_hints={}, reason="long_term_user_profile")

        return _QueryPlan(route="hybrid", time_hints={}, reason="default_hybrid")

    def _records_for_query_plan(self, records: list[dict], query_plan: _QueryPlan, session_id: str) -> list[dict]:
        if query_plan.route == "date_window":
            candidates = [record for record in records if self._record_matches_day(record, query_plan.day)]
            if query_plan.reason in {"semantic_explicit_date", "contextual_date"}:
                return candidates
            if any(str(record.get("source_session_id", "")) != session_id for record in candidates):
                return candidates
            return records

        if query_plan.route == "recent_turn":
            return [record for record in records if str(record.get("source_session_id", "")) == session_id]

        if query_plan.route == "last_session":
            if not query_plan.preferred_session_id:
                return []
            return [record for record in records if str(record.get("source_session_id", "")) == query_plan.preferred_session_id]

        if query_plan.route == "portrait":
            return []

        return records

    def _record_matches_day(self, item: dict, day: str) -> bool:
        if not day:
            return False
        record_day = self._record_day(item)
        if record_day:
            return record_day == day
        created_at = str(item.get("created_at", "") or "")
        return bool(created_at and self._day_bucket(created_at) == day)

    def _mentions_user_portrait(self, query: str) -> bool:
        lower = query.lower()
        return any(
            token in lower
            for token in (
                "我是谁",
                "你觉得我是谁",
                "你了解我",
                "关于我",
                "我的画像",
                "用户画像",
                "我的偏好",
                "我喜欢什么",
                "who am i",
                "what do you know about me",
            )
        )

    def _extract_query_day_with_source(self, query: str, session_id: str) -> tuple[str, str]:
        normalized = query.strip()
        reference_time = self._date_context_by_session.get(session_id) or now_beijing()
        extracted = extract_semantic_time(normalized, reference_time=reference_time)
        if extracted is not None:
            day = extracted[0].date().isoformat()
            source = extracted[1]
        elif "那天" in normalized:
            day = self._date_context_by_session.get(session_id, "")
            source = "contextual_date" if day else ""
        else:
            day = ""
            source = ""
        if day:
            self._date_context_by_session[session_id] = day
        return day, source

    def _mentions_recent_current(self, query: str) -> bool:
        lower = query.lower()
        return any(token in lower for token in ("刚才", "上一轮", "前一轮", "刚刚", "just now", "previous turn"))

    def _mentions_last_session(self, query: str) -> bool:
        lower = query.lower()
        return any(token in lower for token in ("上次", "上一次", "上回", "last time", "previous chat", "last chat"))

    def _latest_non_current_session_id(self, records: list[dict], session_id: str) -> str:
        latest_session_id = ""
        latest_created_at = ""
        for record in records:
            candidate_session_id = str(record.get("source_session_id", ""))
            if not candidate_session_id or candidate_session_id == session_id:
                continue
            created_at = str(record.get("created_at", ""))
            if created_at >= latest_created_at:
                latest_created_at = created_at
                latest_session_id = candidate_session_id
        return latest_session_id

    def _apply_time_score_adjustments(
        self,
        score: float,
        item: dict,
        time_hints: dict[str, str | bool],
        session_id: str,
    ) -> float:
        adjusted, _ = self._apply_time_score_components(score, item, time_hints, session_id)
        return adjusted

    def _apply_time_score_components(
        self,
        score: float,
        item: dict,
        time_hints: dict[str, str | bool],
        session_id: str,
    ) -> tuple[float, dict[str, float | str | bool]]:
        adjusted = max(0.0, score)
        before = adjusted
        rule = "none"
        day = str(time_hints.get("day", "") or "")
        if day:
            record_day = self._record_day(item)
            if record_day == day:
                adjusted = max(adjusted, 0.45)
                rule = "date_match"
                if item.get("type") == "semantic_event":
                    adjusted += 0.25
                    rule = "date_match_semantic_event"
                if item.get("type") == "event_summary":
                    adjusted += 0.2
                    rule = "date_match_event_summary"
                if str(item.get("id", "")).startswith(f"evt_daily_{day.replace('-', '')}"):
                    adjusted += 0.15
                    rule = "date_match_daily_summary"
            elif record_day:
                adjusted *= 0.35
                rule = "date_mismatch_downweight"

        if bool(time_hints.get("recent_current_session")):
            if str(item.get("source_session_id", "")) == session_id:
                adjusted = max(adjusted, 0.35 + (0.25 * self._recency_score(str(item.get("created_at", "")))))
                rule = "recent_current_session"
            elif item.get("source_session_id"):
                adjusted *= 0.6
                rule = "recent_other_session_downweight"

        preferred_session_id = str(time_hints.get("preferred_session_id", "") or "")
        if preferred_session_id:
            if str(item.get("source_session_id", "")) == preferred_session_id:
                adjusted = max(adjusted, 0.55)
                rule = "last_session_match"
                if item.get("type") == "event_summary":
                    adjusted += 0.15
                    rule = "last_session_event_summary"
            elif item.get("source_session_id"):
                adjusted *= 0.5
                rule = "last_session_mismatch_downweight"

        components: dict[str, float | str | bool] = {
            "score_before_temporal": round(before, 4),
            "temporal_adjustment": round(adjusted - before, 4),
            "score_after_temporal": round(adjusted, 4),
            "temporal_rule": rule,
        }
        if day:
            components["query_day"] = day
            components["record_day"] = self._record_day(item)
        if preferred_session_id:
            components["preferred_session_id"] = preferred_session_id
        if bool(time_hints.get("recent_current_session")):
            components["recent_current_session"] = True
        return adjusted, components

    def _record_day(self, item: dict) -> str:
        if item.get("type") == "semantic_event":
            start_at = str(item.get("start_at", "") or "").strip()
            if start_at:
                return self._day_bucket(start_at)
        time_source = str(item.get("time_source", "") or "")
        if time_source.startswith("semantic_"):
            semantic_day = str(item.get("semantic_day", "") or "").strip()
            if semantic_day:
                return semantic_day
            valid_at = str(item.get("valid_at", "") or "").strip()
            if valid_at:
                return self._day_bucket(valid_at)
        day = str(item.get("day", "") or "").strip()
        if item.get("type") == "event_summary" and day:
            return day
        created_at = str(item.get("created_at", "") or "")
        return self._day_bucket(created_at) if item.get("type") == "event_summary" and created_at else ""

    def _apply_decay_policy(self, session_id: str) -> None:
        if not self.decay_enabled:
            return
        total_checked = 0
        total_decayed = 0
        for path in (self.archival_file, self.event_summary_store.file_path, self.session_memory_file, self.semantic_event_file):
            stats = refresh_jsonl_retention(
                path,
                decay_threshold=self.decay_threshold,
                min_age_days=self.decay_min_age_days,
            )
            total_checked += stats["checked"]
            total_decayed += stats["decayed"]
        if total_checked:
            self._debug(
                session_id,
                f"memory_decay enabled=1 checked={total_checked} decayed={total_decayed} threshold={self.decay_threshold:.4f} min_age_days={self.decay_min_age_days:.1f}",
            )

    def _build_session_memory(self, session_id: str, message_id: str, text: str, created_at: str | None = None) -> dict:
        now = created_at or now_beijing_iso()
        return apply_temporal_fields(
            {
                "id": f"turn_{message_id}",
                "type": "conversation_turn",
                "summary": text,
                "raw_excerpt": text,
                "importance": 0.5,
                "confidence": 0.9,
                "created_at": now,
                "last_recalled_at": now,
                "memory_strength": 1,
                "retention": 1.0,
                "source_session_id": session_id,
                "source_message_ids": [message_id],
                "status": "active",
            },
            text=text,
            reference_time=now,
        )

    def _read_session_rows(self, session_file: Path) -> list[dict]:
        rows: list[dict] = []
        if not session_file.exists():
            return rows
        for line in session_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _day_bucket(self, created_at: str) -> str:
        return time_day_bucket(created_at)

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _upsert_jsonl(self, path: Path, payload: dict) -> None:
        payload_id = str(payload.get("id", ""))
        rows = self._read_jsonl_records(path)
        for index, row in enumerate(rows):
            if payload_id and str(row.get("id", "")) == payload_id:
                rows[index] = payload
                break
        else:
            rows.append(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    def _load_memory_by_id(self, path: Path, memory_id: str) -> dict:
        if not path.exists() or not memory_id:
            return {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("id", "")) == memory_id:
                return payload
        return {}

    def _should_reinforce_query(self, query: str) -> bool:
        lower = query.lower()
        triggers = ["记得", "记住", "想起", "回忆", "remember", "recall", "what did", "do you know about me"]
        return any(trigger in lower for trigger in triggers)

    def _normalize_archival_record(self, payload: dict) -> dict | None:
        """Normalize one raw archival payload to internal scoring schema."""
        if payload.get("exclude_from_retrieval") is True:
            return None
        if str(payload.get("type", "")) == "system_feedback":
            return None

        summary = str(payload.get("summary") or payload.get("raw_excerpt") or "").strip()
        if not summary and str(payload.get("type", "")) == "semantic_event":
            description = str(payload.get("description", "")).strip()
            if description:
                start_at = str(payload.get("start_at", "") or "未知时间")
                summary = f"{description}（状态: {payload.get('status', 'active')}; 开始: {start_at}）"
        if not summary:
            return None

        source_message_ids = payload.get("source_message_ids", [])
        if isinstance(source_message_ids, str):
            source_message_ids = [source_message_ids]
        if not isinstance(source_message_ids, list):
            source_message_ids = []

        normalized = normalize_memorybank_fields(payload)
        valid_at = str(payload.get("valid_at") or payload.get("created_at", ""))
        return {
            "id": str(payload.get("id") or f"mem_{uuid4().hex[:10]}"),
            "type": str(payload.get("type", "episodic")),
            "summary": summary,
            "raw_excerpt": str(payload.get("raw_excerpt", "")),
            "source_session_id": str(payload.get("source_session_id", "")),
            "source_message_ids": [str(item) for item in source_message_ids if str(item)],
            "created_at": str(payload.get("created_at", "")),
            "updated_at": str(payload.get("updated_at", "")),
            "day": str(payload.get("day") or self._day_bucket(str(payload.get("created_at", "")))),
            "valid_at": valid_at,
            "invalid_at": str(payload.get("invalid_at", "")),
            "semantic_day": str(payload.get("semantic_day") or (time_day_bucket(valid_at) if valid_at else "")),
            "time_source": str(payload.get("time_source", "")),
            "last_recalled_at": str(normalized.get("last_recalled_at", "")),
            "memory_strength": int(normalized.get("memory_strength", 1) or 1),
            "retention": retention_score(normalized),
            "importance": float(payload.get("importance", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "status": str(payload.get("status", "active")),
        }

    def _env_float(self, name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

    def _env_bool(self, name: str, default: bool = False) -> bool:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on", "y"}

    def _recency_score(self, created_at: str) -> float:
        """Map timestamp recency to [0, 1] with linear decay over 30 days."""
        if not created_at:
            return 0.0
        try:
            ts = datetime.fromisoformat(created_at)
        except ValueError:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - ts.astimezone(timezone.utc)).total_seconds() / 86400
        if age_days <= 0:
            return 1.0
        if age_days >= 30:
            return 0.0
        return 1.0 - (age_days / 30)

    def _debug_pipeline(self, message: str) -> None:
        if not self.debug:
            return
        self._debug("rag", message)

    def _debug(self, session_id: str, message: str) -> None:
        if not self.debug:
            return
        output = f"[DEBUG][MemoryManager][{session_id}] {message}"
        if self.debug_sink is not None:
            self.debug_sink(output)
            return
        print(output)
