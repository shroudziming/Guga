from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from threading import RLock
from time import perf_counter
from uuid import uuid4

from guga.config import (
    DEFAULT_CURRENT_TURN_SCORE_FACTOR,
    DEFAULT_DOCUMENT_TOP_K,
    DEFAULT_MEMORY_MIN_SCORE,
    DEFAULT_MEMORY_RECENCY_WEIGHT,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_RAG_CHUNK_OVERLAP,
    DEFAULT_RAG_CHUNK_SIZE,
    DEFAULT_RAG_EMBEDDING_MODEL,
    DEFAULT_RAG_ENABLE_SEMANTIC,
)
from guga.memory.event_summary_store import EventSummaryStore
from guga.memory.forgetting import normalize_memorybank_fields, refresh_jsonl_retention, reinforce_jsonl_records, retention_score
from guga.memory.portrait import UserPortraitStore
from guga.memory.profile_store import ProfileStore
from guga.memory.summarizer import MemoryBankSummarizer
from guga.rag.pipeline import RagPipeline
from guga.rag.schemas import RetrievalHit
from guga.types import DocumentHit, MemoryContext, MemoryHit
from guga.utils.paths import memory_data_dir, rag_documents_dir


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
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
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
        model=None,
        debug: bool = False,
        debug_sink: Callable[[str], None] | None = None,
        top_k: int = DEFAULT_MEMORY_TOP_K,
        document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
        recency_weight: float = DEFAULT_MEMORY_RECENCY_WEIGHT,
        enable_semantic: bool = DEFAULT_RAG_ENABLE_SEMANTIC,
    ) -> None:
        """Create manager with lexical + optional semantic retrieval capabilities.

        Args:
            memory_root: Root directory for memory files and indexes.
            model: Reserved for future use (kept for API compatibility).
            debug: Whether to emit debug traces.
            debug_sink: Optional debug output sink callback.
            top_k: Max memory hits returned to prompt context.
            document_top_k: Max document hits returned to prompt context.
            recency_weight: Weight for recency term in lexical scoring.
            enable_semantic: Whether to enable vector-based retrieval pipeline.
        """
        self.model = model
        self.memory_root = memory_root or memory_data_dir()
        self.debug = debug
        self.debug_sink = debug_sink
        self.top_k = max(1, top_k)
        self.document_top_k = max(1, document_top_k)
        self.recency_weight = max(0.0, recency_weight)
        self.decay_threshold = 0.05
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
        self.archival_file = self.memory_root / "archival_memory.jsonl"
        self.session_memory_file = self.memory_root / "session_memories.jsonl"
        self.profile_store = ProfileStore(self.memory_root / "profile.json")
        self.event_summary_store = EventSummaryStore(self.memory_root / "event_summaries.jsonl")
        self.portrait_store = UserPortraitStore(self.memory_root / "profile.json", self.memory_root / "personality_insights.jsonl")
        self.summarizer = MemoryBankSummarizer(model=model)
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
                documents_dir=rag_documents_dir(),
                embedding_model=DEFAULT_RAG_EMBEDDING_MODEL,
                chunk_size=DEFAULT_RAG_CHUNK_SIZE,
                chunk_overlap=DEFAULT_RAG_CHUNK_OVERLAP,
                debug_hook=self._debug_pipeline,
            )

    def prepare_context(self, user_text: str, session_id: str) -> MemoryContext:
        """Retrieve context for current user input and return structured hits.

        Upstream:
            Called by ChatSession before model generation.

        Retrieval steps:
            1) Load archival records for lexical matching.
            2) Retrieve semantic hits from RagPipeline (memory + documents).
            3) Compute lexical scores and merge with semantic memory hits.

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
        if self._mentions_recent_current(user_text):
            records.extend(self._load_session_message_records(session_id))
        time_hints = self._build_time_hints(user_text, session_id, records)
        self._debug(
            session_id,
            f"retrieve_start query={json.dumps(user_text, ensure_ascii=False)} top_k={self.top_k} doc_top_k={self.document_top_k} candidates={len(records)} min_score={self.memory_min_score:.2f} current_turn_factor={self.current_turn_score_factor:.2f} time_hints={json.dumps(time_hints, ensure_ascii=False)}",
        )

        semantic_memory_hits, semantic_document_hits = self._retrieve_semantic(user_text=user_text, session_id=session_id)

        lexical_hits: list[MemoryHit] = []
        for record in records:
            score = self._score(record, user_text)
            score = self._apply_time_score_adjustments(score, record, time_hints, session_id)
            if score <= 0:
                continue
            lexical_hits.append(self._to_hit(record, score))

        merged_memory_hits = self._merge_memory_hits(
            semantic_memory_hits,
            lexical_hits,
            records,
            current_turn_ids=current_turn_ids,
            time_hints=time_hints,
            session_id=session_id,
        )
        self._reinforce_recalled_memories(merged_memory_hits, session_id=session_id, query=user_text, current_turn_ids=current_turn_ids)
        document_hits = self._to_document_hits(semantic_document_hits)
        user_portrait = str(self.portrait_store.load().get("portrait_summary", "")).strip()
        event_summary_hits = [hit for hit in merged_memory_hits if hit.memory_type == "event_summary"]

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
                "semantic_score": hit.semantic_score,
                "lexical_score": hit.lexical_score,
                "score_source": hit.score_source,
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
            event_summaries=event_summary_hits,
            user_portrait=user_portrait,
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
        sections.append("\n[User Portrait]")
        if memory_context.user_portrait:
            sections.append(memory_context.user_portrait)
        else:
            sections.append("- 当前还没有稳定用户画像。")

        sections.append("\n[Relevant Memory]")
        sections.append("\n[Relevant Event Summaries]")
        if memory_context.event_summaries:
            for hit in memory_context.event_summaries:
                source_message = hit.source_message_ids[0] if hit.source_message_ids else ""
                source_ref = f"{hit.source_session_id}/{source_message}".strip("/")
                sections.append(
                    f"- ({hit.id} | score={hit.score:.2f} | retention={hit.retention:.2f} | S={hit.memory_strength} | src={source_ref}) {hit.summary}"
                )
        else:
            sections.append("- 当前未检索到相关事件摘要。")

        sections.append("\n[Relevant Conversation Memories]")
        if memory_context.hits:
            has_conversation_memory = False
            for hit in memory_context.hits:
                if hit.memory_type == "event_summary":
                    continue
                has_conversation_memory = True
                source_message = hit.source_message_ids[0] if hit.source_message_ids else ""
                source_ref = f"{hit.source_session_id}/{source_message}".strip("/")
                sections.append(
                    f"- ({hit.id} | score={hit.score:.2f} | retention={hit.retention:.2f} | S={hit.memory_strength} | src={source_ref}) {hit.summary}"
                )
            if not has_conversation_memory:
                sections.append("- 当前未检索到可靠历史记忆。")
        else:
            sections.append("- 当前未检索到可靠历史记忆。")

        sections.append("\n[Relevant Documents]")
        if memory_context.document_hits:
            for hit in memory_context.document_hits:
                source_ref = hit.source_id or hit.source_path
                sections.append(f"- ({hit.chunk_id} | score={hit.score:.2f} | src={source_ref}) {hit.text}")
        else:
            sections.append("- 当前未检索到相关文档片段。")

        sections.append("\n[Current Rule]")
        sections.append("请仅在相关时自然使用记忆和文档，不要机械复述。")
        sections.append("若未命中相关信息，请直接说明没有找到相关历史信息，不要编造。")
        return "\n".join(sections)

    def record_user_message(self, session_id: str, text: str, source: str = "chat") -> str:
        """Persist current user message and cache it in per-turn state.

        Returns:
            message_id written to sessions/<session_id>.jsonl.
        """
        message_id = self.session_store.append_message(session_id=session_id, role="user", content=text, source=source)
        turn_payload = self._build_session_memory(session_id=session_id, message_id=message_id, text=text)
        self._append_jsonl(self.session_memory_file, turn_payload)
        with self._turn_state_lock:
            state = self._turn_state.setdefault(session_id, {})
            state["user_text"] = text
            state["user_message_id"] = message_id
            state["session_memory_id"] = turn_payload["id"]
        self._debug(session_id, f"ingest role=user message_id={message_id}")
        return message_id

    def record_assistant_message(self, session_id: str, text: str, source: str = "chat") -> str:
        """Persist assistant reply and cache it in per-turn state.

        Returns:
            message_id written to sessions/<session_id>.jsonl.
        """
        message_id = self.session_store.append_message(session_id=session_id, role="assistant", content=text, source=source)
        with self._turn_state_lock:
            state = self._turn_state.setdefault(session_id, {})
            state["assistant_text"] = text
            state["assistant_message_id"] = message_id
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

    def _pop_turn_state(self, session_id: str) -> dict[str, str]:
        with self._turn_state_lock:
            return dict(self._turn_state.pop(session_id, {}))

    def _finalize_turn_state(self, session_id: str, state: dict[str, str]) -> None:
        started = perf_counter()
        user_text = state.get("user_text", "").strip()
        assistant_text = state.get("assistant_text", "").strip()
        session_memory_payload = self._load_memory_by_id(self.session_memory_file, state.get("session_memory_id", ""))
        memory_candidate = self.summarizer.extract_archival_memory(user_text=user_text, assistant_text=assistant_text) if user_text else {}
        should_archive = bool(memory_candidate.get("should_archive")) if memory_candidate else False
        if user_text and (should_archive or self._should_archive(user_text)):
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            payload = {
                "id": f"mem_{uuid4().hex[:10]}",
                "type": "episodic",
                "topic": str(memory_candidate.get("topic") or "general"),
                "summary": str(memory_candidate.get("summary") or f"用户提到：{user_text}"),
                "raw_excerpt": user_text,
                "importance": float(memory_candidate.get("importance", 0.7) or 0.7),
                "confidence": float(memory_candidate.get("confidence", 0.7) or 0.7),
                "created_at": now,
                "last_recalled_at": now,
                "memory_strength": 1,
                "retention": 1.0,
                "source_session_id": session_id,
                "source_message_ids": [state.get("user_message_id", "")],
                "status": "active",
            }
            with self.archival_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if self.rag_pipeline is not None:
                try:
                    self._ensure_semantic_index(session_id)
                    self.rag_pipeline.add_memory_record(payload)
                except Exception as exc:
                    self._debug(session_id, f"index_update status=failed reason={exc}")

            elapsed_ms = int((perf_counter() - started) * 1000)
            self._debug(
                session_id,
                f"writeback status=archival_added memory_id={payload['id']} source_ids={payload['source_message_ids']} latency_ms={elapsed_ms}",
            )
        else:
            elapsed_ms = int((perf_counter() - started) * 1000)
            self._debug(session_id, f"writeback status=no_archival_write latency_ms={elapsed_ms}")

        if session_memory_payload and self.rag_pipeline is not None:
            try:
                self.rag_pipeline.add_memory_record(session_memory_payload)
            except Exception as exc:
                self._debug(session_id, f"index_update status=session_memory_failed reason={exc}")
        if user_text:
            self._refresh_hierarchical_memory(session_id=session_id)

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
                memory_top_k=self.top_k,
                document_top_k=self.document_top_k,
            )
        except Exception as exc:
            self._debug(session_id, f"retrieve_semantic_failed reason={exc}")
            return [], []

    def _ensure_semantic_index(self, session_id: str) -> None:
        """Ensure semantic index is loaded; build it once if persisted data is absent."""
        if self.rag_pipeline is None or self._semantic_ready:
            return

        self.rag_pipeline.ensure_loaded()
        if not self.rag_pipeline.store.has_persisted_index():
            result = self.rag_pipeline.rebuild_indexes(memory_root=self.memory_root)
            self._debug(
                session_id,
                f"index_update memory_chunks={result['memory_chunks']} document_chunks={result['document_chunks']} total_chunks={result['total_chunks']}",
            )
        self._semantic_ready = True

    def _merge_memory_hits(
        self,
        semantic_hits: list[RetrievalHit],
        lexical_hits: list[MemoryHit],
        records: list[dict],
        current_turn_ids: set[str],
        time_hints: dict[str, str | bool],
        session_id: str,
    ) -> list[MemoryHit]:
        """Merge semantic and lexical routes by id, then filter prompt noise."""
        candidates: dict[str, MemoryHit] = {}
        record_by_id = {str(record.get("id", "")): record for record in records}

        for hit in semantic_hits:
            key = hit.source_id or hit.chunk_id
            if not key:
                continue
            record = record_by_id.get(key, {})
            normalized = normalize_memorybank_fields(record) if record else {}
            source_message_ids = [hit.source_message_id] if hit.source_message_id else list(record.get("source_message_ids", []))
            retention = float(normalized.get("retention", record.get("retention", 1.0) if record else 1.0) or 1.0)
            score = hit.score * retention
            score = self._apply_time_score_adjustments(score, record or self._record_from_semantic_hit(key, hit), time_hints, session_id)
            self._store_memory_candidate(
                candidates,
                MemoryHit(
                    id=key,
                    summary=str(record.get("summary") or hit.text),
                    raw_excerpt=str(record.get("raw_excerpt") or hit.text),
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
                    semantic_score=round(score, 4),
                    score_source="semantic",
                ),
            )

        lexical_hits.sort(key=lambda item: item.score, reverse=True)
        for hit in lexical_hits:
            self._store_memory_candidate(candidates, hit)

        merged = []
        for hit in candidates.values():
            self._finalize_route_scores(hit)
            if self._is_current_turn_hit(hit, current_turn_ids):
                self._weaken_current_turn_hit(hit)
            merged.append(hit)

        merged.sort(key=lambda item: item.score, reverse=True)
        return self._filter_memory_hits(merged)

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
        }

    def _store_memory_candidate(self, candidates: dict[str, MemoryHit], hit: MemoryHit) -> None:
        existing = candidates.get(hit.id)
        if existing is None:
            candidates[hit.id] = hit
            return

        semantic_score = max(existing.semantic_score, hit.semantic_score)
        lexical_score = max(existing.lexical_score, hit.lexical_score)
        keep = hit if hit.score > existing.score else existing
        candidates[hit.id] = keep
        keep.semantic_score = semantic_score
        keep.lexical_score = lexical_score
        keep.source_message_ids = list(dict.fromkeys(existing.source_message_ids + hit.source_message_ids))
        keep.day = keep.day or existing.day or hit.day

    def _finalize_route_scores(self, hit: MemoryHit) -> None:
        semantic_score = round(max(0.0, hit.semantic_score), 4)
        lexical_score = round(max(0.0, hit.lexical_score), 4)
        hit.semantic_score = semantic_score
        hit.lexical_score = lexical_score
        if semantic_score <= 0 and lexical_score <= 0:
            hit.score = round(max(0.0, hit.score), 4)
            return
        hit.score = max(semantic_score, lexical_score)
        if semantic_score > 0 and lexical_score > 0 and semantic_score == lexical_score:
            hit.score_source = "semantic+lexical"
        elif lexical_score >= semantic_score:
            hit.score_source = "lexical"
        else:
            hit.score_source = "semantic"

    def _weaken_current_turn_hit(self, hit: MemoryHit) -> None:
        hit.is_current_turn = True
        hit.score = round(hit.score * self.current_turn_score_factor, 4)
        hit.semantic_score = round(hit.semantic_score * self.current_turn_score_factor, 4)
        hit.lexical_score = round(hit.lexical_score * self.current_turn_score_factor, 4)

    def _filter_memory_hits(self, hits: list[MemoryHit]) -> list[MemoryHit]:
        eligible = [hit for hit in hits if hit.score >= self.memory_min_score]
        if eligible:
            return eligible[: self.top_k]
        current_hits = [hit for hit in hits if hit.is_current_turn]
        return current_hits[:1]

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
            if hit.id.startswith(("mem_", "evt_", "turn_"))
            and not hit.is_current_turn
            and not self._is_current_turn_hit(hit, current_turn_ids)
            and hit.score >= self.reinforce_min_score
        }
        if not recalled_ids:
            return
        archival_changed = reinforce_jsonl_records(self.archival_file, recalled_ids)
        event_changed = reinforce_jsonl_records(self.event_summary_store.file_path, recalled_ids)
        session_changed = reinforce_jsonl_records(self.session_memory_file, recalled_ids)
        changed = archival_changed + event_changed + session_changed
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

    def _should_archive(self, user_text: str) -> bool:
        """Heuristic writeback policy deciding whether user text becomes memory."""
        if len(user_text) >= 12:
            return True
        trigger_keywords = ["喜欢", "不喜欢", "工作", "焦虑", "压力", "我是", "我叫"]
        return any(keyword in user_text for keyword in trigger_keywords)

    def _load_archival_records(self) -> list[dict]:
        """Load and normalize active archival memory records from JSONL file."""
        records: list[dict] = []
        for path in (self.archival_file, self.event_summary_store.file_path, self.session_memory_file):
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
                }
            )
        return records

    def _build_time_hints(self, query: str, session_id: str, records: list[dict]) -> dict[str, str | bool]:
        hints: dict[str, str | bool] = {}
        day = self._extract_query_day(query, session_id)
        if day:
            hints["day"] = day
        if self._mentions_recent_current(query):
            hints["recent_current_session"] = True
        if self._mentions_last_session(query):
            preferred_session_id = self._latest_non_current_session_id(records, session_id)
            if preferred_session_id:
                hints["preferred_session_id"] = preferred_session_id
        return hints

    def _extract_query_day(self, query: str, session_id: str) -> str:
        normalized = query.strip()
        iso_match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", normalized)
        if iso_match:
            day = self._safe_iso_day(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
            if day:
                self._date_context_by_session[session_id] = day
            return day

        cn_year_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})(?:日|号)?", normalized)
        if cn_year_match:
            day = self._safe_iso_day(int(cn_year_match.group(1)), int(cn_year_match.group(2)), int(cn_year_match.group(3)))
            if day:
                self._date_context_by_session[session_id] = day
            return day

        month_day_match = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})(?:日|号)?", normalized)
        if month_day_match:
            year = datetime.now().astimezone().year
            day = self._safe_iso_day(year, int(month_day_match.group(1)), int(month_day_match.group(2)))
            if day:
                self._date_context_by_session[session_id] = day
            return day

        today = datetime.now().astimezone().date()
        if "前天" in normalized:
            day = (today - timedelta(days=2)).isoformat()
        elif "昨天" in normalized:
            day = (today - timedelta(days=1)).isoformat()
        elif "今天" in normalized:
            day = today.isoformat()
        elif "那天" in normalized:
            day = self._date_context_by_session.get(session_id, "")
        else:
            day = ""
        if day:
            self._date_context_by_session[session_id] = day
        return day

    def _safe_iso_day(self, year: int, month: int, day: int) -> str:
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return ""

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
        adjusted = max(0.0, score)
        day = str(time_hints.get("day", "") or "")
        if day:
            record_day = self._record_day(item)
            if record_day == day:
                adjusted = max(adjusted, 0.45)
                if item.get("type") == "event_summary":
                    adjusted += 0.2
                if str(item.get("id", "")).startswith(f"evt_daily_{day.replace('-', '')}"):
                    adjusted += 0.15
            elif record_day:
                adjusted *= 0.35

        if bool(time_hints.get("recent_current_session")):
            if str(item.get("source_session_id", "")) == session_id:
                adjusted = max(adjusted, 0.35 + (0.25 * self._recency_score(str(item.get("created_at", "")))))
            elif item.get("source_session_id"):
                adjusted *= 0.6

        preferred_session_id = str(time_hints.get("preferred_session_id", "") or "")
        if preferred_session_id:
            if str(item.get("source_session_id", "")) == preferred_session_id:
                adjusted = max(adjusted, 0.55)
                if item.get("type") == "event_summary":
                    adjusted += 0.15
            elif item.get("source_session_id"):
                adjusted *= 0.5

        return adjusted

    def _record_day(self, item: dict) -> str:
        day = str(item.get("day", "") or "").strip()
        if day:
            return day
        created_at = str(item.get("created_at", "") or "")
        return self._day_bucket(created_at) if created_at else ""

    def _apply_decay_policy(self, session_id: str) -> None:
        total_checked = 0
        total_decayed = 0
        for path in (self.archival_file, self.event_summary_store.file_path, self.session_memory_file):
            stats = refresh_jsonl_retention(path, decay_threshold=self.decay_threshold)
            total_checked += stats["checked"]
            total_decayed += stats["decayed"]
        if total_checked:
            self._debug(session_id, f"memory_decay checked={total_checked} decayed={total_decayed}")

    def _build_session_memory(self, session_id: str, message_id: str, text: str) -> dict:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        return {
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
        }

    def _refresh_hierarchical_memory(self, session_id: str) -> None:
        session_file = self.memory_root / "sessions" / f"{session_id}.jsonl"
        day, dialogue, message_ids = self._load_daily_dialogue(session_file)
        if not dialogue:
            return
        daily_event = self.event_summary_store.refresh_daily_summary(
            session_id=session_id,
            day=day,
            dialogue=dialogue,
            source_message_ids=message_ids,
            summarizer=self.summarizer,
        )
        global_event = self.event_summary_store.refresh_global_summary(self.summarizer)
        daily_portrait = self.portrait_store.refresh_daily_insight(
            day=day,
            dialogue=dialogue,
            source_session_id=session_id,
            source_message_ids=message_ids,
            summarizer=self.summarizer,
        )
        profile = self.portrait_store.refresh_global_portrait(self.summarizer)
        if self.rag_pipeline is not None:
            for payload in (daily_event, global_event):
                if payload:
                    try:
                        self.rag_pipeline.add_memory_record(payload)
                    except Exception as exc:
                        self._debug(session_id, f"index_update status=hierarchy_failed id={payload.get('id', '')} reason={exc}")
        self._debug(
            session_id,
            f"hierarchy_update daily_event_id={daily_event.get('id', '') if daily_event else ''} global_event_id={global_event.get('id', '') if global_event else ''} daily_portrait_id={daily_portrait.get('id', '') if daily_portrait else ''} portrait_len={len(str(profile.get('portrait_summary', '')))}",
        )

    def _load_daily_dialogue(self, session_file: Path) -> tuple[str, str, list[str]]:
        if not session_file.exists():
            return datetime.now().astimezone().date().isoformat(), "", []

        current_rows = self._read_session_rows(session_file)
        if not current_rows:
            return datetime.now().astimezone().date().isoformat(), "", []

        day = self._day_bucket(str(current_rows[-1].get("created_at", "")))
        rows: list[dict] = []
        sessions_dir = self.memory_root / "sessions"
        if sessions_dir.exists():
            for path in sessions_dir.glob("*.jsonl"):
                rows.extend(row for row in self._read_session_rows(path) if self._day_bucket(str(row.get("created_at", ""))) == day)
        rows.sort(key=lambda row: str(row.get("created_at", "")))

        message_ids = [str(row.get("id", "")) for row in rows if str(row.get("id", ""))]
        lines = []
        for row in rows:
            role = str(row.get("role", "")).strip() or "message"
            content = str(row.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return day, "\n".join(lines), message_ids

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
        try:
            return datetime.fromisoformat(created_at).date().isoformat()
        except ValueError:
            return datetime.now().astimezone().date().isoformat()

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
        summary = str(payload.get("summary") or payload.get("raw_excerpt") or "").strip()
        if not summary:
            return None

        source_message_ids = payload.get("source_message_ids", [])
        if isinstance(source_message_ids, str):
            source_message_ids = [source_message_ids]
        if not isinstance(source_message_ids, list):
            source_message_ids = []

        normalized = normalize_memorybank_fields(payload)
        return {
            "id": str(payload.get("id") or f"mem_{uuid4().hex[:10]}"),
            "type": str(payload.get("type", "episodic")),
            "summary": summary,
            "raw_excerpt": str(payload.get("raw_excerpt", "")),
            "source_session_id": str(payload.get("source_session_id", "")),
            "source_message_ids": [str(item) for item in source_message_ids if str(item)],
            "created_at": str(payload.get("created_at", "")),
            "day": str(payload.get("day") or self._day_bucket(str(payload.get("created_at", "")))),
            "last_recalled_at": str(normalized.get("last_recalled_at", "")),
            "memory_strength": int(normalized.get("memory_strength", 1) or 1),
            "retention": retention_score(normalized),
            "importance": float(payload.get("importance", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "status": str(payload.get("status", "active")),
        }

    def _to_hit(self, item: dict, score: float) -> MemoryHit:
        """Convert normalized archival dict to MemoryHit with rounded score."""
        return MemoryHit(
            id=item["id"],
            summary=item["summary"],
            raw_excerpt=item["raw_excerpt"],
            score=round(score, 4),
            memory_type=item["type"],
            source_session_id=item["source_session_id"],
            source_message_ids=item["source_message_ids"],
            created_at=item["created_at"],
            last_recalled_at=item["last_recalled_at"],
            memory_strength=item["memory_strength"],
            retention=item["retention"],
            importance=item["importance"],
            confidence=item["confidence"],
            day=item.get("day", ""),
            lexical_score=round(score, 4),
            score_source="lexical",
        )

    def _score(self, item: dict, query: str) -> float:
        """Compute lexical relevance score for one archival memory record.

        Score terms:
            overlap(query_tokens, memory_text) + recency term + small priors
            for importance/confidence.
        """
        text = f"{item.get('summary', '')} {item.get('raw_excerpt', '')}"
        query_tokens = self._tokens(query)
        if not query_tokens:
            return 0.0

        matches = sum(1 for token in query_tokens if token in text)
        overlap_score = matches / max(1, len(query_tokens))
        if overlap_score <= 0:
            return 0.0

        recency_score = self._recency_score(item.get("created_at", ""))
        importance = max(0.0, min(float(item.get("importance", 0.0) or 0.0), 1.0))
        confidence = max(0.0, min(float(item.get("confidence", 0.0) or 0.0), 1.0))

        retention = max(0.0, min(float(item.get("retention", 1.0) or 1.0), 1.0))
        return (
            (overlap_score * retention)
            + (self.recency_weight * recency_score)
            + (0.05 * importance)
            + (0.05 * confidence)
        )

    def _env_float(self, name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

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

    def _tokens(self, text: str) -> list[str]:
        """Tokenize query for lexical matching (word split + CJK n-gram fallback)."""
        compact = text.replace("，", " ").replace("。", " ").replace("？", " ").strip()
        pieces = [item for item in compact.split() if item]
        if len(pieces) >= 2:
            return pieces[:12]

        if len(pieces) == 1:
            piece = pieces[0]
            if piece.isascii():
                return [piece]
            return self._char_ngrams(piece)

        compact = compact.replace(" ", "")
        if len(compact) < 2:
            return [compact] if compact else []
        return self._char_ngrams(compact)

    def _char_ngrams(self, text: str) -> list[str]:
        """Build 2/3-gram token list for short CJK/ascii text matching."""
        if len(text) < 2:
            return [text] if text else []

        tokens: list[str] = []
        for size in (2, 3):
            for index in range(0, max(0, len(text) - size + 1)):
                tokens.append(text[index : index + size])
        return tokens[:20]

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
