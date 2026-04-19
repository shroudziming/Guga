from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from guga.config import (
    DEFAULT_DOCUMENT_TOP_K,
    DEFAULT_MEMORY_RECENCY_WEIGHT,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_RAG_CHUNK_OVERLAP,
    DEFAULT_RAG_CHUNK_SIZE,
    DEFAULT_RAG_EMBEDDING_MODEL,
    DEFAULT_RAG_ENABLE_SEMANTIC,
)
from guga.memory.profile_store import ProfileStore
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
    """Local RAG memory manager for learning (memory + document retrieval)."""

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
        _ = model
        self.memory_root = memory_root or memory_data_dir()
        self.debug = debug
        self.debug_sink = debug_sink
        self.top_k = max(1, top_k)
        self.document_top_k = max(1, document_top_k)
        self.recency_weight = max(0.0, recency_weight)

        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.archival_file = self.memory_root / "archival_memory.jsonl"
        self.profile_store = ProfileStore(self.memory_root / "profile.json")
        self.session_store = _SessionStore(self.memory_root / "sessions")
        self._turn_state: dict[str, dict[str, str]] = {}

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
        started = perf_counter()
        records = self._load_archival_records()
        self._debug(
            session_id,
            f"retrieve_start query={json.dumps(user_text, ensure_ascii=False)} top_k={self.top_k} doc_top_k={self.document_top_k} candidates={len(records)}",
        )

        semantic_memory_hits, semantic_document_hits = self._retrieve_semantic(user_text=user_text, session_id=session_id)

        lexical_hits: list[MemoryHit] = []
        for record in records:
            score = self._score(record, user_text)
            if score <= 0:
                continue
            lexical_hits.append(self._to_hit(record, score))

        merged_memory_hits = self._merge_memory_hits(semantic_memory_hits, lexical_hits)
        document_hits = self._to_document_hits(semantic_document_hits)

        elapsed_ms = int((perf_counter() - started) * 1000)
        memory_hit_ids = [hit.id for hit in merged_memory_hits]
        doc_hit_ids = [hit.chunk_id for hit in document_hits]
        source_ids = [hit.source_session_id for hit in merged_memory_hits]
        self._debug(
            session_id,
            f"retrieve_done query={json.dumps(user_text, ensure_ascii=False)} top_k={self.top_k} doc_top_k={self.document_top_k} selected_mem={len(merged_memory_hits)} selected_doc={len(document_hits)} hit_ids={memory_hit_ids} doc_hit_ids={doc_hit_ids} source_ids={source_ids} latency_ms={elapsed_ms}",
        )

        return MemoryContext(
            archival_memories=[hit.summary for hit in merged_memory_hits],
            hits=merged_memory_hits,
            document_hits=document_hits,
        )

    def compose_system_prompt(self, base_prompt: str, memory_context: MemoryContext) -> str:
        sections = ["[Base Persona]", base_prompt]
        sections.append("\n[Relevant Memory]")
        if memory_context.hits:
            for hit in memory_context.hits:
                source_message = hit.source_message_ids[0] if hit.source_message_ids else ""
                source_ref = f"{hit.source_session_id}/{source_message}".strip("/")
                sections.append(f"- ({hit.id} | score={hit.score:.2f} | src={source_ref}) {hit.summary}")
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
        message_id = self.session_store.append_message(session_id=session_id, role="user", content=text, source=source)
        state = self._turn_state.setdefault(session_id, {})
        state["user_text"] = text
        state["user_message_id"] = message_id
        self._debug(session_id, f"ingest role=user message_id={message_id}")
        return message_id

    def record_assistant_message(self, session_id: str, text: str, source: str = "chat") -> str:
        message_id = self.session_store.append_message(session_id=session_id, role="assistant", content=text, source=source)
        state = self._turn_state.setdefault(session_id, {})
        state["assistant_text"] = text
        state["assistant_message_id"] = message_id
        self._debug(session_id, f"ingest role=assistant message_id={message_id}")
        return message_id

    def finalize_turn(self, session_id: str) -> None:
        started = perf_counter()
        state = self._turn_state.get(session_id, {})
        user_text = state.get("user_text", "").strip()
        if user_text and self._should_archive(user_text):
            payload = {
                "id": f"mem_{uuid4().hex[:10]}",
                "type": "episodic",
                "topic": "general",
                "summary": f"用户提到：{user_text}",
                "raw_excerpt": user_text,
                "importance": 0.7,
                "confidence": 0.7,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
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

        self._turn_state.pop(session_id, None)

    def rebuild_rag_indexes(self, session_id: str = "manual") -> dict[str, int]:
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

    def _merge_memory_hits(self, semantic_hits: list[RetrievalHit], lexical_hits: list[MemoryHit]) -> list[MemoryHit]:
        merged: list[MemoryHit] = []
        seen: set[str] = set()

        for hit in semantic_hits:
            key = hit.source_id or hit.chunk_id
            if key in seen:
                continue
            seen.add(key)
            source_message_ids = [hit.source_message_id] if hit.source_message_id else []
            merged.append(
                MemoryHit(
                    id=key,
                    summary=hit.text,
                    raw_excerpt=hit.text,
                    score=hit.score,
                    source_session_id=hit.source_session_id,
                    source_message_ids=source_message_ids,
                    created_at=hit.created_at,
                    importance=0.0,
                    confidence=0.0,
                )
            )

        lexical_hits.sort(key=lambda item: item.score, reverse=True)
        for hit in lexical_hits:
            if hit.id in seen:
                continue
            seen.add(hit.id)
            merged.append(hit)

        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[: self.top_k]

    def _to_document_hits(self, semantic_document_hits: list[RetrievalHit]) -> list[DocumentHit]:
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
        if len(user_text) >= 12:
            return True
        trigger_keywords = ["喜欢", "不喜欢", "工作", "焦虑", "压力", "我是", "我叫"]
        return any(keyword in user_text for keyword in trigger_keywords)

    def _load_archival_records(self) -> list[dict]:
        if not self.archival_file.exists():
            return []

        records: list[dict] = []
        for line in self.archival_file.read_text(encoding="utf-8").splitlines():
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

    def _normalize_archival_record(self, payload: dict) -> dict | None:
        summary = str(payload.get("summary") or payload.get("raw_excerpt") or "").strip()
        if not summary:
            return None

        source_message_ids = payload.get("source_message_ids", [])
        if isinstance(source_message_ids, str):
            source_message_ids = [source_message_ids]
        if not isinstance(source_message_ids, list):
            source_message_ids = []

        return {
            "id": str(payload.get("id") or f"mem_{uuid4().hex[:10]}"),
            "summary": summary,
            "raw_excerpt": str(payload.get("raw_excerpt", "")),
            "source_session_id": str(payload.get("source_session_id", "")),
            "source_message_ids": [str(item) for item in source_message_ids if str(item)],
            "created_at": str(payload.get("created_at", "")),
            "importance": float(payload.get("importance", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "status": str(payload.get("status", "active")),
        }

    def _to_hit(self, item: dict, score: float) -> MemoryHit:
        return MemoryHit(
            id=item["id"],
            summary=item["summary"],
            raw_excerpt=item["raw_excerpt"],
            score=round(score, 4),
            source_session_id=item["source_session_id"],
            source_message_ids=item["source_message_ids"],
            created_at=item["created_at"],
            importance=item["importance"],
            confidence=item["confidence"],
        )

    def _score(self, item: dict, query: str) -> float:
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

        return overlap_score + (self.recency_weight * recency_score) + (0.05 * importance) + (0.05 * confidence)

    def _recency_score(self, created_at: str) -> float:
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
