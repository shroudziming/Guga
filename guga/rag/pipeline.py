from __future__ import annotations

import hashlib
import json
from pathlib import Path

from guga.memory_source_validity import active_event_ids, uses_only_active_event_sources
from guga.rag.chunker import chunk_text
from guga.rag.embedder import BaseEmbedder, build_embedder
from guga.rag.faiss_store import VectorStore
from guga.rag.schemas import DocumentChunk, RetrievalHit


class RagPipeline:
    """Build and query vector indexes for memory/documents in the RAG flow.

    Upstream caller is MemoryManager:
    - prepare_context -> retrieve
    - finalize_turn -> add_memory_record
    - manual command -> rebuild_indexes
    """

    def __init__(
        self,
        index_dir: Path,
        documents_dir: Path,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        debug_hook=None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        """Initialize embedder + vector store configuration.

        Args:
            index_dir: Persisted index directory (chunks + vectors).
            documents_dir: Root directory of external/local documents.
            embedding_model: Embedding model name for sentence-transformers.
            chunk_size: Max chars per chunk.
            chunk_overlap: Overlap chars between adjacent chunks.
            debug_hook: Optional callback used for debug logs.
            embedder: Optional injected embedder for tests/custom behavior.
        """
        self.index_dir = index_dir
        self.documents_dir = documents_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug_hook = debug_hook
        self.embedder = embedder or build_embedder(embedding_model)
        self.store = VectorStore(index_dir, embedding_model=embedding_model)
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Load persisted vector/chunk index into memory once per process."""
        if self._loaded:
            return
        self.store.load()
        self._loaded = True

    def rebuild_indexes(self, memory_root: Path, documents_dir: Path | None = None) -> dict[str, int]:
        """Rebuild full index from memory files and document directory.

        Data sources:
            - memory_root/archival_memory.jsonl
            - memory_root/event_summaries.jsonl
            - memory_root/session_memories.jsonl
            - memory_root/semantic_events.jsonl
            - memory_root/sessions/**/*.jsonl (legacy fallback user messages)
            - documents_dir/**/*.txt|md|json|jsonl

        Returns:
            Chunk counters for memory/document/total.
        """
        target_docs = documents_dir or self.documents_dir
        memory_chunks = self._collect_memory_chunks(memory_root)
        document_chunks = self._collect_document_chunks(target_docs)
        all_chunks = memory_chunks + document_chunks

        if all_chunks:
            vectors = self.embedder.encode([chunk.text for chunk in all_chunks])
        else:
            vectors = []

        self.store.rebuild(all_chunks, vectors)
        self.store.save()
        self._loaded = True
        self._debug(
            f"index_update memory_chunks={len(memory_chunks)} document_chunks={len(document_chunks)} total_chunks={len(all_chunks)}"
        )
        return {
            "memory_chunks": len(memory_chunks),
            "document_chunks": len(document_chunks),
            "total_chunks": len(all_chunks),
        }

    def add_memory_record(self, payload: dict, active_event_ids: set[str] | None = None) -> None:
        """Incrementally append one new memory record into vector index."""
        self.ensure_loaded()
        source_id = str(payload.get("id", ""))
        chunks = self._memory_chunks_from_payload(payload, active_event_ids=active_event_ids)
        vectors = self.embedder.encode([chunk.text for chunk in chunks]) if chunks else []
        if source_id:
            self.store.replace_by_source_id(source_id, chunks, vectors)
        elif chunks:
            self.store.add(chunks, vectors)
        self.store.save()

    def prune_invalid_memory_records(self, memory_root: Path) -> int:
        self.ensure_loaded()
        valid_source_ids = {chunk.source_id for chunk in self._collect_memory_chunks(memory_root)}
        removed = self.store.prune_memory_sources(valid_source_ids)
        if removed:
            self.store.save()
        return removed

    def retrieve(self, query: str, memory_top_k: int, document_top_k: int) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
        """Retrieve semantic hits for memory and documents with one query.

        Args:
            query: User query text for semantic retrieval.
            memory_top_k: Max memory results.
            document_top_k: Max document results.

        Returns:
            Tuple: (memory_hits, document_hits).
        """
        self.ensure_loaded()
        if not query.strip() or not self.store.chunks:
            return [], []

        query_vec = self.embedder.encode([query])[0]
        memory_hits = self._search(query_vec, top_k=memory_top_k, source_type="memory")
        document_hits = self._search(query_vec, top_k=document_top_k, source_type="document")
        return memory_hits, document_hits

    def _search(self, query_vec: list[float], top_k: int, source_type: str) -> list[RetrievalHit]:
        """Search vectors by source_type and map rows into RetrievalHit objects."""
        rows = self.store.search(query_vec=query_vec, top_k=top_k, source_type=source_type)
        hits: list[RetrievalHit] = []
        for idx, score in rows:
            chunk = self.store.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=round(float(score), 4),
                    source_type=chunk.source_type,
                    source_id=chunk.source_id,
                    source_path=chunk.source_path,
                    source_session_id=chunk.source_session_id,
                    source_message_id=chunk.source_message_id,
                    created_at=chunk.created_at,
                )
            )
        return hits

    def _collect_memory_chunks(self, memory_root: Path) -> list[DocumentChunk]:
        """Collect chunked memory texts from archival and session user messages."""
        chunks: list[DocumentChunk] = []
        semantic_event_file = memory_root / "semantic_events.jsonl"
        event_rows = self._read_jsonl_payloads(semantic_event_file)
        current_event_ids = active_event_ids(event_rows)
        session_memory_file = memory_root / "session_memories.jsonl"
        for jsonl_file in (
            memory_root / "archival_memory.jsonl",
            memory_root / "event_summaries.jsonl",
            session_memory_file,
            memory_root / "semantic_events.jsonl",
        ):
            if jsonl_file.exists():
                for payload in self._read_jsonl_payloads(jsonl_file):
                    chunks.extend(self._memory_chunks_from_payload(payload, active_event_ids=current_event_ids))

        if session_memory_file.exists():
            return chunks

        sessions_dir = memory_root / "sessions"
        if sessions_dir.exists():
            for path in sessions_dir.glob("**/*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(payload.get("role", "")) != "user":
                        continue
                    content = str(payload.get("content", "")).strip()
                    if not content:
                        continue
                    chunks.extend(self._build_chunks(
                        source_type="memory",
                        source_id=str(payload.get("id", self._hash_text(content))),
                        text=content,
                        source_session_id=str(payload.get("session_id", "")),
                        source_message_id=str(payload.get("id", "")),
                        created_at=str(payload.get("created_at", "")),
                    ))

        return chunks

    def _memory_chunks_from_payload(
        self,
        payload: dict,
        active_event_ids: set[str] | None = None,
    ) -> list[DocumentChunk]:
        """Convert one archival memory payload into chunk list (if active/valid)."""
        if payload.get("exclude_from_retrieval") is True:
            return []
        if str(payload.get("type", "")) == "system_feedback":
            return []

        status = str(payload.get("status", "active"))
        if status != "active":
            return []
        if active_event_ids is not None and not uses_only_active_event_sources(payload, active_event_ids):
            return []

        summary = str(payload.get("summary") or payload.get("raw_excerpt") or "").strip()
        if not summary and str(payload.get("type", "")) == "semantic_event":
            summary = self._semantic_event_text(payload)
        if not summary:
            return []

        source_message_ids = payload.get("source_message_ids", [])
        if isinstance(source_message_ids, str):
            source_message_ids = [source_message_ids]
        if not isinstance(source_message_ids, list):
            source_message_ids = []

        source_id = str(payload.get("id") or self._hash_text(summary))
        source_message_id = str(source_message_ids[0]) if source_message_ids else ""
        return self._build_chunks(
            source_type="memory",
            source_id=source_id,
            text=summary,
            source_session_id=str(payload.get("source_session_id", "")),
            source_message_id=source_message_id,
            created_at=str(payload.get("created_at", "")),
            metadata={
                "memory_type": str(payload.get("type", "episodic")),
                "retention": str(payload.get("retention", "")),
                "memory_strength": str(payload.get("memory_strength", "")),
            },
        )

    def _read_jsonl_payloads(self, path: Path) -> list[dict]:
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

    def _semantic_event_text(self, payload: dict) -> str:
        description = str(payload.get("description", "")).strip()
        if not description:
            return ""
        status = str(payload.get("status", "active"))
        start_at = str(payload.get("start_at", "") or "未知时间")
        end_at = str(payload.get("end_at", "") or ("结束未知" if payload.get("end_unknown") else ""))
        return f"{description}（状态: {status}; 开始: {start_at}; 结束: {end_at}）"

    def _collect_document_chunks(self, docs_dir: Path) -> list[DocumentChunk]:
        """Collect chunked document texts from supported file extensions."""
        if not docs_dir.exists():
            return []

        chunks: list[DocumentChunk] = []
        patterns = ("*.txt", "*.md", "*.json", "*.jsonl")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(docs_dir.glob(f"**/{pattern}"))

        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            rel_path = file_path.relative_to(docs_dir).as_posix()
            chunks.extend(
                self._build_chunks(
                    source_type="document",
                    source_id=rel_path,
                    text=text,
                    source_path=str(file_path),
                )
            )
        return chunks

    def _build_chunks(
        self,
        source_type: str,
        source_id: str,
        text: str,
        source_path: str = "",
        source_session_id: str = "",
        source_message_id: str = "",
        created_at: str = "",
        metadata: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks and attach retrieval metadata."""
        pieces = chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks: list[DocumentChunk] = []
        for idx, piece in enumerate(pieces):
            chunk_id = f"{source_type}:{source_id}:c{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=piece,
                    source_type=source_type,
                    source_id=source_id,
                    source_path=source_path,
                    source_session_id=source_session_id,
                    source_message_id=source_message_id,
                    created_at=created_at,
                    metadata=metadata or {},
                )
            )
        return chunks

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    def _debug(self, message: str) -> None:
        if self.debug_hook is None:
            return
        self.debug_hook(message)
