# BGE-M3 And Active Event Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BGE-M3 with FAISS the required production RAG path and ensure inactive semantic events and memory derived from them never participate in consolidation, indexing, retrieval, or prompt context.

**Architecture:** BGE-M3 produces normalized 1024-dimensional vectors and FAISS `IndexFlatIP` performs exact inner-product search. Persisted indexes carry embedding-model metadata so incompatible legacy vectors cannot be queried. Semantic events remain append-preserved lifecycle records, but only active events are exposed; derived records are eligible only when every referenced event is active.

**Tech Stack:** Python stdlib, sentence-transformers, BAAI/bge-m3, faiss-cpu, unittest, JSON/JSONL stores.

## Global Constraints

- Production semantic retrieval must fail clearly if BGE-M3 or FAISS cannot load; it must not silently use `HashingEmbedder` or Python vector search.
- `replace` deactivates the old event and creates an active successor with `replaces_event_id`.
- `cancel` deactivates the target and creates no successor.
- Inactive events remain on disk for audit only and are excluded from Stage 1, Stage 2, indexes, retrieval, prompt context, and derived-source validity.
- Raw session evidence remains immutable and retrievable.
- Do not add keyword matching.

---

### Task 1: Require BGE-M3 And FAISS

**Files:**
- Modify: `guga/config.py`
- Modify: `guga/rag/embedder.py`
- Modify: `guga/rag/faiss_store.py`
- Modify: `guga/rag/pipeline.py`
- Test: `test/test_rag_pipeline.py`

**Interfaces:**
- `build_embedder(model_name: str) -> SentenceTransformerEmbedder` raises a descriptive error on load failure.
- `VectorStore(..., require_faiss: bool = True)` rejects missing FAISS in production.
- Persisted `index_meta.json` records embedding model and vector dimension.

- [ ] Add failing tests for the BGE-M3 default, strict embedder loading, strict FAISS loading, and stale-index model mismatch.
- [ ] Run `python -m unittest test.test_rag_pipeline` and verify the new tests fail for the intended missing behavior.
- [ ] Implement the smallest strict BGE-M3/FAISS path and index metadata validation.
- [ ] Re-run `python -m unittest test.test_rag_pipeline` and verify it passes.
- [ ] Commit with `feat(rag):切换BGE-M3与FAISS检索`.

### Task 2: Enforce Active Event Source Validity

**Files:**
- Modify: `guga/memory/manager.py`
- Modify: `guga/rag/pipeline.py`
- Test: `test/test_memory_consolidation.py`
- Test: `test/test_memory_manager.py`
- Test: `test/test_rag_pipeline.py`

**Interfaces:**
- Stage 2 receives only `SemanticEventStore.load_active()` events.
- RAG collection receives the active-event ID set and excludes derived records whose `source_event_ids` or `covered_event_ids` contain inactive events.
- `prepare_context()` applies the same source-validity gate to lexical and semantic hits.

- [ ] Add failing tests proving inactive events never reach Stage 2, canceled events have no successor, and stale summary/archival records are excluded.
- [ ] Run the narrow tests and verify each fails for the expected reason.
- [ ] Implement active-only packets, source-validity filtering, and index removal for deactivated events.
- [ ] Re-run the narrow tests and verify they pass.
- [ ] Commit with `fix(memory):隔离失效事件及派生记忆`.

### Task 3: Integration Verification

**Files:**
- Modify only files required by failing compatibility tests.

**Interfaces:**
- Existing chat, benchmark, and consolidation APIs remain unchanged.

- [ ] Run `python -m unittest test.test_semantic_events test.test_memory_consolidation test.test_memory_manager test.test_rag_pipeline`.
- [ ] Run `python -m unittest discover -s test`.
- [ ] Confirm `rg` finds no production `HashingEmbedder` fallback and no Stage 2 `semantic_event_store.load_all()` input.
- [ ] Review `git diff --check` and `git status --short`; commit only directly related compatibility fixes.

