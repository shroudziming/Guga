# Agent Semantic Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate each persona's memory and replace timeline facts with objective semantic events plus agent-specific reflection.

**Architecture:** Daily memory is rooted at `data/memory/agents/<agent_id>/`; legacy `data/memory/` is never read. Low-level consolidation writes `semantic_events.jsonl` and derived batch summaries. High-level consolidation reads those derived layers only and writes archival memory plus one agent-specific user model. Benchmark keeps its existing run/case roots and disables reflections and user models.

**Tech Stack:** Python stdlib, existing `MemoryManager`, `MemoryBankSummarizer`, JSON/JSONL stores, unittest.

## Global Constraints

- No new external dependencies.
- Default consolidation remains every 10 complete turns and on session flush.
- Raw sessions and session memories are immutable evidence; they are never rewritten by LLM consolidation.
- Semantic event fact fields and `guga_reflection` must remain semantically separate.
- Benchmark must not generate or inject reflections or a user model.
- Existing `data/memory/` is legacy: preserve it and never fall back to reading it.
- Do not push; make a local commit after each verified task.

---

### Task 1: Agent-root isolation and persona identity

**Files:**
- Create: `guga/memory/agent_identity.py`, `test/test_agent_memory_isolation.py`
- Modify: `guga/utils/paths.py`, `guga/persona/manager.py`, `guga/types.py`, `guga/chat/session.py`, `src/basic_cli_chat.py`, `src/voice_cli_chat.py`

**Interfaces:**
- `AgentIdentity(agent_id, reflection_context, persona_source, persona_fingerprint)` validates safe IDs.
- `agent_memory_root(agent_id) -> Path` returns `data/memory/agents/<agent_id>`.
- `MemoryManager(..., agent_identity: AgentIdentity | None = None)` writes an `agent_manifest.json` only when identity is present.

- [ ] Write tests proving `default`, `gentle`, and `rational` use disjoint sessions, RAG indexes, and debug roots, and that a legacy session record is not read by the new default root.
- [ ] Run `python -m unittest discover -s test -p test_agent_memory_isolation.py` and confirm the missing namespace API causes failure.
- [ ] Add persona `agent_id` and `reflection_context`; have CLI entrypoints construct an identity-bound manager and debug sink.
- [ ] Make `ChatSession` fallback use the empty `agents/default` root, never `data/memory`.
- [ ] Create and validate a root manifest containing only schema version, agent ID, persona source, persona fingerprint, and creation time.
- [ ] Re-run the task test and commit `feat(memory):隔离人格记忆根目录`.

### Task 2: Deterministic event time and event store

**Files:**
- Create: `guga/memory/semantic_events.py`, `test/test_semantic_events.py`
- Modify: `guga/memory/time_utils.py`

**Interfaces:**
- `resolve_event_time(time_expression, reference_created_at, end_unknown)` returns `start_at`, `end_at`, `time_source`, and `time_granularity`.
- `SemanticEventStore.apply_operations(operations, session_id, include_guga_reflection)` applies validated `create/update/replace/cancel/ignore` operations atomically.

- [ ] Write failing tests for Sunday/this Sunday/next Tuesday, explicit dates, date ranges, unknown time, unknown end, replacement, cancellation, update, and active-only loading.
- [ ] Run the event test and confirm imports or resolver APIs are absent.
- [ ] Implement deterministic parsing without trusting any LLM absolute date fields. A date-only one-day event uses start-of-day/end-of-day and `time_granularity=date`; unresolved time remains null and `unknown`.
- [ ] Persist exactly the approved core fields: event identity/kind/subject/entity/description, semantic time fields, lifecycle fields, raw evidence links, transaction fields, extraction confidence, and optional nested reflection.
- [ ] Ensure `replace` creates a new event with `replaces_event_id`, marks the old event inactive/replaced, and `cancel` adds lifecycle evidence without creating a false current event.
- [ ] Re-run task tests and commit `feat(memory):添加语义事件存储`.

### Task 3: Two-stage consolidation and derived summaries

**Files:**
- Modify: `guga/memory/consolidation.py`, `guga/memory/summarizer.py`, `guga/memory/manager.py`, `guga/memory/event_summary_store.py`, `test/test_memory_consolidation.py`

**Interfaces:**
- Stage 1 returns `semantic_event_operations` and `event_summaries`; operations contain relative expressions only, an optional target selected from candidates, and a reflection only in daily mode.
- Stage 2 returns `decision`, `archival_operations`, `user_model_operations`, and `reason`.

- [ ] Write failing consolidation tests for event operation validation, target-candidate validation, reflection presence in daily mode, reflection absence in benchmark mode, and no partial writes after invalid Stage 1 or Stage 2 JSON.
- [ ] Replace low-level timeline-fact prompts and packets with semantic events, conflict candidates, derived summaries, and retrieved event context.
- [ ] Build deterministic candidate sets from subject/entity/kind similarity, time overlap, open-ended related events, and reschedule/cancellation language before asking the LLM to choose a target.
- [ ] Make `guga_reflection` contain only appraisal, felt response, relational intent, and interpretation confidence; never use it for time resolution, conflict matching, or source-of-truth ranking.
- [ ] Retain only batch `event_summaries` with `source_of_truth=false`, event ID coverage/change lists, time window, evidence links, and transaction time. Delete daily/global refresh APIs and callers.
- [ ] Re-run consolidation tests and commit `feat(memory):按事件批量整理记忆`.

### Task 4: High-level user model, retrieval, and old-path removal

**Files:**
- Create: `guga/memory/user_model.py`
- Modify: `guga/memory/manager.py`, `guga/rag/pipeline.py`, `guga/types.py`, `test/test_memory_manager.py`, `test/test_chat_session_rag_flow.py`
- Delete: `guga/memory/timeline_facts.py` and obsolete portrait/profile-only modules if no runtime caller remains.

**Interfaces:**
- `guga_user_model.json` stores active understanding entries with statement, kind, confidence, stability, source event IDs, status, and update time.
- `MemoryContext` exposes semantic event hits separately from derived summary hits and raw evidence groups.

- [ ] Write failing tests asserting Stage 2 prompts contain no raw sessions, archival and user-model entries reference event IDs only, and no-op leaves high-level files untouched.
- [ ] Replace profile/personality double writes with one user-model store. Allow inactive events as labeled historical evidence but never as current facts.
- [ ] Update RAG collection and memory ranking to index semantic events, summaries, archival memory, and session memories; index active events for ordinary state queries and include inactive events only for lifecycle/history routes.
- [ ] Render prompt sections in authority order: `[Semantic Events]`, `[Derived Event Summaries]`, `[Raw Evidence]`; suppress stale raw evidence when a matching active event is authoritative.
- [ ] Remove runtime references to `timeline_facts`, old route targets, old profile files, old daily/global summaries, and flattened `guga_assessment/guga_thought` fields.
- [ ] Re-run affected tests and commit `refactor(memory):以事件重构检索与用户理解`.

### Task 5: LongMemEval chronology and isolated replay

**Files:**
- Modify: `guga/benchmark/longmemeval.py`, `src/run_longmemeval_benchmark.py`, `test/test_longmemeval_benchmark.py`, `test/test_longmemeval_scoring.py`

- [ ] Write failing tests that require preserved `haystack_session_ids`, `haystack_dates`, `question_date`, and answer-session metadata.
- [ ] Replay using source session IDs and original message dates; keep case/run root isolation and call flush at each source session end.
- [ ] Configure replay with reflections and user-model updates disabled, while retaining semantic event and archival updates.
- [ ] Add a state-change regression based on the mortgage update: a later `$400,000` semantic event replaces the earlier `$350,000` event and retrieval presents the active latest state as authoritative.
- [ ] Re-run LongMemEval tests and CLI help, then commit `fix(benchmark):保留LongMemEval原始时间线`.

### Task 6: Compatibility verification and redundancy audit

**Files:**
- Modify: applicable tests and benchmark/readme documentation only when required by changed public behavior.

- [ ] Run focused test files, then `python -m unittest discover -s test`; record the pre-existing Windows temporary-directory cleanup flake separately if it recurs after all functional assertions pass.
- [ ] Run `python -B src\\run_longmemeval_benchmark.py --help` and a small isolated replay case.
- [ ] Run a real LongMemEval long-sample API replay only after local verification; inspect memory root, event lifecycle, prompt evidence, and score output.
- [ ] Audit runtime imports and calls with `rg`: no `timeline_facts`, daily/global summary refresh, old profile/personality double writes, or flattened reflection fields may remain outside legacy-data documentation and deleted-test references.
- [ ] Review `git diff`, remove only code made obsolete by this feature, re-run full verification, and commit `refactor(memory):清理旧记忆路径` if cleanup produces a separate logical diff.
