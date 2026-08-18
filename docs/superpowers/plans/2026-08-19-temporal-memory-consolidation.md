# Temporal Memory Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed persisted turn-level time interpretations into the existing asynchronous two-stage consolidation pipeline, enforce the new temporal schema for generated memories, and guarantee fixed-turn plus graceful-exit consolidation.

**Architecture:** Keep the existing ordered low/high consolidation state machine and single background executor. Extend its batch packet with turn metadata, make low-level event output and high-level archival output use validated canonical intervals, then enrich state/debug records with trigger and watermark information. Finish by covering `/exit`, EOF, and input-time interruption in both CLIs and running the approved real-API restart scenario.

**Tech Stack:** Python 3, JSONL memory/session stores, atomic JSON state writes, existing OpenAI-compatible structured generation, `ThreadPoolExecutor`, `unittest`, BGE-M3/FAISS through the retrieval plan.

**Spec:** `docs/superpowers/specs/2026-08-18-temporal-memory-retrieval-consolidation-design.md`

## Global Constraints

- Execute this plan only after `docs/superpowers/plans/2026-08-19-temporal-memory-retrieval.md` is complete.
- Keep consolidation asynchronous at the configured fixed-turn threshold and synchronous only during graceful shutdown settlement.
- Preserve the existing two stages: semantic events/event summaries first, archival memory/user model second.
- Treat later explicit user correction as stronger evidence than earlier model-resolved `temporal_context`.
- Treat assistant text as conversational context, never as sole evidence for a new user fact.
- Program code owns `created_at` and `updated_at`; model output cannot set them.
- Require new semantic-event and event-summary intervals to be timezone-aware, left-closed/right-open, and non-empty.
- Permit `archival_memory.invalid_at` to be empty only when the fact is intentionally open-ended; never infer an event interval from legacy fields.
- Do not migrate old memory data or add old-format retrieval compatibility in this plan.
- Do not solve cross-layer duplicate memory in this plan.
- Run real API validation only through explicit live-test commands and never print API secrets.
- Use TDD and commit each independently verified task with the repository commit format.

---

### Task 1: Carry turn-level temporal evidence into low-level consolidation

**Files:**
- Modify: `guga/memory/manager.py`
- Modify: `guga/memory/summarizer.py`
- Modify: `test/test_memory_consolidation.py`

**Interfaces:**
- Consumes: user-message `metadata.turn` written by the retrieval plan.
- Produces: each `new_turns[]` item with `turn_id`, `temporal_context`, and `tool_interactions` in addition to existing message IDs/text/timestamp.
- Preserves: `MemoryBankSummarizer.consolidate_low_level_memory(packet, include_guga_reflection, reflection_context)`.

- [ ] **Step 1: Write a failing packet-propagation test**

Add this test around `_build_low_level_packet`:

```python
    def test_low_level_packet_includes_persisted_turn_temporal_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            user_id = manager.record_user_message(
                "sess_temporal_packet",
                "我8月20号有个面试",
                created_at="2026-08-18T10:00:00+08:00",
            )
            assistant_id = manager.record_assistant_message("sess_temporal_packet", "记住了。")
            manager.attach_turn_metadata(
                session_id="sess_temporal_packet",
                user_message_id=user_id,
                metadata={
                    "turn": {
                        "turn_id": f"turn_{user_id}",
                        "assistant_message_id": assistant_id,
                        "temporal_context": {
                            "source_message_id": user_id,
                            "reference_time": "2026-08-18T10:00:00+08:00",
                            "time_queries": [{
                                "original_expression": "8月20号",
                                "start_at": "2026-08-20T00:00:00+08:00",
                                "end_at": "2026-08-21T00:00:00+08:00",
                                "relation": "overlap",
                                "granularity": "day",
                            }],
                        },
                        "tool_interactions": [{
                            "tool_call_id": "call_time",
                            "tool": "memory_time_search",
                            "arguments": {},
                            "ok": True,
                            "returned_memory_ids": [],
                            "error": "",
                        }],
                    }
                },
            )
            packet = manager._build_low_level_packet(
                session_id="sess_temporal_packet",
                pending_turns=[{
                    "turn_id": f"turn_{user_id}",
                    "user_message_id": user_id,
                    "assistant_message_id": assistant_id,
                }],
            )
            turn = packet["new_turns"][0]
            self.assertEqual(turn["turn_id"], f"turn_{user_id}")
            self.assertEqual(turn["temporal_context"]["time_queries"][0]["start_at"], "2026-08-20T00:00:00+08:00")
            self.assertEqual(turn["tool_interactions"][0]["tool"], "memory_time_search")
```

Add this prompt-capture test for the evidence priority:

```python
    def test_low_level_prompt_declares_temporal_evidence_priority(self) -> None:
        model = FixedLowLevelModel(reflection=None)
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
        summarizer.consolidate_low_level_memory(_reflection_packet(), include_guga_reflection=False)
        prompt = model.prompts[0]
        self.assertIn(
            "later explicit user correction > explicit absolute user time > temporal_context > existing memory time > model inference",
            prompt,
        )
```

Give `FixedLowLevelModel` a `prompts: list[str]` field and append `messages[-1]["content"]` in `generate_reply` so this assertion uses the actual prompt. The prompt must contain this exact text:

```text
later explicit user correction > explicit absolute user time > temporal_context > existing memory time > model inference
```

- [ ] **Step 2: Run the two focused tests and verify temporal metadata is absent**

Run: `python -m unittest test.test_memory_consolidation.MemoryConsolidationTest.test_low_level_packet_includes_persisted_turn_temporal_context -v`

Expected: FAIL because `_load_pending_turn_messages` omits the turn metadata.

- [ ] **Step 3: Extend pending-turn loading without trusting assistant output as fact**

Change `_load_pending_turn_messages` so each output row is built as follows:

```python
            user_metadata = dict(user.get("metadata", {}) or {})
            turn_metadata = dict(user_metadata.get("turn", {}) or {})
            new_turns.append({
                "turn_id": str(turn_metadata.get("turn_id") or turn.get("turn_id") or f"turn_{user.get('id', '')}"),
                "user_message_id": str(user.get("id", "")),
                "assistant_message_id": str(assistant.get("id", "")),
                "created_at": str(user.get("created_at", "")),
                "user_text": str(user.get("content", "")),
                "assistant_text": str(assistant.get("content", "")),
                "temporal_context": dict(turn_metadata.get("temporal_context", {}) or {}),
                "tool_interactions": list(turn_metadata.get("tool_interactions", []) or []),
            })
```

Update `_trim_packet` so truncation shortens only `user_text`, `assistant_text`, and verbose retrieved summaries; it must never remove or truncate ISO timestamps, IDs, `temporal_context`, or tool result IDs.

- [ ] **Step 4: Update the low-level prompt with exact source and correction rules**

Add these rules to `consolidate_low_level_memory`:

```text
- User messages are the primary factual evidence. Assistant messages explain dialogue flow and cannot alone create a user fact.
- new_turns[*].temporal_context contains absolute intervals resolved during the Agent Loop. Reuse them when they match the user's meaning.
- Resolve conflicts using: later explicit user correction > explicit absolute user time > temporal_context > existing memory time > model inference.
- A later user correction may replace an earlier temporal_context interval from the same batch.
- Never copy a retrieved tool result into a new user fact unless the user's messages support it.
```

The prompt must continue to include the whole JSON packet so the model sees evidence turn IDs and existing event IDs.

- [ ] **Step 5: Run consolidation packet and summarizer tests**

Run: `python -m unittest test.test_memory_consolidation -v`

Expected: all tests PASS, including temporal metadata propagation and the exact priority rule.

- [ ] **Step 6: Commit temporal-evidence propagation**

```powershell
git add guga/memory/manager.py guga/memory/summarizer.py test/test_memory_consolidation.py
git commit -m "feat(memory): 向巩固传递轮次时间证据"
```

### Task 2: Enforce canonical semantic-event and event-summary intervals

**Files:**
- Modify: `guga/memory/summarizer.py`
- Modify: `guga/memory/semantic_events.py`
- Modify: `guga/memory/event_summary_store.py`
- Modify: `test/test_semantic_events.py`
- Modify: `test/test_memory_consolidation.py`
- Modify: `test/test_memory_time_utils.py`

**Interfaces:**
- Consumes: low-level structured operations.
- Produces: every new/replaced event with timezone-aware `start_at`, exclusive `end_at`, `time_granularity`, and persisted `end_unknown=False`.
- Produces: every created event summary with a non-empty `time_window_start/time_window_end` derived from covered canonical events.

- [ ] **Step 1: Replace zero-length/unknown-end tests with strict interval tests**

Add these store tests and update existing fixtures from identical timestamps to valid ranges:

```python
    def test_store_rejects_create_without_complete_non_empty_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            base = {
                "operation": "create",
                "event_kind": "appointment",
                "subject": "user",
                "entity": "面试",
                "description": "用户参加面试",
                "time_expression": "8月20号",
                "time_granularity": "day",
            }
            for start_at, end_at, error in (
                ("2026-08-20T00:00:00+08:00", None, "end_at is required"),
                ("2026-08-20T00:00:00+08:00", "2026-08-20T00:00:00+08:00", "start_at must be earlier than end_at"),
                ("2026-08-20T00:00:00", "2026-08-21T00:00:00+08:00", "start_at must include timezone"),
            ):
                with self.subTest(start_at=start_at, end_at=end_at):
                    with self.assertRaisesRegex(ValueError, error):
                        store.apply_operations(
                            operations=[{**base, "start_at": start_at, "end_at": end_at}],
                            session_id="sess_strict",
                            include_guga_reflection=False,
                        )

    def test_store_persists_end_exclusive_interval_and_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SemanticEventStore(Path(tmp_dir) / "semantic_events.jsonl")
            event_id = store.apply_operations(
                operations=[{
                    "operation": "create",
                    "event_kind": "appointment",
                    "subject": "user",
                    "entity": "面试",
                    "description": "用户参加面试",
                    "time_expression": "8月20号",
                    "start_at": "2026-08-20T00:00:00+08:00",
                    "end_at": "2026-08-21T00:00:00+08:00",
                    "time_granularity": "day",
                }],
                session_id="sess_strict",
                include_guga_reflection=False,
            ).created_event_ids[0]
            event = next(item for item in store.load_all() if item["id"] == event_id)
            self.assertEqual(event["end_at"], "2026-08-21T00:00:00+08:00")
            self.assertEqual(event["time_granularity"], "day")
            self.assertFalse(event["end_unknown"])
```

Add these event-summary tests:

```python
    def test_event_summary_uses_covered_event_exclusive_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventSummaryStore(Path(tmp_dir) / "event_summaries.jsonl")
            result = EventApplyResult(["evt_1", "evt_2"], [], [])
            summary = store.upsert_batch_summary(
                session_id="sess_summary",
                batch_seq=1,
                payload={"summary": "两项安排", "confidence": 0.9},
                source_message_ids=["msg_1"],
                event_result=result,
                covered_events=[
                    {"id": "evt_1", "start_at": "2026-08-20T14:00:00+08:00", "end_at": "2026-08-20T15:00:00+08:00"},
                    {"id": "evt_2", "start_at": "2026-08-21T09:00:00+08:00", "end_at": "2026-08-21T10:00:00+08:00"},
                ],
            )
            self.assertEqual(summary["time_window_start"], "2026-08-20T14:00:00+08:00")
            self.assertEqual(summary["time_window_end"], "2026-08-21T10:00:00+08:00")

    def test_event_summary_rejects_missing_covered_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventSummaryStore(Path(tmp_dir) / "event_summaries.jsonl")
            with self.assertRaisesRegex(ValueError, "covered canonical event intervals"):
                store.upsert_batch_summary(
                    session_id="sess_summary",
                    batch_seq=1,
                    payload={"summary": "无来源总结"},
                    source_message_ids=["msg_1"],
                    event_result=EventApplyResult([], [], []),
                    covered_events=[],
                )
```

- [ ] **Step 2: Run strict event and summary tests and confirm old contracts fail**

Run: `python -m unittest test.test_semantic_events test.test_memory_time_utils -v`

Expected: FAIL because `_llm_time_fields` still permits missing ends and identical start/end timestamps.

- [ ] **Step 3: Change the low-level model schema to canonical intervals**

In `MemoryBankSummarizer.consolidate_low_level_memory`, change event output fields to:

```json
{
  "time_expression": "string",
  "start_at": "timezone-aware ISO 8601 inclusive start",
  "end_at": "timezone-aware ISO 8601 exclusive end",
  "time_granularity": "minute|hour|day|week|month|year|range"
}
```

Remove instructions that allow `null` or `end_at == start_at`. Add:

```text
- Every create and replace operation must contain a complete non-empty interval with start_at < end_at.
- end_at is exclusive. A calendar day is [00:00:00 on that day, 00:00:00 on the next day).
- For a stated hour or minute without a duration, use the smallest expressed granularity as the exclusive bucket end.
- If the user gives an unknown open-ended state, do not invent an event end; route stable open-ended truth to high-level archival memory only when supported by committed facts.
```

In `_validate_low_level_result`:

- Require `start_at`, `end_at`, and `time_granularity` for `create` and `replace`.
- For `update`, require the complete trio when any one temporal field is present.
- Parse both timestamps with `datetime.fromisoformat`, require non-null `tzinfo`, and require `start < end`.
- Remove model-supplied `end_unknown`; the application persists `False`.
- Leave `cancel` free of new time fields.

Update shared test model fixtures at the same time. `ConsolidationModel` and `FixedLowLevelModel` must return `start_at="2026-07-03T00:00:00+08:00"`, `end_at="2026-07-04T00:00:00+08:00"`, and `time_granularity="day"`; remove model-supplied `end_unknown`. Update every semantic-event fixture used for create/replace to a non-empty exclusive-end interval.

- [ ] **Step 4: Make SemanticEventStore enforce the same contract**

Replace `_llm_time_fields` with strict behavior:

```python
def _llm_time_fields(payload: dict) -> dict:
    start_raw = str(payload.get("start_at", "")).strip()
    end_raw = str(payload.get("end_at", "")).strip()
    if not start_raw:
        raise ValueError("start_at is required")
    if not end_raw:
        raise ValueError("end_at is required")
    try:
        start_source = datetime.fromisoformat(start_raw)
        end_source = datetime.fromisoformat(end_raw)
    except ValueError as exc:
        raise ValueError("start_at and end_at must be ISO 8601") from exc
    if start_source.tzinfo is None:
        raise ValueError("start_at must include timezone")
    if end_source.tzinfo is None:
        raise ValueError("end_at must include timezone")
    start_at = parse_datetime(start_raw)
    end_at = parse_datetime(end_raw)
    if start_at is None or end_at is None or start_at >= end_at:
        raise ValueError("start_at must be earlier than end_at")
    granularity = str(payload.get("time_granularity", "")).strip()
    if granularity not in {"minute", "hour", "day", "week", "month", "year", "range"}:
        raise ValueError("time_granularity is invalid")
    return {
        "start_at": format_beijing(start_at),
        "end_at": format_beijing(end_at),
        "end_unknown": False,
        "time_source": "llm_resolved",
        "time_granularity": granularity,
    }
```

Call it for new/replaced events. For updates, preserve the old interval when no temporal key is present; otherwise require the full new trio.

- [ ] **Step 5: Require event summaries to cover canonical event time**

In `EventSummaryStore.upsert_batch_summary`, build starts and ends separately:

```python
        starts = [str(event.get("start_at", "")) for event in covered_events if str(event.get("start_at", ""))]
        ends = [str(event.get("end_at", "")) for event in covered_events if str(event.get("end_at", ""))]
        if not starts or not ends:
            raise ValueError("event summary requires covered canonical event intervals")
        time_window_start = min(starts)
        time_window_end = max(ends)
        if parse_datetime(time_window_start) >= parse_datetime(time_window_end):
            raise ValueError("event summary time window must be non-empty")
```

Persist `time_window_start` and `time_window_end`; do not fall back to summary creation time.

Wrap `SemanticEventStore.apply_operations` and `EventSummaryStore.upsert_batch_summary` validation failures inside `_apply_low_level_consolidation`:

```python
        try:
            outcome = self.semantic_event_store.apply_operations(
                operations=operations,
                session_id=session_id,
                include_guga_reflection=self.consolidation_config.include_guga_reflection,
            )
        except ValueError as exc:
            raise SummaryGenerationError(f"invalid semantic event operation: {exc}") from exc
```

Use the same pattern around each event-summary write with `invalid event summary`. This keeps failures in the existing low-stage retry/debug state machine instead of escaping the background future.

- [ ] **Step 6: Run semantic-event, summary, and consolidation suites**

Run: `python -m unittest test.test_semantic_events test.test_memory_time_utils test.test_memory_consolidation -v`

Expected: all tests PASS under the new exclusive-end contract.

- [ ] **Step 7: Commit canonical event intervals**

```powershell
git add guga/memory/summarizer.py guga/memory/semantic_events.py guga/memory/event_summary_store.py test/test_semantic_events.py test/test_memory_consolidation.py test/test_memory_time_utils.py
git commit -m "feat(memory): 约束新事件时间区间"
```

### Task 3: Generate and validate archival fact validity intervals

**Files:**
- Modify: `guga/memory/summarizer.py`
- Modify: `guga/memory/manager.py`
- Modify: `test/test_memory_consolidation.py`
- Modify: `test/test_memorybank_repro.py`

**Interfaces:**
- Consumes: committed active semantic events and event summaries in the high-level packet.
- Produces: archival operations with `valid_at: str` and `invalid_at: str | null`.
- Preserves stored archival `type=episodic` for existing layer compatibility while temporal retrieval exposes it as `record_type=archival_memory`.

- [ ] **Step 1: Write failing high-level validation and persistence tests**

Add this validator model and coverage:

```python
    def test_high_level_archival_operation_requires_validity_interval(self) -> None:
        class MissingValidityModel:
            def generate_reply(self, messages, gen):
                return json.dumps({
                    "decision": "update_high_level_memory",
                    "archival_operations": [{
                        "topic": "interview",
                        "summary": "用户正在准备面试",
                        "importance": 0.8,
                        "confidence": 0.9,
                        "source_event_ids": ["evt_1"],
                    }],
                    "user_model_operations": [],
                    "reason": "time-bound fact",
                }, ensure_ascii=False)

        model = MissingValidityModel()
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
        with self.assertRaisesRegex(SummaryGenerationError, "valid_at"):
            summarizer.consolidate_high_level_memory({
                "semantic_events": [{"id": "evt_1"}],
                "event_summaries": [],
                "archival_memory": [],
                "guga_user_model": {"insights": []},
            })
```

Add this manager persistence test:

```python
    def test_archival_update_persists_closed_and_open_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(memory_root=Path(tmp), model=None, enable_semantic=False)
            base = {
                "topic": "interview",
                "summary": "用户正在准备面试",
                "importance": 0.8,
                "confidence": 0.9,
                "source_event_ids": ["evt_1"],
                "valid_at": "2026-08-18T10:00:00+08:00",
            }
            closed = manager._build_archival_update(
                item={**base, "invalid_at": "2026-08-21T00:00:00+08:00"},
                session_id="sess_archive",
                batch_seq=1,
                operation_index=0,
            )
            opened = manager._build_archival_update(
                item={**base, "invalid_at": None},
                session_id="sess_archive",
                batch_seq=1,
                operation_index=1,
            )
            self.assertEqual(closed["valid_at"], "2026-08-18T10:00:00+08:00")
            self.assertEqual(closed["invalid_at"], "2026-08-21T00:00:00+08:00")
            self.assertEqual(opened["invalid_at"], "")
```

Update `ConsolidationModel` high-level fixtures to include `valid_at="2026-07-03T00:00:00+08:00"` and `invalid_at="2026-07-04T00:00:00+08:00"` so existing successful high-stage tests use the new contract.

- [ ] **Step 2: Run the focused tests and confirm the old schema accepts missing validity**

Run: `python -m unittest test.test_memory_consolidation test.test_memorybank_repro -v`

Expected: FAIL because high-level output validation does not require `valid_at` and `_build_archival_update` reparses the summary text.

- [ ] **Step 3: Extend the high-level prompt and validator**

Change each archival operation schema to:

```json
{
  "topic": "string",
  "summary": "string",
  "importance": 0.0,
  "confidence": 0.0,
  "source_event_ids": ["evt_id"],
  "valid_at": "timezone-aware ISO 8601",
  "invalid_at": "timezone-aware ISO 8601 or null"
}
```

Add prompt rules:

```text
- valid_at/invalid_at describe when the archival fact is true; they are not record creation timestamps.
- Derive validity only from committed semantic_events and event_summaries in this packet.
- invalid_at may be null only for an intentionally open-ended fact.
- Never output created_at or updated_at.
```

In `_validate_high_level_result`, require timezone-aware `valid_at`; if `invalid_at` is non-null, require timezone awareness and `valid_at < invalid_at`. Continue validating all `source_event_ids` against the packet.

- [ ] **Step 4: Persist model-validated validity without regex inference**

Replace the `apply_temporal_fields` call inside `_build_archival_update` with direct fields:

```python
        valid_at = format_beijing(parse_datetime(str(item["valid_at"])))
        invalid_raw = item.get("invalid_at")
        invalid_at = format_beijing(parse_datetime(str(invalid_raw))) if invalid_raw else ""
        return normalize_memorybank_fields({
            "id": f"mem_{hashlib.sha256(f'{session_id}:{batch_seq}:archival:{operation_index}'.encode('utf-8')).hexdigest()[:24]}",
            "type": "episodic",
            "topic": str(item.get("topic") or "general").strip()[:64] or "general",
            "summary": summary[:500],
            "raw_excerpt": summary[:500],
            "importance": self._clamp_float(item.get("importance"), 0.7),
            "confidence": self._clamp_float(item.get("confidence"), 0.7),
            "created_at": now,
            "updated_at": now,
            "valid_at": valid_at,
            "invalid_at": invalid_at,
            "semantic_day": time_day_bucket(valid_at),
            "time_source": "llm_consolidated",
            "time_granularity": "range",
            "last_recalled_at": now,
            "memory_strength": 1,
            "retention": 1.0,
            "source_session_id": session_id,
            "source_event_ids": source_event_ids,
            "status": "active",
        })
```

If validation is bypassed and parsing fails, raise `SummaryGenerationError` before any file write. Do not fall back to `now` or summary regex extraction.

- [ ] **Step 5: Run high-level and retrieval integration tests**

Run: `python -m unittest test.test_memory_consolidation test.test_memorybank_repro test.test_memory_manager test.test_temporal_retrieval -v`

Expected: all tests PASS; open-ended archival validity works, while semantic events remain closed intervals.

- [ ] **Step 6: Commit archival validity generation**

```powershell
git add guga/memory/summarizer.py guga/memory/manager.py test/test_memory_consolidation.py test/test_memorybank_repro.py
git commit -m "feat(memory): 生成归档记忆有效区间"
```

### Task 4: Record interval/exit triggers and consolidation watermarks

**Files:**
- Modify: `guga/memory/manager.py`
- Modify: `test/test_memory_consolidation.py`
- Modify: `test/test_chat_session_rag_flow.py`

**Interfaces:**
- Consumes: existing `MemoryConsolidationConfig.batch_turns`, pending/queued turns, and active-batch retry state.
- Produces: `_consolidate_pending_turns(session_id, force, trigger_reason)` with `interval`, `manual`, or `exit`.
- Produces state keys: `last_completed_turn_id`, `last_consolidated_turn_id`, and active-batch `trigger_reason/turn_ids`.

- [ ] **Step 1: Write failing fixed-turn, exit-tail, and watermark tests**

In the existing threshold test, capture `first_user_id` and `second_user_id` from `record_user_message`, then add:

```python
            state = json.loads((Path(tmp) / "consolidation_state.json").read_text(encoding="utf-8"))
            session_state = state["sessions"]["sess_batch"]
            self.assertEqual(session_state["last_completed_turn_id"], f"turn_{second_user_id}")
            self.assertEqual(session_state["last_consolidated_turn_id"], f"turn_{second_user_id}")
```

Add a debug-capture test:

```python
    def test_interval_and_exit_consolidation_log_distinct_trigger_reasons(self) -> None:
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(
                memory_root=Path(tmp),
                model=ConsolidationModel(),
                enable_semantic=False,
                debug=True,
                debug_sink=logs.append,
                consolidation_config=MemoryConsolidationConfig(batch_turns=2),
            )
            manager.record_user_message("sess_trigger", "第一轮", created_at="2026-08-18T10:00:00+08:00")
            manager.record_assistant_message("sess_trigger", "已记录。", created_at="2026-08-18T10:01:00+08:00")
            manager.finalize_turn_async("sess_trigger")
            manager.wait_for_background_tasks()
            self.assertFalse(any('"trigger_reason":"interval"' in line for line in logs))
            manager.consolidate_until_settled("sess_trigger", retry_delays=())
            self.assertTrue(any('"trigger_reason":"exit"' in line for line in logs))
```

In `test_restart_resumes_high_stage_without_repeating_low_stage`, add this assertion before and after constructing the recovery manager:

```python
            failed_state = json.loads((root / "consolidation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(failed_state["sessions"]["sess_restart"]["active_batch"]["trigger_reason"], "interval")
```

After recovery, assert the completed debug/state watermark refers to the same batch turn IDs and that `active_batch` is `None`.

- [ ] **Step 2: Run consolidation trigger tests and verify watermarks are missing**

Run: `python -m unittest test.test_memory_consolidation test.test_chat_session_rag_flow -v`

Expected: FAIL on missing watermark and trigger-reason fields.

- [ ] **Step 3: Thread explicit trigger reasons through existing entry points**

Use these calls:

```python
# threshold reached inside _finalize_turn_state
self._consolidate_pending_turns(session_id=session_id, force=False, trigger_reason="interval")

# manual public flush
def flush_session_memory(self, session_id: str, trigger_reason: str = "manual") -> dict[str, Any]:
    return self._consolidate_pending_turns(session_id=session_id, force=True, trigger_reason=trigger_reason)

# graceful shutdown
result = self.flush_session_memory(session_id, trigger_reason="exit")
```

Every recursive or retry call must preserve the original trigger reason from `active_batch`; it must not change an interval batch into an exit batch merely because shutdown waits for it.

- [ ] **Step 4: Persist turn IDs and advance watermarks only after committed stages**

When appending a pending turn, persist `turn_id` from `_turn_state`, defaulting deterministically to `turn_<user_message_id>`, and update `last_completed_turn_id`.

When creating `active_batch`, add:

```python
"trigger_reason": trigger_reason,
"turn_ids": [str(turn.get("turn_id", "")) for turn in pending_turns if str(turn.get("turn_id", ""))],
```

After the high stage commits and before clearing `active_batch`, set:

```python
turn_ids = [str(value) for value in active.get("turn_ids", []) if str(value)]
if turn_ids:
    session_state["last_consolidated_turn_id"] = turn_ids[-1]
```

Initialize `last_completed_turn_id` and `last_consolidated_turn_id` to empty strings in `_session_consolidation_state`. Do not advance `last_consolidated_turn_id` on low-stage or high-stage failure.

- [ ] **Step 5: Emit structured trigger and completion debug records**

At batch creation log:

```python
self._debug(session_id, "consolidation_trigger " + json.dumps({
    "batch_seq": active["batch_seq"],
    "trigger_reason": trigger_reason,
    "turn_ids": active["turn_ids"],
    "pending_count": len(pending_turns),
}, ensure_ascii=False, separators=(",", ":")))
```

Extend `consolidation_done` with `trigger_reason`, `turn_ids`, `last_completed_turn_id`, and `last_consolidated_turn_id`. Keep existing structured-call diagnostics and error hashes.

- [ ] **Step 6: Run trigger, failure, restart, and shutdown tests**

Run: `python -m unittest test.test_memory_consolidation test.test_chat_session_rag_flow -v`

Expected: all tests PASS. The existing high-stage retry test proves the low stage is not repeated, and new assertions prove watermarks do not advance prematurely.

- [ ] **Step 7: Commit trigger and watermark state**

```powershell
git add guga/memory/manager.py test/test_memory_consolidation.py test/test_chat_session_rag_flow.py
git commit -m "feat(memory): 记录巩固触发与处理水位"
```

### Task 5: Settle memory on CLI exit, EOF, and input interruption

**Files:**
- Modify: `src/basic_cli_chat.py`
- Modify: `src/voice_cli_chat.py`
- Create: `test/test_cli_shutdown.py`

**Interfaces:**
- Consumes: `ChatSession.settle_memory_for_shutdown()`.
- Produces: exactly one shutdown settlement for `/exit`, EOF, or `KeyboardInterrupt` raised while waiting for input.
- Preserves: generation-time Ctrl+C cancellation behavior inside an active streamed response.

- [ ] **Step 1: Write failing basic and voice CLI shutdown tests**

Use `unittest.mock` to replace model/persona/audio construction and drive input termination:

```python
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import src.basic_cli_chat as basic_cli
import src.voice_cli_chat as voice_cli


class CliShutdownTest(unittest.TestCase):
    def _fake_session(self) -> MagicMock:
        session = MagicMock()
        session.settle_memory_for_shutdown.return_value = {"status": "complete"}
        return session

    def test_basic_cli_eof_settles_once(self) -> None:
        session = self._fake_session()
        persona = MagicMock(system_prompt="test", reflection_context="test", source_path="test", persona_fingerprint="test", expression_tags=())
        with (
            patch.dict(os.environ, {"Guga_DEBUG": "0"}, clear=False),
            patch.object(basic_cli, "_load_env_file"),
            patch.object(basic_cli, "read_user_input", side_effect=EOFError),
            patch.object(basic_cli, "create_chat_model", return_value=MagicMock()),
            patch.object(basic_cli, "MemoryManager", return_value=MagicMock()),
            patch.object(basic_cli, "ChatSession", return_value=session),
            patch.object(basic_cli.PersonaManager, "load", return_value=persona),
            patch.object(basic_cli, "identity_from_persona", return_value=MagicMock(agent_id="test")),
        ):
            basic_cli.main()
        session.settle_memory_for_shutdown.assert_called_once_with()

    def test_voice_cli_input_interrupt_settles_once(self) -> None:
        session = self._fake_session()
        persona = MagicMock(system_prompt="test", reflection_context="test", source_path="test", persona_fingerprint="test", expression_tags=())
        with (
            patch.dict(os.environ, {"Guga_DEBUG": "0"}, clear=False),
            patch.object(voice_cli, "_load_env_file"),
            patch.object(voice_cli, "read_user_input", side_effect=KeyboardInterrupt),
            patch.object(voice_cli, "create_chat_model", return_value=MagicMock()),
            patch.object(voice_cli, "MemoryManager", return_value=MagicMock()),
            patch.object(voice_cli, "ChatSession", return_value=session),
            patch.object(voice_cli.PersonaManager, "load", return_value=persona),
            patch.object(voice_cli, "identity_from_persona", return_value=MagicMock(agent_id="test")),
            patch.object(voice_cli, "configure_voice_tool_mode", return_value=False),
            patch.object(voice_cli.GptSoVitsConfig, "from_env", return_value=MagicMock(endpoint="test", ref_audio_path="test")),
            patch.object(voice_cli, "GptSoVitsHttpClient", return_value=MagicMock()),
            patch.object(voice_cli, "_prewarm_tts"),
        ):
            voice_cli.main()
        session.settle_memory_for_shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
```

Add `/exit` cases by replacing each `read_user_input` side effect above with `return_value="/exit"`; reuse the same patch set and assert `settle_memory_for_shutdown.assert_called_once_with()`. Because `Guga_DEBUG=0`, neither test constructs a debug sink, and all model/TTS/device factories are already replaced.

- [ ] **Step 2: Run the new CLI tests and confirm EOF/interrupt currently escape**

Run: `python -m unittest test.test_cli_shutdown -v`

Expected: FAIL with uncaught `EOFError` or `KeyboardInterrupt`.

- [ ] **Step 3: Restructure the basic CLI loop around one settlement point**

Use this control shape after session creation:

```python
    try:
        while True:
            try:
                user_text = read_user_input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text in {"/exit", "exit", "quit"}:
                break
            if user_text == "/clear":
                session.clear()
                print("会话已清空。")
                continue
            if user_text == "/rag_rebuild":
                result = session.memory_manager.rebuild_rag_indexes(session_id=session.session_id)
                print(f"RAG 索引已重建: memory_chunks={result['memory_chunks']}, document_chunks={result['document_chunks']}, total_chunks={result['total_chunks']}")
                continue
            cancel_event = Event()
            stream = session.reply_stream(user_text, cancel_event=cancel_event)
            output_parser = PersonaOutputParser(persona.expression_tags)
            print("小咕嘎> ", end="", flush=True)
            try:
                for chunk in stream:
                    _render_persona_events(output_parser.feed(chunk), debug_enabled=debug_enabled)
                _render_persona_events(output_parser.flush(), debug_enabled=debug_enabled)
                print("\n")
            except KeyboardInterrupt:
                cancel_event.set()
                for _ in stream:
                    pass
                _render_persona_events(output_parser.flush(), debug_enabled=debug_enabled)
                print("\n[已停止生成]\n")
    finally:
        result = session.settle_memory_for_shutdown()
        print(f"记忆整理: {result.get('status', 'unknown')}")
        print("已退出。")
```

The inner generation interrupt remains a cancellation and continues the chat. Only an interrupt from the input read exits the outer loop.

- [ ] **Step 4: Apply the same one-settlement shape to voice CLI**

Wrap the voice input loop in `try/finally`, break for `/exit`, `exit`, `quit`, EOF, or input-time `KeyboardInterrupt`, and call `settle_memory_for_shutdown()` only in `finally`. Preserve `VoiceChatRunner.run_turn` and its current inner generation-interrupt cancellation.

- [ ] **Step 5: Run CLI and voice regression tests**

Run: `python -m unittest test.test_cli_shutdown test.test_cli_input test.test_voice_pipeline -v`

Expected: all tests PASS without real device or network access.

- [ ] **Step 6: Commit graceful CLI settlement**

```powershell
git add src/basic_cli_chat.py src/voice_cli_chat.py test/test_cli_shutdown.py
git commit -m "fix(cli): 正常退出时收尾记忆巩固"
```

### Task 6: Add and run the real-API restart acceptance scenario

**Files:**
- Modify: `scripts/live_memory_api_validation.py`
- Modify: `test/test_live_memory_api_validation.py`
- Create: `test/test_temporal_memory_live_api.py`

**Interfaces:**
- Consumes: completed retrieval and consolidation plans plus project `.env` API configuration.
- Produces: live scenario `temporal-e2e` and a guarded live test enabled by `GUGA_RUN_LIVE_API_TESTS=1`.

- [ ] **Step 1: Update unit validators to the exclusive-end contract**

Replace the old same-timestamp/unknown-end validator with:

```python
def validate_time_event(event: dict) -> None:
    start = _parse_iso(str(event.get("start_at") or ""), "start_at")
    end = _parse_iso(str(event.get("end_at") or ""), "end_at")
    if start >= end:
        raise AssertionError(f"event interval must be non-empty and end-exclusive: {event}")
    if event.get("end_unknown") is not False:
        raise AssertionError(f"new event must persist end_unknown=false: {event}")
```

Update `test/test_live_memory_api_validation.py` to accept `[2026-08-20T00:00:00+08:00, 2026-08-21T00:00:00+08:00)` and reject equal or missing ends.

Remove the obsolete `multi-day` unknown-end live scenario. Update `same-day`, `replace`, and `cancel` seed operations to use exclusive ends and `time_granularity`; set `_REFLECTION_KEYS={"appraisal", "felt_response"}` to match the current reflection contract. The `all` scenario list must contain only scenarios that satisfy the new canonical interval schema.

- [ ] **Step 2: Write the guarded live test before adding the scenario**

```python
from __future__ import annotations

import os
import unittest

from scripts.live_memory_api_validation import run_temporal_e2e


@unittest.skipUnless(os.environ.get("GUGA_RUN_LIVE_API_TESTS") == "1", "live API test disabled")
class TemporalMemoryLiveApiTest(unittest.TestCase):
    def test_real_api_consolidates_on_exit_and_retrieves_after_restart(self) -> None:
        result = run_temporal_e2e()
        self.assertEqual(result["event_start_at"], "2026-08-20T00:00:00+08:00")
        self.assertEqual(result["event_end_at"], "2026-08-21T00:00:00+08:00")
        self.assertIn(result["event_id"], result["retrieved_memory_ids"])
        self.assertTrue(result["first_turn_has_temporal_context"])
        self.assertTrue(result["second_turn_has_temporal_context"])
        self.assertTrue(result["debug_has_time_search"])
        self.assertTrue(result["debug_has_exit_consolidation"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run unit mode and confirm the new scenario function is missing**

Run: `python -m unittest test.test_live_memory_api_validation test.test_temporal_memory_live_api -v`

Expected: the validator unit tests run; the live test is skipped unless enabled, and direct import initially fails until `run_temporal_e2e` is added.

- [ ] **Step 4: Implement the isolated real-API scenario**

In `scripts/live_memory_api_validation.py`, add `temporal-e2e` to `--scenario` choices and implement `run_temporal_e2e()` with these exact actions:

1. Call `load_env_file()` and create the real configured chat model.
2. Create a `TemporaryDirectory` for memory and a second temporary directory for `FileDebugSink` output.
3. Create `MemoryManager` with `batch_turns=10`, semantic retrieval enabled, and debug enabled.
4. Create a first `ChatSession` and call `reply("我8月20号有个面试", created_at="2026-08-18T10:00:00+08:00")`.
5. Call `settle_memory_for_shutdown()` and require status `complete`.
6. Read `semantic_events.jsonl`; require exactly one active interview event with `[2026-08-20T00:00:00+08:00, 2026-08-21T00:00:00+08:00)`.
7. Create a new `MemoryManager` over the same memory root and a new `ChatSession` with a different session ID.
8. Call `reply("8月20号有什么事情", finalize_memory=False, created_at="2026-08-19T10:00:00+08:00")`.
9. Read both session files. Assert their user rows contain `metadata.turn.temporal_context`.
10. Read the second turn's `tool_interactions`, find `memory_time_search`, and collect `returned_memory_ids`.
11. Read debug JSONL/text files and assert one `time_search` success record and one `consolidation_trigger` with `trigger_reason=exit`.
12. Return only non-secret assertion data: event ID/times, returned IDs, boolean metadata/debug checks, and final answer text.

Never return prompts containing API configuration, request headers, or environment values. On failure, write only model response excerpts and the temporary artifact paths.

- [ ] **Step 5: Run all non-live tests before spending API calls**

Run: `python -m unittest discover -s test -p "test_*.py" -v`

Expected: all non-live tests PASS and the guarded real-API test is SKIPPED.

- [ ] **Step 6: Run the approved real API scenario**

Run in PowerShell:

```powershell
$env:GUGA_RUN_LIVE_API_TESTS="1"
python -m unittest test.test_temporal_memory_live_api -v
```

Expected: PASS. The first interaction writes turn temporal metadata, exit creates the canonical event, restart calls `memory_time_search`, and the time tool returns that event ID.

- [ ] **Step 7: Inspect real artifacts for semantic correctness**

Run the script directly for its sanitized report:

```powershell
python scripts/live_memory_api_validation.py --scenario temporal-e2e
```

Expected JSON fields:

```json
{
  "event_start_at": "2026-08-20T00:00:00+08:00",
  "event_end_at": "2026-08-21T00:00:00+08:00",
  "first_turn_has_temporal_context": true,
  "second_turn_has_temporal_context": true,
  "debug_has_time_search": true,
  "debug_has_exit_consolidation": true
}
```

Also inspect the reported temporary session, semantic-event, archival, event-summary, user-model, consolidation-state, and debug paths. The answer must describe the interview on August 20; wording need not be byte-identical.

- [ ] **Step 8: Commit live acceptance coverage**

```powershell
git add scripts/live_memory_api_validation.py test/test_live_memory_api_validation.py test/test_temporal_memory_live_api.py
git commit -m "test(memory): 覆盖真实时间记忆链路"
```

- [ ] **Step 9: Run final verification and push**

Run: `python -m unittest discover -s test -p "test_*.py" -v`

Expected: all non-live tests PASS; the live test is skipped unless the environment flag remains enabled.

Run: `git status --short`

Expected: no unintended tracked changes.

Run: `git push`

Expected: all implementation commits are present on the current remote branch.
