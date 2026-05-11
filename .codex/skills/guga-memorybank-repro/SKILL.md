---
name: guga-memorybank-repro
description: Use when implementing, refactoring, or evaluating Guga's long-term conversational memory mechanism based on the MemoryBank paper. Triggers include MemoryBank, long-term memory, AI companion memory, Ebbinghaus forgetting curve, user portrait, daily event summary, memory retrieval/update, memory storage, probing questions, or reproducing a paper memory mechanism inside the Guga repository. Use this skill to map the paper's storage, retrieval, summarization, user portrait, and forgetting-curve update mechanisms onto Guga's ChatSession, MemoryManager, RagPipeline, and local JSONL memory files.
---

# Guga MemoryBank Reproduction

## Objective

Implement a Guga-native reproduction of the MemoryBank paper's long-term memory mechanism.

Preserve the paper's core mechanism while adapting it to Guga's current local CLI architecture:

- memory storage
- memory retrieval
- memory updating with Ebbinghaus forgetting curve
- daily and global event summaries
- daily and global user portrait
- memory-augmented prompt injection
- probing-style evaluation

Do not reproduce unrelated paper components unless explicitly requested. Do not implement SiliconFriend psychological LoRA tuning by default.

## Initial Checks

Before editing code:

1. Confirm current branch and git status.
2. Read relevant local docs:
   - `README.md`
   - `CLAUDE.md`
   - `Notes/rag_current_implementation_learning_notes.md` if available from the workspace root
   - memory-related notes if available
3. Inspect current backend memory code:
   - `guga/chat/session.py`
   - `guga/memory/manager.py`
   - `guga/memory/profile_store.py`
   - `guga/rag/pipeline.py`
   - `guga/rag/faiss_store.py`
   - `guga/types.py`
4. Do not modify frontend files.

## Paper Mechanism To Preserve

MemoryBank has three pillars:

1. Memory Storage
   - chronological multi-turn conversation records
   - daily event summaries
   - global event summaries
   - daily user personality and emotion analysis
   - global user portrait

2. Memory Retrieval
   - treat conversation turns and event summaries as memory pieces
   - encode memory pieces into vectors
   - index with FAISS or an equivalent vector store
   - encode current conversation context as the query
   - retrieve relevant memories for prompt augmentation

3. Memory Updating
   - use Ebbinghaus forgetting curve: `R = exp(-t / S)`
   - `R`: retention score
   - `t`: elapsed time since last learning or recall
   - `S`: memory strength
   - initialize `S = 1`
   - when memory is recalled, increase `S` by 1 and reset elapsed time
   - older, unrecalled, low-strength memories should decay or be deprioritized

## Guga Mapping

Map paper concepts to Guga modules as follows:

| MemoryBank concept | Guga implementation target |
|---|---|
| Raw conversations | `data/memory/sessions/*.jsonl` |
| Memory pieces | archival records, session user turns, event summaries |
| Daily event summary | new JSONL store under `data/memory/` |
| Global event summary | profile or dedicated summary file |
| User portrait | `data/memory/profile.json` |
| Dense retrieval | existing `RagPipeline` and `VectorStore` |
| FAISS fallback | existing vector store fallback behavior |
| Memory update | `MemoryManager.finalize_turn` or a dedicated updater |
| Forgetting curve | scorer/updater used before retrieval and after recall |
| Memory-augmented prompt | `MemoryManager.compose_system_prompt` |

Prefer extending existing modules over creating disconnected parallel systems.

## Implementation Policy

Implement in small, testable phases. Keep the CLI and current RAG loop running after every phase.

### Phase 1: Schema And Storage

Add or extend memory record fields:

```json
{
  "id": "mem_xxx",
  "type": "episodic|event_summary|user_portrait|preference|profile_fact",
  "summary": "...",
  "raw_excerpt": "...",
  "source_session_id": "...",
  "source_message_ids": ["msg_xxx"],
  "created_at": "...",
  "last_recalled_at": "...",
  "memory_strength": 1,
  "retention": 1.0,
  "importance": 0.0,
  "confidence": 0.0,
  "status": "active|decayed|superseded|deleted"
}
```

Keep backward compatibility with existing `archival_memory.jsonl`.

### Phase 2: MemoryBank Retrieval

Use current Guga semantic retrieval as the base.

Retrieval should consider:

- raw user conversation turns
- archival memories
- daily event summaries
- global event summaries
- user portrait entries when relevant

Ranking should combine:

```text
final_score =
  semantic_score
  * retention_score
  + lexical_score
  + importance_bonus
  + confidence_bonus
  + recency_bonus
```

Avoid making old memories disappear immediately. First deprioritize them through retention score. Mark memories as `decayed` only when policy requires it.

### Phase 3: Ebbinghaus Memory Updating

Implement retention calculation:

```python
retention = exp(-elapsed_days / max(memory_strength, 1))
```

Operational rules:

- On first write: `memory_strength = 1`
- On retrieval hit used in prompt: `memory_strength += 1`
- On recall: update `last_recalled_at = now`
- Recompute retention during retrieval or periodic maintenance
- If retention falls below threshold, either lower ranking or mark `status = decayed`

Keep the formula centralized and unit tested.

### Phase 4: Event Summary And User Portrait

Use an LLM or small model only for summarization and classification tasks. This should not block the main chat response unless explicitly requested.

Daily event summary prompt adapted from the paper:

```text
Summarize the events and key information in the following dialogue.
Return concise factual bullet points. Avoid unsupported inference.
```

Daily personality prompt adapted from the paper:

```text
Based on the following dialogue, summarize the user's personality traits, preferences, and emotional state.
Separate stable traits from temporary emotions.
```

Global portrait prompt adapted from the paper:

```text
The following are user traits and emotions observed across multiple days.
Provide a concise, general, non-duplicative user portrait.
Preserve uncertainty when evidence is weak.
```

Store summaries separately from raw conversation logs. Do not overwrite raw logs.

### Phase 5: Prompt Injection

The final system prompt should have explicit sections:

```text
[Base Persona]
...

[User Portrait]
...

[Relevant Event Summaries]
...

[Relevant Conversation Memories]
...

[Current Rule]
Use memory only when relevant. If memory is absent or uncertain, say so directly.
Do not invent past events.
```

Keep source ids and scores available in debug logs.

### Phase 6: Evaluation

Create probing-style tests modeled after the paper:

1. Positive recall
   - seed a memory
   - ask later about the remembered fact
   - verify retrieved memory contains expected source

2. Negative recall
   - ask about something never discussed
   - verify system does not invent memory

3. Portrait personalization
   - seed user traits or preferences
   - ask for suggestions
   - verify answer uses portrait naturally

4. Forgetting/update
   - old unrecalled memory should rank lower
   - recalled memory should gain strength
   - retention should be deterministic

Use or extend tests under `test/`.

## Engineering Constraints

- Do not touch frontend code.
- Preserve current CLI behavior.
- Preserve the existing RAG closed loop.
- Maintain backward compatibility with existing JSONL memory files.
- Keep model calls optional or asynchronous where possible.
- Prefer deterministic rules for scoring and forgetting.
- Use LLM calls for summarization and classification, not for core state mutation without validation.
- Keep debug logs detailed enough to trace:
  - query
  - retrieved memory ids
  - retention scores
  - memory strength updates
  - prompt sections

## Recommended Module Shape

Prefer these modules if adding files:

```text
guga/memory/forgetting.py
guga/memory/event_summary_store.py
guga/memory/portrait.py
guga/memory/scorer.py
```

Avoid putting all logic into `manager.py`. `MemoryManager` should coordinate, not own every algorithm.

## Validation Checklist

After implementation, run:

```text
python -m unittest discover -s test
```

Also verify:

- session JSONL still records user and assistant messages
- archival memory still writes correctly
- vector index can rebuild
- relevant memory appears in system prompt
- debug logs include memory ids and retention/update details
- no frontend files changed

## Final Report

When finishing a MemoryBank reproduction task, report:

1. What paper mechanisms were implemented.
2. What was intentionally simplified.
3. Files changed.
4. Tests run.
5. Known limitations.
6. Recommended next step.
