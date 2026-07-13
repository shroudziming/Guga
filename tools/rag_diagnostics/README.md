# RAG Score Diagnostics

Standalone, read-only diagnostics for persisted Guga RAG indexes. The tool does
not import the `guga` package, call a chat API, or modify index files.

## Run

From the repository root:

```powershell
python tools\rag_diagnostics\analyze_scores.py `
  --index-dir data\benchmarks\longmemeval\runs\live_api_semantic_events_002\cases\852ce960\memory\rag\index `
  --query "what does this article say about AI regulation?" `
  --query "What rights does GDPR give data subjects?" `
  --query "When is my dentist appointment?" `
  --top-k 10 `
  --focus-source-id turn_msg_5c37846c8b `
  --json-output data\benchmarks\longmemeval\runs\live_api_semantic_events_002\gdpr-score-report.json
```

Use at least one relevant query and one clearly unrelated control query. The
report includes score percentiles, standard deviation, Top-K separation, source
concentration, chunk excerpts, and a focused distribution for one source.

`scores_are_tightly_clustered=true` means the Top-K score range is below 0.05.
This is a diagnostic warning, not a universal retrieval-quality threshold.

## Test

```powershell
Set-Location tools\rag_diagnostics
python -m unittest test_analyze_scores.py
```

The current tool deliberately accepts only 128-dimensional indexes produced by
the fallback `HashingEmbedder`. It fails explicitly for other dimensions instead
of pretending to reproduce a Sentence Transformer index with a different model.

## Compare Current Hashing With Controls

`compare_retrievers.py` compares the persisted hashing scores against an
isolated BM25 control and a per-source chunk cap. This does not change the
production retriever.

```powershell
python tools\rag_diagnostics\compare_retrievers.py `
  --index-dir data\benchmarks\longmemeval\runs\live_api_semantic_events_003\cases\852ce960\memory\rag\index `
  --query "What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?" `
  --expect '$350,000' `
  --expect '$400,000' `
  --expect "The regulation does not purport" `
  --top-k 10 `
  --max-per-source 1
```

For this case, persisted hashing ranks the old `$350,000` evidence first, the
new `$400,000` evidence third, and a GDPR chunk fourth. BM25 ranks the two
mortgage passages first and second while moving that GDPR chunk below rank 400.
BM25 still needs lifecycle/time resolution to choose the later `$400,000`
statement over the earlier `$350,000` statement.
