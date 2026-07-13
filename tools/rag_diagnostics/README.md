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
