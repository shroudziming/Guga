# Voice Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GPT-SoVITS job failures recoverable without truncating chat and make voice-gap causes observable.

**Architecture:** `TextSentenceBuffer` owns safe segmentation, the TTS worker owns bounded retry and job-local errors, and the audio player owns actual playback timing. The shared cancellation event remains for user cancellation only.

**Tech Stack:** Python 3.11, `unittest`, `urllib`, Windows `winsound`.

## Global Constraints

- Default `GUGA_TTS_SENTENCE_MAX_CHARS` is 48 in Python and PowerShell.
- Retry once after 350 ms only for transport and HTTP 5xx failures.
- Do not set the LLM turn cancellation event for a TTS-job error.
- Do not add byte-level HTTP streaming or a new playback backend.

---

### Task 1: Safe segmentation

**Files:** `guga/voice/sentence_buffer.py`, `scripts/voice_chat_guga_smoke.ps1`, `test/test_voice_pipeline.py`

- [ ] Write tests asserting `TextSentenceBuffer(max_chars=6).feed("一二三四五六。后") == ["一二三四五六。"]`, punctuation-only text is not speakable, and the env default emits 48 characters.
- [ ] Run `python -m unittest discover -s test -p 'test_voice_pipeline.py' -v`; verify the new assertions fail.
- [ ] Add an `is_speakable_text(text: str) -> bool` helper using Unicode letter/number categories; apply it before every emission. Extend forced split through one immediately following terminal boundary and set the Python/PowerShell defaults to 48.
- [ ] Re-run the focused tests; commit with `fix(voice):稳健切分语音片段`.

### Task 2: Retry and isolation

**Files:** `guga/voice/tts_client.py`, `guga/voice/runner.py`, `test/test_voice_pipeline.py`

- [ ] Write a fake TTS client that raises `urllib.error.URLError` once then succeeds; assert exactly two attempts. Write another that permanently fails first job while the session still emits a later text and TTS job.
- [ ] Run `python -m unittest discover -s test -p 'test_voice_pipeline.py' -v`; verify new tests fail.
- [ ] Add recoverable-error classification and a one-time 350 ms retry around `tts_client.synthesize`. Record final errors without setting `cancel_event`.
- [ ] Re-run focused tests; commit with `fix(voice):隔离并重试合成失败`.

### Task 3: Playback-boundary diagnostics

**Files:** `guga/voice/audio_player.py`, `guga/voice/runner.py`, `test/test_voice_pipeline.py`

- [ ] Write a controllable player test that asserts `playback_started` is emitted immediately before `_play` and `playback_finished` immediately after it.
- [ ] Run `python -m unittest discover -s test -p 'test_voice_pipeline.py' -v`; verify new test fails.
- [ ] Carry sequence IDs into queued audio, add optional playback callback, and emit debug records for queue/synthesis/playback events with queue depth and elapsed milliseconds.
- [ ] Re-run `python -m unittest discover -s test -p 'test_voice_pipeline.py' -v`; commit with `feat(voice):记录语音链路时序`.

### Task 4: End-to-end verification

- [ ] Run `python -m unittest discover -s test -p 'test_voice_pipeline.py' -v`.
- [ ] Run `git diff main...HEAD --check`, inspect `git status --short`, then merge to `main`, rerun the same test command from `main`, and push.
