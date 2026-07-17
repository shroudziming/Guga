# Voice resilience design

## Goal

Prevent malformed short segments or recoverable GPT-SoVITS failures from truncating a chat response, and record enough timing data to attribute inter-segment silence to synthesis supply or playback handoff.

## Approved scope

1. Default sentence limit is 48 characters. When a forced split is followed immediately by a terminal mark, include that mark in the same segment.
2. Do not enqueue text without Chinese, Latin-letter, or numeric content. This excludes empty and punctuation-only fragments such as `。` and `——`.
3. Retry a recoverable synthesis failure exactly once after 350 ms. HTTP 5xx, endpoint unavailable, and timeout failures are recoverable; HTTP 4xx and validation/value errors are not.
4. TTS failure is job-local: log it and continue the LLM stream plus later jobs. Only user cancellation sets the shared turn cancellation event.
5. Log per-job queue, synthesis, and actual playback timing with sequence ID, queue depth, and turn-relative monotonic time.

## Non-goals

This change neither implements byte-level HTTP audio streaming nor replaces Windows `winsound` playback. The current client still receives one complete GPT-SoVITS response per sentence.

## Diagnostic interpretation

- If `tts_finished(N+1)` is later than `playback_finished(N)`, synthesis did not supply the next audio in time.
- If `tts_finished(N+1)` is earlier than `playback_finished(N)`, but a large gap remains before `playback_started(N+1)`, playback handoff is responsible.

## Verification

Tests cover boundary retention, punctuation filtering, one transient retry, non-cancellation after a failed job, and actual playback boundary callbacks. Run the full voice test module after implementation.
