from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VoiceMetricsSummary:
    sentences: int
    tts_seconds: float
    audio_seconds: float
    average_rtf: float | None
    first_text_ms: int | None
    first_sentence_ms: int | None
    first_audio_ms: int | None
    total_ms: int | None


class VoiceMetrics:
    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._first_text_at: float | None = None
        self._first_sentence_at: float | None = None
        self._first_audio_at: float | None = None
        self._sentences = 0
        self._tts_seconds = 0.0
        self._audio_seconds = 0.0

    def turn_started(self) -> None:
        self._started_at = self._clock()
        self._ended_at = None
        self._first_text_at = None
        self._first_sentence_at = None
        self._first_audio_at = None
        self._sentences = 0
        self._tts_seconds = 0.0
        self._audio_seconds = 0.0

    def text_chunk_received(self) -> None:
        if self._first_text_at is None:
            self._first_text_at = self._clock()

    def sentence_queued(self, text: str) -> None:
        if not text:
            return
        if self._first_sentence_at is None:
            self._first_sentence_at = self._clock()
        self._sentences += 1

    def tts_finished(self, sequence_id: int, text: str, elapsed_seconds: float, audio_seconds: float) -> None:
        _ = sequence_id, text
        self._clock()
        self._tts_seconds += elapsed_seconds
        self._audio_seconds += audio_seconds

    def audio_enqueued(self, sequence_id: int) -> None:
        _ = sequence_id
        if self._first_audio_at is None:
            self._first_audio_at = self._clock()

    def turn_finished(self) -> None:
        self._ended_at = self._clock()

    def summary(self) -> VoiceMetricsSummary:
        total_ms = None
        if self._started_at is not None and self._ended_at is not None:
            total_ms = _elapsed_ms(self._started_at, self._ended_at)

        average_rtf = None
        if self._audio_seconds > 0:
            average_rtf = self._tts_seconds / self._audio_seconds

        return VoiceMetricsSummary(
            sentences=self._sentences,
            tts_seconds=self._tts_seconds,
            audio_seconds=self._audio_seconds,
            average_rtf=average_rtf,
            first_text_ms=_optional_elapsed_ms(self._started_at, self._first_text_at),
            first_sentence_ms=_optional_elapsed_ms(self._started_at, self._first_sentence_at),
            first_audio_ms=_optional_elapsed_ms(self._started_at, self._first_audio_at),
            total_ms=total_ms,
        )


def _optional_elapsed_ms(started_at: float | None, ended_at: float | None) -> int | None:
    if started_at is None or ended_at is None:
        return None
    return _elapsed_ms(started_at, ended_at)


def _elapsed_ms(started_at: float, ended_at: float) -> int:
    return int((ended_at - started_at) * 1000)
