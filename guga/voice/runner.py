from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from guga.voice.audio_player import AudioData
from guga.voice.metrics import VoiceMetrics, VoiceMetricsSummary
from guga.voice.sentence_buffer import TextSentenceBuffer
from guga.voice.tts_client import TtsClient


class StreamingSession(Protocol):
    def reply_stream(self, user_input: str, cancel_event: threading.Event | None = None):
        ...


class AudioPlayer(Protocol):
    def start(self) -> None:
        ...

    def enqueue(self, audio: AudioData) -> None:
        ...

    def join(self, timeout: float | None = None) -> None:
        ...

    def stop(self, clear: bool = False) -> None:
        ...


@dataclass(frozen=True)
class _TtsJob:
    sequence_id: int
    text: str


class VoiceChatRunner:
    def __init__(
        self,
        *,
        session: StreamingSession,
        tts_client: TtsClient,
        audio_player: AudioPlayer,
        text_sink,
        sentence_buffer: TextSentenceBuffer | None = None,
        metrics: VoiceMetrics | None = None,
        max_queue_size: int = 8,
        raise_tts_errors: bool = True,
    ) -> None:
        self.session = session
        self.tts_client = tts_client
        self.audio_player = audio_player
        self.text_sink = text_sink
        self.sentence_buffer = sentence_buffer or TextSentenceBuffer()
        self.metrics = metrics or VoiceMetrics()
        self.raise_tts_errors = raise_tts_errors
        self._queue: queue.Queue[_TtsJob | None] = queue.Queue(maxsize=max_queue_size)
        self._errors: list[BaseException] = []

    def run_turn(
        self,
        user_input: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> VoiceMetricsSummary:
        cancel_event = cancel_event or threading.Event()
        self.metrics.turn_started()
        self.audio_player.start()

        worker = threading.Thread(target=self._tts_worker, args=(cancel_event,), name="guga-tts-worker", daemon=True)
        worker.start()

        sequence_id = 0
        stream = self.session.reply_stream(user_input, cancel_event=cancel_event)
        try:
            for chunk in stream:
                if cancel_event.is_set():
                    break
                self.metrics.text_chunk_received()
                self.text_sink(chunk)
                for sentence in self.sentence_buffer.feed(chunk):
                    sequence_id += 1
                    self._enqueue_sentence(sequence_id, sentence)

            for sentence in self.sentence_buffer.flush():
                sequence_id += 1
                self._enqueue_sentence(sequence_id, sentence)
        except BaseException:
            cancel_event.set()
            self.audio_player.stop(clear=True)
            raise
        finally:
            self._queue.put(None)
            self._queue.join()
            worker.join(timeout=2.0)
            self.audio_player.join()
            self.audio_player.stop(clear=False)
            self.metrics.turn_finished()

        if self._errors and self.raise_tts_errors:
            raise RuntimeError("TTS worker failed") from self._errors[0]

        return self.metrics.summary()

    def _enqueue_sentence(self, sequence_id: int, sentence: str) -> None:
        self.metrics.sentence_queued(sentence)
        self._queue.put(_TtsJob(sequence_id=sequence_id, text=sentence))

    def _tts_worker(self, cancel_event: threading.Event) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if cancel_event.is_set():
                    continue

                started = time.perf_counter()
                audio = self.tts_client.synthesize(job.text)
                elapsed = time.perf_counter() - started
                self.metrics.tts_finished(
                    sequence_id=job.sequence_id,
                    text=job.text,
                    elapsed_seconds=elapsed,
                    audio_seconds=audio.duration_seconds,
                )
                if cancel_event.is_set():
                    continue
                self.audio_player.enqueue(audio)
                self.metrics.audio_enqueued(job.sequence_id)
            except BaseException as exc:
                self._errors.append(exc)
                cancel_event.set()
            finally:
                self._queue.task_done()
