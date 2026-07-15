from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from guga.persona import PersonaExpression, PersonaOutputParser, PersonaText
from guga.voice.audio_player import AudioData
from guga.voice.metrics import VoiceMetrics, VoiceMetricsSummary
from guga.voice.sentence_buffer import TextSentenceBuffer
from guga.voice.text_filter import SpokenTextFilter
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
    split_reason: str


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
        expression_tags: tuple[str, ...] = (),
        expression_sink=None,
    ) -> None:
        self.session = session
        self.tts_client = tts_client
        self.audio_player = audio_player
        self.text_sink = text_sink
        self.sentence_buffer = sentence_buffer or TextSentenceBuffer()
        self.metrics = metrics or VoiceMetrics()
        self.raise_tts_errors = raise_tts_errors
        self.expression_tags = expression_tags
        self.expression_sink = expression_sink
        self.max_queue_size = max_queue_size

    def run_turn(
        self,
        user_input: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> VoiceMetricsSummary:
        cancel_event = cancel_event or threading.Event()
        self.metrics.turn_started()
        self.audio_player.start()

        tts_queue: queue.Queue[_TtsJob | None] = queue.Queue(maxsize=self.max_queue_size)
        errors: list[BaseException] = []
        worker_done = threading.Event()
        worker = threading.Thread(
            target=self._tts_worker,
            args=(cancel_event, tts_queue, errors, worker_done),
            name="guga-tts-worker",
            daemon=True,
        )
        worker.start()

        sequence_id = 0
        persona_parser = PersonaOutputParser(self.expression_tags)
        spoken_text_filter = SpokenTextFilter()

        def route_persona_events(events, *, spoken: bool = True) -> None:
            nonlocal sequence_id
            for event in events:
                if isinstance(event, PersonaExpression):
                    if self.expression_sink is not None:
                        self.expression_sink(event.tag)
                    continue
                if isinstance(event, PersonaText):
                    self.text_sink(event.text)
                    if not spoken:
                        continue
                    spoken_chunk = spoken_text_filter.feed(event.text)
                    for segment in self.sentence_buffer.feed_segments(spoken_chunk):
                        sequence_id += 1
                        self._enqueue_sentence(
                            tts_queue,
                            cancel_event,
                            sequence_id,
                            segment.text,
                            segment.split_reason,
                        )

        stream = self.session.reply_stream(user_input, cancel_event=cancel_event)
        try:
            for chunk in stream:
                if cancel_event.is_set():
                    break
                self.metrics.text_chunk_received()
                route_persona_events(persona_parser.feed(chunk))

            if not cancel_event.is_set():
                route_persona_events(persona_parser.flush())
                for segment in self.sentence_buffer.flush_segments():
                    sequence_id += 1
                    self._enqueue_sentence(
                        tts_queue,
                        cancel_event,
                        sequence_id,
                        segment.text,
                        segment.split_reason,
                    )
            else:
                route_persona_events(persona_parser.flush(), spoken=False)
                self.sentence_buffer.flush()
        except BaseException:
            cancel_event.set()
            raise
        finally:
            self._finish_tts_turn(cancel_event, tts_queue, worker, worker_done)
            self.metrics.turn_finished()

        if errors and self.raise_tts_errors:
            raise RuntimeError("TTS worker failed") from errors[0]

        return self.metrics.summary()

    def _enqueue_sentence(
        self,
        tts_queue: queue.Queue[_TtsJob | None],
        cancel_event: threading.Event,
        sequence_id: int,
        sentence: str,
        split_reason: str,
    ) -> None:
        job = _TtsJob(sequence_id=sequence_id, text=sentence, split_reason=split_reason)
        while not cancel_event.is_set():
            try:
                tts_queue.put(job, timeout=0.05)
            except queue.Full:
                continue
            self.metrics.sentence_queued(sentence)
            return

    def _tts_worker(
        self,
        cancel_event: threading.Event,
        tts_queue: queue.Queue[_TtsJob | None],
        errors: list[BaseException],
        worker_done: threading.Event,
    ) -> None:
        try:
            while True:
                job = tts_queue.get()
                try:
                    if job is None:
                        return
                    if cancel_event.is_set():
                        continue

                    started = time.perf_counter()
                    audio = self.tts_client.synthesize(job.text)
                    elapsed = time.perf_counter() - started
                    if cancel_event.is_set():
                        continue
                    self.metrics.tts_finished(
                        sequence_id=job.sequence_id,
                        text=job.text,
                        elapsed_seconds=elapsed,
                        audio_seconds=audio.duration_seconds,
                    )
                    self._debug_voice_playback_start(job)
                    self.audio_player.enqueue(audio)
                    self.metrics.audio_enqueued(job.sequence_id)
                except BaseException as exc:
                    if not cancel_event.is_set():
                        errors.append(exc)
                        cancel_event.set()
                finally:
                    tts_queue.task_done()
        finally:
            worker_done.set()

    def _finish_tts_turn(
        self,
        cancel_event: threading.Event,
        tts_queue: queue.Queue[_TtsJob | None],
        worker: threading.Thread,
        worker_done: threading.Event,
    ) -> None:
        if cancel_event.is_set():
            self._cancel_tts_turn(tts_queue)
            return

        while not cancel_event.is_set():
            try:
                tts_queue.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        if cancel_event.is_set():
            self._cancel_tts_turn(tts_queue)
            return

        while not worker_done.wait(timeout=0.05):
            if cancel_event.is_set():
                self._cancel_tts_turn(tts_queue)
                return

        tts_queue.join()
        worker.join()
        if cancel_event.is_set():
            self.audio_player.stop(clear=True)
            return
        self.audio_player.join()
        self.audio_player.stop(clear=False)

    def _cancel_tts_turn(self, tts_queue: queue.Queue[_TtsJob | None]) -> None:
        try:
            self.audio_player.stop(clear=True)
        finally:
            while True:
                try:
                    tts_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    tts_queue.task_done()
            tts_queue.put_nowait(None)

    def _debug_voice_playback_start(self, job: _TtsJob) -> None:
        if not bool(getattr(self.session, "debug", False)):
            return
        text = json.dumps(job.text, ensure_ascii=False)
        message = (
            "voice_playback_start "
            f"sequence_id={job.sequence_id} "
            f"token_count={len(job.text)} "
            f"split_reason={job.split_reason} "
            f"text={text}"
        )
        output = f"[DEBUG][VoiceChatRunner][{getattr(self.session, 'session_id', 'unknown_session')}] {message}"
        debug_sink = getattr(self.session, "debug_sink", None)
        if callable(debug_sink):
            debug_sink(output)
            return
        print(output)
