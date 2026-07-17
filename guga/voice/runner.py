from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
from dataclasses import dataclass, replace
from typing import Protocol

from guga.persona import PersonaExpression, PersonaOutputParser, PersonaText
from guga.voice.audio_player import AudioData
from guga.voice.metrics import VoiceMetrics, VoiceMetricsSummary
from guga.voice.sentence_buffer import TextSentenceBuffer
from guga.voice.text_filter import SpokenTextFilter
from guga.voice.tts_client import TtsClient


def is_retryable_tts_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return 500 <= error.code < 600
    return isinstance(error, urllib.error.URLError)


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
        self._turn_started_monotonic: float | None = None

    def run_turn(
        self,
        user_input: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> VoiceMetricsSummary:
        cancel_event = cancel_event or threading.Event()
        self._turn_started_monotonic = time.monotonic()
        set_playback_event_callback = getattr(self.audio_player, "set_playback_event_callback", None)
        if callable(set_playback_event_callback):
            set_playback_event_callback(self._debug_voice_playback_event)
        self.metrics.turn_started()
        self.audio_player.start()

        tts_queue: queue.Queue[_TtsJob | None] = queue.Queue(maxsize=self.max_queue_size)
        errors: list[BaseException] = []
        worker_done = threading.Event()
        publish_gate = threading.Lock()
        publish_closed = threading.Event()
        worker = threading.Thread(
            target=self._tts_worker,
            args=(
                cancel_event,
                tts_queue,
                errors,
                worker_done,
                publish_gate,
                publish_closed,
            ),
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
            self._finish_tts_turn(
                cancel_event,
                tts_queue,
                worker,
                worker_done,
                publish_gate,
                publish_closed,
            )
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
            self._debug_voice_event(
                "voice_queue",
                sequence_id=sequence_id,
                queue_depth=tts_queue.qsize(),
                split_reason=split_reason,
                text=sentence,
            )
            return

    def _tts_worker(
        self,
        cancel_event: threading.Event,
        tts_queue: queue.Queue[_TtsJob | None],
        errors: list[BaseException],
        worker_done: threading.Event,
        publish_gate: threading.Lock,
        publish_closed: threading.Event,
    ) -> None:
        try:
            while True:
                job = tts_queue.get()
                try:
                    if job is None:
                        return
                    if cancel_event.is_set():
                        continue

                    queue_depth = tts_queue.qsize()
                    self._debug_voice_event(
                        "voice_synthesis_started",
                        sequence_id=job.sequence_id,
                        queue_depth=queue_depth,
                    )
                    started = time.perf_counter()
                    audio = self._synthesize_with_retry(job.text)
                    elapsed = time.perf_counter() - started
                    self._debug_voice_event(
                        "voice_synthesis_finished",
                        sequence_id=job.sequence_id,
                        queue_depth=tts_queue.qsize(),
                        synthesis_ms=int(elapsed * 1000),
                        split_reason=job.split_reason,
                        text=job.text,
                    )
                    with publish_gate:
                        if publish_closed.is_set() or cancel_event.is_set():
                            continue
                        self.metrics.tts_finished(
                            sequence_id=job.sequence_id,
                            text=job.text,
                            elapsed_seconds=elapsed,
                            audio_seconds=audio.duration_seconds,
                        )
                        if publish_closed.is_set() or cancel_event.is_set():
                            continue
                        self.audio_player.enqueue(replace(audio, sequence_id=job.sequence_id))
                        self.metrics.audio_enqueued(job.sequence_id)
                except BaseException as exc:
                    if not cancel_event.is_set():
                        errors.append(exc)
                finally:
                    tts_queue.task_done()
        finally:
            worker_done.set()

    def _synthesize_with_retry(self, text: str) -> AudioData:
        try:
            return self.tts_client.synthesize(text)
        except BaseException as exc:
            if not is_retryable_tts_error(exc):
                raise
            time.sleep(0.35)
            return self.tts_client.synthesize(text)

    def _finish_tts_turn(
        self,
        cancel_event: threading.Event,
        tts_queue: queue.Queue[_TtsJob | None],
        worker: threading.Thread,
        worker_done: threading.Event,
        publish_gate: threading.Lock,
        publish_closed: threading.Event,
    ) -> None:
        if cancel_event.is_set():
            self._cancel_tts_turn(
                cancel_event,
                tts_queue,
                publish_gate,
                publish_closed,
            )
            return

        while not cancel_event.is_set():
            try:
                tts_queue.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        if cancel_event.is_set():
            self._cancel_tts_turn(
                cancel_event,
                tts_queue,
                publish_gate,
                publish_closed,
            )
            return

        while not worker_done.wait(timeout=0.05):
            if cancel_event.is_set():
                self._cancel_tts_turn(
                    cancel_event,
                    tts_queue,
                    publish_gate,
                    publish_closed,
                )
                return

        tts_queue.join()
        worker.join()
        if cancel_event.is_set():
            with publish_gate:
                cancel_event.set()
                publish_closed.set()
                self.audio_player.stop(clear=True)
            return
        self.audio_player.join()
        self.audio_player.stop(clear=False)

    def _cancel_tts_turn(
        self,
        cancel_event: threading.Event,
        tts_queue: queue.Queue[_TtsJob | None],
        publish_gate: threading.Lock,
        publish_closed: threading.Event,
    ) -> None:
        try:
            with publish_gate:
                cancel_event.set()
                publish_closed.set()
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

    def _debug_voice_playback_event(self, event: str, sequence_id: int | None, queue_depth: int) -> None:
        self._debug_voice_event(
            f"voice_{event}",
            sequence_id=sequence_id,
            queue_depth=queue_depth,
        )

    def _debug_voice_event(
        self,
        event: str,
        *,
        sequence_id: int | None,
        queue_depth: int,
        **details: object,
    ) -> None:
        if not bool(getattr(self.session, "debug", False)):
            return
        turn_started_monotonic = self._turn_started_monotonic
        turn_ms = 0 if turn_started_monotonic is None else int((time.monotonic() - turn_started_monotonic) * 1000)
        fields = [
            event,
            f"sequence_id={sequence_id}",
            f"queue_depth={queue_depth}",
            f"turn_ms={turn_ms}",
        ]
        for key, value in details.items():
            if key == "text":
                fields.append(f"token_count={len(str(value))}")
                fields.append(f"text={json.dumps(value, ensure_ascii=False)}")
            else:
                fields.append(f"{key}={value}")
        message = " ".join(fields)
        output = f"[DEBUG][VoiceChatRunner][{getattr(self.session, 'session_id', 'unknown_session')}] {message}"
        debug_sink = getattr(self.session, "debug_sink", None)
        if callable(debug_sink):
            debug_sink(output)
            return
        print(output)
