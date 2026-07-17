from __future__ import annotations

import os
import threading
import time
import unittest
from collections.abc import Iterator

from guga.voice.audio_player import AudioData, NullAudioPlayer, WavAudioPlayer, audio_player_from_env
from guga.voice.metrics import VoiceMetrics
from guga.voice.runner import VoiceChatRunner
from guga.voice.sentence_buffer import TextSentenceBuffer, sentence_buffer_from_env
from guga.voice.text_filter import SpokenTextFilter
from guga.voice.tool_mode import configure_voice_tool_mode
from guga.voice.tts_client import GptSoVitsConfig, GptSoVitsHttpClient, prewarm_tts_client


class SentenceBufferTest(unittest.TestCase):
    def test_buffers_chunks_until_sentence_boundary(self) -> None:
        buffer = TextSentenceBuffer()

        self.assertEqual(buffer.feed("你好，"), [])
        self.assertEqual(buffer.feed("我是咕"), [])
        self.assertEqual(buffer.feed("嘎。今天"), ["你好，我是咕嘎。"])
        self.assertEqual(buffer.feed("继续聊"), [])
        self.assertEqual(buffer.flush(), ["今天继续聊"])

    def test_forces_long_text_without_boundary(self) -> None:
        buffer = TextSentenceBuffer(max_chars=6)

        self.assertEqual(buffer.feed("一二三四五六七八"), ["一二三四五六"])
        self.assertEqual(buffer.flush(), ["七八"])

    def test_forced_split_includes_immediately_following_terminal_boundary(self) -> None:
        buffer = TextSentenceBuffer(max_chars=6)

        self.assertEqual(buffer.feed("一二三四五六"), [])
        self.assertEqual(buffer.feed("。"), ["一二三四五六。"])

    def test_forced_split_keeps_terminal_boundary_in_same_chunk(self) -> None:
        self.assertEqual(TextSentenceBuffer(max_chars=6).feed("一二三四五六。后"), ["一二三四五六。"])

    def test_punctuation_only_text_is_not_speakable(self) -> None:
        self.assertEqual(TextSentenceBuffer().feed("。！？；!?;"), [])

    def test_reports_split_reasons_for_debugging(self) -> None:
        buffer = TextSentenceBuffer(max_chars=6)

        boundary_segments = buffer.feed_segments("咕咕嘎嘎！是你")
        max_segments = buffer.feed_segments("是你我刚刚看到")
        flush_segments = buffer.flush_segments()

        self.assertEqual([segment.text for segment in boundary_segments], ["咕咕嘎嘎！"])
        self.assertEqual(boundary_segments[0].split_reason, "boundary:！")
        self.assertEqual([segment.text for segment in max_segments], ["是你是你我刚"])
        self.assertEqual(max_segments[0].split_reason, "max_chars:6")
        self.assertEqual([segment.text for segment in flush_segments], ["刚看到"])
        self.assertEqual(flush_segments[0].split_reason, "flush")

    def test_voice_env_defaults_to_48_character_split(self) -> None:
        buffer = sentence_buffer_from_env({})

        self.assertEqual(buffer.feed("一" * 49), ["一" * 48])


class SpokenTextFilterTest(unittest.TestCase):
    def test_removes_parenthesized_text(self) -> None:
        text_filter = SpokenTextFilter()

        self.assertEqual(text_filter.feed("咕嘎！（眼睛放光）汉堡。"), "咕嘎！汉堡。")

    def test_tracks_parenthesized_text_across_chunks(self) -> None:
        text_filter = SpokenTextFilter()

        self.assertEqual(text_filter.feed("咕嘎（眼睛"), "咕嘎")
        self.assertEqual(text_filter.feed("放光）好吃。"), "好吃。")


class VoiceMetricsTest(unittest.TestCase):
    def test_records_tts_duration_audio_duration_and_rtf(self) -> None:
        clock_values = iter([10.0, 10.2, 10.5, 11.5, 12.0])
        metrics = VoiceMetrics(clock=lambda: next(clock_values))

        metrics.turn_started()
        metrics.text_chunk_received()
        metrics.sentence_queued("你好。")
        metrics.tts_finished(sequence_id=1, text="你好。", elapsed_seconds=1.0, audio_seconds=2.0)
        metrics.audio_enqueued(sequence_id=1)

        summary = metrics.summary()

        self.assertEqual(summary.sentences, 1)
        self.assertEqual(summary.audio_seconds, 2.0)
        self.assertEqual(summary.tts_seconds, 1.0)
        self.assertEqual(summary.average_rtf, 0.5)
        self.assertEqual(summary.first_text_ms, 199)
        self.assertEqual(summary.first_audio_ms, 2000)


class FakeSession:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def reply_stream(self, user_input: str, cancel_event: threading.Event | None = None) -> Iterator[str]:
        _ = user_input
        for chunk in self.chunks:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield chunk


class CancelAfterOneChunkSession:
    def reply_stream(self, user_input: str, cancel_event: threading.Event | None = None) -> Iterator[str]:
        _ = user_input
        yield "未完成"
        if cancel_event is not None:
            cancel_event.set()


class DebugFakeSession(FakeSession):
    def __init__(self, chunks: list[str], logs: list[str]) -> None:
        super().__init__(chunks)
        self.debug = True
        self.debug_sink = logs.append
        self.session_id = "debug_session"


class MultiTurnSession:
    def __init__(self, turns: list[list[str]]) -> None:
        self.turns = turns
        self.turn_index = 0

    def reply_stream(self, user_input: str, cancel_event: threading.Event | None = None) -> Iterator[str]:
        _ = user_input
        chunks = self.turns[self.turn_index]
        self.turn_index += 1
        for chunk in chunks:
            yield chunk
            if self.turn_index == 1 and cancel_event is not None:
                cancel_event.set()


class FakeTtsClient:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def synthesize(self, text: str) -> AudioData:
        self.requests.append(text)
        return AudioData(
            data=f"audio:{text}".encode("utf-8"),
            sample_rate=32000,
            channels=1,
            sample_width=2,
            duration_seconds=0.5,
            media_type="wav",
        )


class FakeAudioPlayer:
    def __init__(self) -> None:
        self.items: list[bytes] = []
        self.started = False
        self.stopped = False
        self.audio_enqueued = threading.Event()
        self.cancelled = threading.Event()
        self.cleanup_calls: list[str] = []

    def start(self) -> None:
        self.started = True

    def enqueue(self, audio: AudioData) -> None:
        self.items.append(audio.data)
        self.audio_enqueued.set()

    def stop(self, clear: bool = False) -> None:
        self.cleanup_calls.append(f"stop:{clear}")
        if clear:
            self.items.clear()
            self.cancelled.set()
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        _ = timeout
        self.cleanup_calls.append("join")


class GptSoVitsHttpClientTest(unittest.TestCase):
    def test_reads_integer_streaming_mode_from_env(self) -> None:
        previous = os.environ.get("GUGA_TTS_STREAMING_MODE")
        try:
            os.environ["GUGA_TTS_STREAMING_MODE"] = "3"

            config = GptSoVitsConfig.from_env()

            self.assertEqual(config.streaming_mode, 3)
        finally:
            if previous is None:
                os.environ.pop("GUGA_TTS_STREAMING_MODE", None)
            else:
                os.environ["GUGA_TTS_STREAMING_MODE"] = previous

    def test_posts_non_streaming_parallel_tts_request(self) -> None:
        requests: list[dict] = []

        def fake_post(url: str, payload: dict, timeout_seconds: float) -> bytes:
            requests.append(
                {
                    "url": url,
                    "payload": payload,
                    "timeout_seconds": timeout_seconds,
                }
            )
            return (
                b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                b"\x01\x00\x01\x00\x00}\x00\x00\x00\xfa\x00\x00"
                b"\x02\x00\x10\x00data\x00\x00\x00\x00"
            )

        client = GptSoVitsHttpClient(
            GptSoVitsConfig(
                endpoint="http://127.0.0.1:9880/tts",
                ref_audio_path="D:/voice/ref.wav",
                prompt_text="参考文本",
            ),
            post_json=fake_post,
        )

        audio = client.synthesize("你好。")

        self.assertEqual(audio.media_type, "wav")
        self.assertEqual(requests[0]["url"], "http://127.0.0.1:9880/tts")
        self.assertEqual(requests[0]["payload"]["text"], "你好。")
        self.assertEqual(requests[0]["payload"]["text_lang"], "zh")
        self.assertEqual(requests[0]["payload"]["prompt_lang"], "zh")
        self.assertEqual(requests[0]["payload"]["ref_audio_path"], "D:/voice/ref.wav")
        self.assertEqual(requests[0]["payload"]["prompt_text"], "参考文本")
        self.assertTrue(requests[0]["payload"]["parallel_infer"])
        self.assertFalse(requests[0]["payload"]["streaming_mode"])

    def test_wraps_raw_pcm_response_as_wav_audio(self) -> None:
        raw_pcm = b"\x00\x00\x00\x00" * 3200

        def fake_post(url: str, payload: dict, timeout_seconds: float) -> bytes:
            _ = url, payload, timeout_seconds
            return raw_pcm

        client = GptSoVitsHttpClient(
            GptSoVitsConfig(
                endpoint="http://127.0.0.1:9880/tts",
                ref_audio_path="D:/voice/ref.wav",
                prompt_text="参考文本",
                media_type="raw",
            ),
            post_json=fake_post,
        )

        audio = client.synthesize("你好。")

        self.assertEqual(audio.media_type, "wav")
        self.assertEqual(audio.sample_rate, 32000)
        self.assertEqual(audio.channels, 1)
        self.assertAlmostEqual(audio.duration_seconds, 0.2, places=2)

    def test_prewarm_synthesizes_short_text_by_default(self) -> None:
        tts = FakeTtsClient()

        result = prewarm_tts_client(tts, {})

        self.assertTrue(result.ok)
        self.assertEqual(tts.requests, ["嗯。"])

    def test_prewarm_can_be_disabled(self) -> None:
        tts = FakeTtsClient()

        result = prewarm_tts_client(tts, {"GUGA_TTS_PREWARM": "0"})

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "disabled")
        self.assertEqual(tts.requests, [])


class AudioPlayerFactoryTest(unittest.TestCase):
    def test_can_disable_real_audio_playback_from_env(self) -> None:
        player = audio_player_from_env({"GUGA_TTS_PLAY_AUDIO": "0"})

        self.assertIsInstance(player, NullAudioPlayer)


class WavAudioPlayerTest(unittest.TestCase):
    def test_cancel_interrupts_current_playback_and_player_can_be_reused(self) -> None:
        class ControllableWavAudioPlayer(WavAudioPlayer):
            def __init__(self) -> None:
                super().__init__()
                self.playback_started = threading.Event()
                self.release_current = threading.Event()
                self.played: list[bytes] = []

            def _play(self, audio: AudioData) -> None:
                self.played.append(audio.data)
                self.playback_started.set()
                self.release_current.wait(timeout=3.0)

            def _stop_current_playback(self) -> None:
                self.release_current.set()

        def audio(label: str) -> AudioData:
            return AudioData(
                data=label.encode("ascii"),
                sample_rate=32000,
                channels=1,
                sample_width=2,
                duration_seconds=1.0,
            )

        player = ControllableWavAudioPlayer()
        player.start()
        player.enqueue(audio("cancelled"))
        self.assertTrue(player.playback_started.wait(timeout=0.5))
        cancelled_worker = player._thread
        self.assertIsNotNone(cancelled_worker)

        cancel_thread = threading.Thread(target=player.stop, kwargs={"clear": True}, daemon=True)
        cancel_thread.start()
        try:
            cancel_thread.join(timeout=0.5)
            self.assertFalse(cancel_thread.is_alive(), "cancel stop waited for synchronous playback")
            self.assertFalse(cancelled_worker.is_alive())
        finally:
            player.release_current.set()
            cancel_thread.join(timeout=2.5)

        player.playback_started.clear()
        player.release_current.clear()
        player.start()
        player.enqueue(audio("next"))
        self.assertTrue(player.playback_started.wait(timeout=0.5))
        player.enqueue(audio("queued"))

        graceful_thread = threading.Thread(target=player.stop, kwargs={"clear": False}, daemon=True)
        graceful_thread.start()
        deadline = time.monotonic() + 0.5
        while player._queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        player.release_current.set()
        graceful_thread.join(timeout=1.0)

        self.assertFalse(graceful_thread.is_alive(), "graceful stop did not drain queued playback")
        self.assertEqual(player.played, [b"cancelled", b"next", b"queued"])

    def test_cancel_retries_interrupt_when_playback_starts_after_first_attempt(self) -> None:
        class WindowedWavAudioPlayer(WavAudioPlayer):
            def __init__(self) -> None:
                super().__init__()
                self.before_playback = threading.Event()
                self.allow_playback = threading.Event()
                self.underlying_playing = threading.Event()
                self.release_current = threading.Event()
                self.interrupt_attempted = threading.Event()
                self.interrupted = threading.Event()
                self.interrupt_attempts = 0
                self.played: list[bytes] = []

            def _play(self, audio: AudioData) -> None:
                self.played.append(audio.data)
                if len(self.played) > 1:
                    return
                self.before_playback.set()
                if not self.allow_playback.wait(timeout=2.0):
                    return
                self.underlying_playing.set()
                try:
                    self.release_current.wait(timeout=2.0)
                finally:
                    self.underlying_playing.clear()

            def _stop_current_playback(self) -> None:
                self.interrupt_attempts += 1
                self.interrupt_attempted.set()
                if self.underlying_playing.is_set():
                    self.interrupted.set()
                    self.release_current.set()

        def audio(label: str) -> AudioData:
            return AudioData(
                data=label.encode("ascii"),
                sample_rate=32000,
                channels=1,
                sample_width=2,
                duration_seconds=1.0,
            )

        player = WindowedWavAudioPlayer()
        player.start()
        player.enqueue(audio("cancelled-window"))
        self.assertTrue(player.before_playback.wait(timeout=0.5))
        cancelled_worker = player._thread
        self.assertIsNotNone(cancelled_worker)
        stop_errors: list[BaseException] = []

        def stop_cancelled_playback() -> None:
            try:
                player.stop(clear=True)
            except BaseException as exc:
                stop_errors.append(exc)

        stop_thread = threading.Thread(target=stop_cancelled_playback, daemon=True)
        stop_thread.start()
        self.assertTrue(player.interrupt_attempted.wait(timeout=0.5))
        player.allow_playback.set()
        stop_thread.join(timeout=0.75)
        stopped_before_deadline = not stop_thread.is_alive()
        try:
            self.assertTrue(stopped_before_deadline, "cancel did not retry a missed playback interrupt")
        finally:
            player.allow_playback.set()
            player.release_current.set()
            stop_thread.join(timeout=2.5)

        self.assertEqual(stop_errors, [])
        self.assertGreaterEqual(player.interrupt_attempts, 2)
        self.assertTrue(player.interrupted.is_set())
        self.assertFalse(cancelled_worker.is_alive())

        player.start()
        player.enqueue(audio("next"))
        player.join()
        player.stop(clear=False)

        self.assertEqual(player.played, [b"cancelled-window", b"next"])


class VoiceToolModeTest(unittest.TestCase):
    def test_voice_chat_disables_tool_path_by_default(self) -> None:
        env: dict[str, str] = {}

        enabled = configure_voice_tool_mode(env)

        self.assertFalse(enabled)
        self.assertEqual(env["Guga_MAX_TOOL_ROUNDS"], "0")

    def test_voice_chat_can_keep_tools_when_explicitly_enabled(self) -> None:
        env = {"GUGA_VOICE_WITH_TOOLS": "1", "Guga_MAX_TOOL_ROUNDS": "2"}

        enabled = configure_voice_tool_mode(env)

        self.assertTrue(enabled)
        self.assertEqual(env["Guga_MAX_TOOL_ROUNDS"], "2")


class VoiceChatRunnerTest(unittest.TestCase):
    def test_cancel_after_worker_check_cannot_publish_audio_or_pollute_reuse(self) -> None:
        class ReusableSession:
            def __init__(self) -> None:
                self.turns = iter((["第一轮。"], ["第二轮。"], ["第三轮。"]))

            def reply_stream(
                self,
                user_input: str,
                cancel_event: threading.Event | None = None,
            ) -> Iterator[str]:
                _ = user_input, cancel_event
                yield from next(self.turns)

        class PublishBarrierMetrics(VoiceMetrics):
            def __init__(self) -> None:
                super().__init__()
                self.first_publish_started = threading.Event()
                self.release_first_publish = threading.Event()
                self._tts_finished_calls = 0

            def tts_finished(
                self,
                sequence_id: int,
                text: str,
                elapsed_seconds: float,
                audio_seconds: float,
            ) -> None:
                self._tts_finished_calls += 1
                if self._tts_finished_calls == 1:
                    self.first_publish_started.set()
                    self.release_first_publish.wait(timeout=2.0)
                super().tts_finished(sequence_id, text, elapsed_seconds, audio_seconds)

        cancel_event = threading.Event()
        metrics = PublishBarrierMetrics()
        player = FakeAudioPlayer()
        runner = VoiceChatRunner(
            session=ReusableSession(),
            tts_client=FakeTtsClient(),
            audio_player=player,
            text_sink=lambda chunk: None,
            metrics=metrics,
        )
        first_result: list[object] = []

        def run_cancelled_turn() -> None:
            try:
                first_result.append(runner.run_turn("first", cancel_event=cancel_event))
            except BaseException as exc:
                first_result.append(exc)

        first_thread = threading.Thread(target=run_cancelled_turn, daemon=True)
        first_thread.start()
        self.assertTrue(metrics.first_publish_started.wait(timeout=0.5))
        cancel_event.set()
        try:
            player.cancelled.wait(timeout=0.2)
            metrics.release_first_publish.set()
            first_thread.join(timeout=0.5)
            self.assertFalse(first_thread.is_alive(), "cancelled turn did not finish after publish barrier released")
            self.assertEqual(len(first_result), 1)
            self.assertFalse(isinstance(first_result[0], BaseException))
            self.assertTrue(player.cancelled.wait(timeout=0.5))
            self.assertEqual(player.items, [])

            second_summary = runner.run_turn("second")
            third_summary = runner.run_turn("third")

            self.assertEqual(second_summary.sentences, 1)
            self.assertEqual(third_summary.sentences, 1)
            self.assertEqual(
                player.items,
                ["audio:第二轮。".encode("utf-8"), "audio:第三轮。".encode("utf-8")],
            )
        finally:
            metrics.release_first_publish.set()
            first_thread.join(timeout=2.0)

    def test_cancel_does_not_wait_for_blocked_tts_and_late_worker_cannot_pollute_reuse(self) -> None:
        class ReusableSession:
            def __init__(self) -> None:
                self.turns = iter((["第一轮。"], ["第二轮。"], ["第三轮。"]))

            def reply_stream(
                self,
                user_input: str,
                cancel_event: threading.Event | None = None,
            ) -> Iterator[str]:
                _ = user_input, cancel_event
                yield from next(self.turns)

        class BlockingFirstTts(FakeTtsClient):
            def __init__(self) -> None:
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.first_finished = threading.Event()

            def synthesize(self, text: str) -> AudioData:
                self.requests.append(text)
                if len(self.requests) == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=3.0)
                    self.first_finished.set()
                    raise RuntimeError("late cancelled TTS failure")
                return AudioData(
                    data=f"audio:{text}".encode("utf-8"),
                    sample_rate=32000,
                    channels=1,
                    sample_width=2,
                    duration_seconds=0.5,
                    media_type="wav",
                )

        cancel_event = threading.Event()
        tts = BlockingFirstTts()
        player = FakeAudioPlayer()
        runner = VoiceChatRunner(
            session=ReusableSession(),
            tts_client=tts,
            audio_player=player,
            text_sink=lambda chunk: None,
        )
        first_result: list[object] = []

        def run_cancelled_turn() -> None:
            try:
                first_result.append(runner.run_turn("first", cancel_event=cancel_event))
            except BaseException as exc:
                first_result.append(exc)

        first_thread = threading.Thread(target=run_cancelled_turn, daemon=True)
        first_thread.start()
        self.assertTrue(tts.first_started.wait(timeout=0.5))
        cancel_event.set()
        try:
            self.assertTrue(player.cancelled.wait(timeout=0.3), "cancel did not stop playback immediately")
            first_thread.join(timeout=0.3)
            self.assertFalse(first_thread.is_alive(), "cancel waited for synchronous TTS")
            self.assertEqual(len(first_result), 1)
            self.assertFalse(isinstance(first_result[0], BaseException))

            second_summary = runner.run_turn("second")
            self.assertEqual(second_summary.sentences, 1)
            self.assertEqual(player.items, ["audio:第二轮。".encode("utf-8")])

            tts.release_first.set()
            self.assertTrue(tts.first_finished.wait(timeout=0.5))
            third_summary = runner.run_turn("third")
            self.assertEqual(third_summary.sentences, 1)
            self.assertEqual(
                player.items,
                ["audio:第二轮。".encode("utf-8"), "audio:第三轮。".encode("utf-8")],
            )
        finally:
            tts.release_first.set()
            first_thread.join(timeout=3.0)

    def test_cancel_clears_audio_already_queued_without_waiting_for_playback(self) -> None:
        player = FakeAudioPlayer()

        class CancelAfterAudioQueuedSession:
            def reply_stream(
                self,
                user_input: str,
                cancel_event: threading.Event | None = None,
            ) -> Iterator[str]:
                _ = user_input
                yield "已经入队。"
                if not player.audio_enqueued.wait(timeout=1.0):
                    raise AssertionError("audio was not enqueued before cancellation")
                if cancel_event is not None:
                    cancel_event.set()

        runner = VoiceChatRunner(
            session=CancelAfterAudioQueuedSession(),
            tts_client=FakeTtsClient(),
            audio_player=player,
            text_sink=lambda chunk: None,
        )

        runner.run_turn("hi", cancel_event=threading.Event())

        self.assertEqual(player.items, [])
        self.assertEqual(player.cleanup_calls, ["stop:True"])

    def test_filters_expression_tags_and_emits_expression_events(self) -> None:
        session = FakeSession(["[hap", "py]你好。[side]（挥手）继续。"])
        printed: list[str] = []
        expressions: list[str] = []
        tts = FakeTtsClient()
        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
            expression_tags=("happy", "side"),
            expression_sink=expressions.append,
        )

        runner.run_turn("hi")

        self.assertEqual("".join(printed), "你好。（挥手）继续。")
        self.assertEqual(tts.requests, ["你好。", "继续。"])
        self.assertEqual(expressions, ["happy", "side"])

    def test_keeps_unknown_and_unterminated_tags_as_visible_text(self) -> None:
        session = FakeSession(["[unknown]你好。[hap"])
        printed: list[str] = []
        tts = FakeTtsClient()
        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
            expression_tags=("happy",),
        )

        runner.run_turn("hi")

        self.assertEqual("".join(printed), "[unknown]你好。[hap")
        self.assertEqual(tts.requests, ["[unknown]你好。", "[hap"])

    def test_partial_tag_state_does_not_leak_after_cancelled_turn(self) -> None:
        session = MultiTurnSession([["[hap"], ["py]你好。"]])
        printed: list[str] = []
        runner = VoiceChatRunner(
            session=session,
            tts_client=FakeTtsClient(),
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
            expression_tags=("happy",),
        )

        runner.run_turn("first", cancel_event=threading.Event())
        runner.run_turn("second", cancel_event=threading.Event())

        self.assertEqual(printed, ["[hap", "py]你好。"])

    def test_spoken_text_filter_state_does_not_leak_after_cancelled_turn(self) -> None:
        session = MultiTurnSession([["（未闭合"], ["下一轮。"]])
        tts = FakeTtsClient()
        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=lambda chunk: None,
        )

        runner.run_turn("first", cancel_event=threading.Event())
        runner.run_turn("second", cancel_event=threading.Event())

        self.assertEqual(tts.requests, ["下一轮。"])

    def test_streams_text_and_synthesizes_sentences_in_order(self) -> None:
        session = FakeSession(["你好，", "我是咕嘎。", "今天继续。"])
        tts = FakeTtsClient()
        player = FakeAudioPlayer()
        printed: list[str] = []

        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=player,
            text_sink=printed.append,
        )

        summary = runner.run_turn("hi")

        self.assertEqual(printed, ["你好，", "我是咕嘎。", "今天继续。"])
        self.assertEqual(tts.requests, ["你好，我是咕嘎。", "今天继续。"])
        self.assertEqual(player.items, [b"audio:\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe6\x88\x91\xe6\x98\xaf\xe5\x92\x95\xe5\x98\x8e\xe3\x80\x82", b"audio:\xe4\xbb\x8a\xe5\xa4\xa9\xe7\xbb\xa7\xe7\xbb\xad\xe3\x80\x82"])
        self.assertEqual(player.cleanup_calls, ["join", "stop:False"])
        self.assertEqual(summary.sentences, 2)

    def test_reports_tts_errors_after_worker_cleanup(self) -> None:
        class FailingTts(FakeTtsClient):
            def synthesize(self, text: str) -> AudioData:
                _ = text
                raise ValueError("synthesis failed")

        player = FakeAudioPlayer()
        runner = VoiceChatRunner(
            session=FakeSession(["失败。"]),
            tts_client=FailingTts(),
            audio_player=player,
            text_sink=lambda chunk: None,
        )

        with self.assertRaisesRegex(RuntimeError, "TTS worker failed") as caught:
            runner.run_turn("hi")

        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(player.cleanup_calls, ["stop:True"])

    def test_text_sink_is_not_blocked_by_slow_tts(self) -> None:
        class SlowTts(FakeTtsClient):
            def synthesize(self, text: str) -> AudioData:
                time.sleep(0.2)
                return super().synthesize(text)

        session = FakeSession(["第一句。", "第二句。"])
        tts = SlowTts()
        player = FakeAudioPlayer()
        printed: list[tuple[str, float]] = []

        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=player,
            text_sink=lambda chunk: printed.append((chunk, time.perf_counter())),
        )

        started = time.perf_counter()
        summary = runner.run_turn("hi")

        self.assertEqual([chunk for chunk, _ in printed], ["第一句。", "第二句。"])
        self.assertLess(printed[-1][1] - started, 0.15)
        self.assertEqual(summary.sentences, 2)

    def test_cancel_does_not_flush_partial_sentence_to_tts(self) -> None:
        cancel_event = threading.Event()
        tts = FakeTtsClient()
        printed: list[str] = []

        runner = VoiceChatRunner(
            session=CancelAfterOneChunkSession(),
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
        )

        summary = runner.run_turn("hi", cancel_event=cancel_event)

        self.assertEqual(printed, ["未完成"])
        self.assertEqual(tts.requests, [])
        self.assertEqual(summary.sentences, 0)

    def test_excludes_parenthesized_actions_from_tts_but_keeps_display_text(self) -> None:
        session = FakeSession(["咕嘎！（眼睛放光，蹦跶了两下）", "汉堡。", "(挥手)继续。"])
        tts = FakeTtsClient()
        printed: list[str] = []

        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
        )

        summary = runner.run_turn("hi")

        self.assertEqual(printed, ["咕嘎！（眼睛放光，蹦跶了两下）", "汉堡。", "(挥手)继续。"])
        self.assertEqual(tts.requests, ["咕嘎！", "汉堡。", "继续。"])
        self.assertEqual(summary.sentences, 3)

    def test_excludes_parenthesized_actions_across_chunks_from_tts(self) -> None:
        session = FakeSession(["咕嘎（眼睛", "放光）好吃。"])
        tts = FakeTtsClient()
        printed: list[str] = []

        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=printed.append,
        )

        runner.run_turn("hi")

        self.assertEqual(printed, ["咕嘎（眼睛", "放光）好吃。"])
        self.assertEqual(tts.requests, ["咕嘎好吃。"])

    def test_debug_logs_voice_playback_split_points(self) -> None:
        logs: list[str] = []
        session = DebugFakeSession(
            ["咕咕嘎嘎！是你是你！我刚刚看到你来就好开心呀！（摇摇摆摆跑过来）要陪我玩吗？"],
            logs,
        )
        tts = FakeTtsClient()

        runner = VoiceChatRunner(
            session=session,
            tts_client=tts,
            audio_player=FakeAudioPlayer(),
            text_sink=lambda chunk: None,
            sentence_buffer=TextSentenceBuffer(max_chars=16),
        )

        runner.run_turn("hi")

        playback_logs = [log for log in logs if "voice_playback_start" in log]
        self.assertEqual(len(playback_logs), 4)
        self.assertIn("[DEBUG][VoiceChatRunner][debug_session]", playback_logs[0])
        self.assertIn("sequence_id=1", playback_logs[0])
        self.assertIn("token_count=5", playback_logs[0])
        self.assertIn("split_reason=boundary:！", playback_logs[0])
        self.assertIn('text="咕咕嘎嘎！"', playback_logs[0])
        self.assertIn('text="要陪我玩吗？"', playback_logs[-1])


if __name__ == "__main__":
    unittest.main()
