from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Iterator

from guga.voice.audio_player import AudioData
from guga.voice.metrics import VoiceMetrics
from guga.voice.runner import VoiceChatRunner
from guga.voice.sentence_buffer import TextSentenceBuffer
from guga.voice.tts_client import GptSoVitsConfig, GptSoVitsHttpClient


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

    def start(self) -> None:
        self.started = True

    def enqueue(self, audio: AudioData) -> None:
        self.items.append(audio.data)

    def stop(self, clear: bool = False) -> None:
        _ = clear
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        _ = timeout


class GptSoVitsHttpClientTest(unittest.TestCase):
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


class VoiceChatRunnerTest(unittest.TestCase):
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
        self.assertEqual(summary.sentences, 2)

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


if __name__ == "__main__":
    unittest.main()
