from __future__ import annotations

import io
import queue
import tempfile
import threading
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioData:
    data: bytes
    sample_rate: int
    channels: int
    sample_width: int
    duration_seconds: float
    media_type: str = "wav"

    @classmethod
    def from_wav_bytes(cls, data: bytes) -> "AudioData":
        with wave.open(_BytesReader(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
        duration = frames / sample_rate if sample_rate else 0.0
        return cls(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            duration_seconds=duration,
            media_type="wav",
        )

    @classmethod
    def from_pcm16_mono(cls, data: bytes, *, sample_rate: int) -> "AudioData":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(data)
        return cls.from_wav_bytes(buffer.getvalue())


class NullAudioPlayer:
    def start(self) -> None:
        return

    def enqueue(self, audio: AudioData) -> None:
        _ = audio

    def join(self, timeout: float | None = None) -> None:
        _ = timeout

    def stop(self, clear: bool = False) -> None:
        _ = clear


class WavAudioPlayer:
    """Background WAV player using Windows winsound."""

    def __init__(self) -> None:
        self._queue: queue.Queue[AudioData | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, name="guga-wav-player", daemon=True)
        self._thread.start()

    def enqueue(self, audio: AudioData) -> None:
        self._queue.put(audio)

    def join(self, timeout: float | None = None) -> None:
        _ = timeout
        self._queue.join()

    def stop(self, clear: bool = False) -> None:
        if clear:
            self._clear_queue()
        self._stop_requested.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            audio = self._queue.get()
            try:
                if audio is None:
                    return
                if self._stop_requested.is_set():
                    continue
                self._play(audio)
            finally:
                self._queue.task_done()

    def _play(self, audio: AudioData) -> None:
        if audio.media_type.lower() != "wav":
            raise ValueError("WavAudioPlayer only supports wav audio")

        import winsound

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio.data)
                temp_path = Path(temp_file.name)
            winsound.PlaySound(str(temp_path), winsound.SND_FILENAME)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _clear_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                _ = item
                self._queue.task_done()


def audio_player_from_env(env: Mapping[str, str]) -> NullAudioPlayer | WavAudioPlayer:
    raw = env.get("GUGA_TTS_PLAY_AUDIO", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return NullAudioPlayer()
    return WavAudioPlayer()


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._pos
        start = self._pos
        end = min(len(self._data), start + size)
        self._pos = end
        return self._data[start:end].tobytes()

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self._pos + offset
        elif whence == 2:
            new_pos = len(self._data) + offset
        else:
            raise ValueError("invalid whence")
        self._pos = max(0, min(len(self._data), new_pos))
        return self._pos

    def tell(self) -> int:
        return self._pos
