from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from guga.voice.audio_player import AudioData


class TtsClient(Protocol):
    def synthesize(self, text: str) -> AudioData:
        ...


PostJson = Callable[[str, dict[str, Any], float], bytes]


@dataclass(frozen=True)
class GptSoVitsConfig:
    endpoint: str = "http://127.0.0.1:9880/tts"
    ref_audio_path: str = ""
    prompt_text: str = ""
    text_lang: str = "zh"
    prompt_lang: str = "zh"
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    parallel_infer: bool = True
    streaming_mode: bool = False
    media_type: str = "wav"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "GptSoVitsConfig":
        return cls(
            endpoint=os.environ.get("GUGA_TTS_ENDPOINT", cls.endpoint),
            ref_audio_path=os.environ.get("GUGA_TTS_REF_AUDIO_PATH", ""),
            prompt_text=os.environ.get("GUGA_TTS_PROMPT_TEXT", ""),
            text_lang=os.environ.get("GUGA_TTS_TEXT_LANG", "zh"),
            prompt_lang=os.environ.get("GUGA_TTS_PROMPT_LANG", "zh"),
            text_split_method=os.environ.get("GUGA_TTS_TEXT_SPLIT_METHOD", "cut5"),
            batch_size=_env_int("GUGA_TTS_BATCH_SIZE", 1),
            batch_threshold=_env_float("GUGA_TTS_BATCH_THRESHOLD", 0.75),
            split_bucket=_env_bool("GUGA_TTS_SPLIT_BUCKET", True),
            parallel_infer=_env_bool("GUGA_TTS_PARALLEL_INFER", True),
            streaming_mode=_env_bool("GUGA_TTS_STREAMING_MODE", False),
            media_type=os.environ.get("GUGA_TTS_MEDIA_TYPE", "wav"),
            timeout_seconds=_env_float("GUGA_TTS_TIMEOUT_SECONDS", 120.0),
        )


class GptSoVitsHttpClient:
    def __init__(
        self,
        config: GptSoVitsConfig,
        *,
        post_json: PostJson | None = None,
    ) -> None:
        if not config.ref_audio_path:
            raise ValueError("GUGA_TTS_REF_AUDIO_PATH is required for GPT-SoVITS voice chat")
        self.config = config
        self._post_json = post_json or _post_json

    def synthesize(self, text: str) -> AudioData:
        payload = self._payload(text)
        data = self._post_json(self.config.endpoint, payload, self.config.timeout_seconds)
        if self.config.media_type.lower() == "wav":
            return AudioData.from_wav_bytes(data)
        return AudioData(
            data=data,
            sample_rate=0,
            channels=0,
            sample_width=0,
            duration_seconds=0.0,
            media_type=self.config.media_type,
        )

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "text_lang": self.config.text_lang,
            "ref_audio_path": self.config.ref_audio_path,
            "prompt_text": self.config.prompt_text,
            "prompt_lang": self.config.prompt_lang,
            "text_split_method": self.config.text_split_method,
            "batch_size": self.config.batch_size,
            "batch_threshold": self.config.batch_threshold,
            "split_bucket": self.config.split_bucket,
            "parallel_infer": self.config.parallel_infer,
            "streaming_mode": self.config.streaming_mode,
            "media_type": self.config.media_type,
        }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPT-SoVITS request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GPT-SoVITS endpoint is unavailable: {exc}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
