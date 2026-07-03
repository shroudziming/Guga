"""Voice adapter layer for streaming Guga replies into local TTS."""

from guga.voice.audio_player import AudioData, NullAudioPlayer, WavAudioPlayer, audio_player_from_env
from guga.voice.metrics import VoiceMetrics, VoiceMetricsSummary
from guga.voice.runner import VoiceChatRunner
from guga.voice.sentence_buffer import TextSentenceBuffer, sentence_buffer_from_env
from guga.voice.tts_client import GptSoVitsConfig, GptSoVitsHttpClient
from guga.voice.tool_mode import configure_voice_tool_mode

__all__ = [
    "AudioData",
    "GptSoVitsConfig",
    "GptSoVitsHttpClient",
    "NullAudioPlayer",
    "TextSentenceBuffer",
    "VoiceChatRunner",
    "VoiceMetrics",
    "VoiceMetricsSummary",
    "WavAudioPlayer",
    "audio_player_from_env",
    "configure_voice_tool_mode",
    "sentence_buffer_from_env",
]
