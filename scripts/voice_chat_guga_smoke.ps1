param(
    [string]$Endpoint = "http://127.0.0.1:9880/tts",
    [string]$RefAudioPath = "D:\work\LLM\voice_dataset\guga_voice\wav_mono_24k\justme.wav",
    [string]$PromptText = "对呀，马上来，马上到，但是都没有到，所以就只有我一个人啦！",
    [switch]$NoAudio
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $RefAudioPath)) {
    throw "Reference audio not found: $RefAudioPath"
}

$env:GUGA_TTS_ENDPOINT = $Endpoint
$env:GUGA_TTS_REF_AUDIO_PATH = $RefAudioPath
$env:GUGA_TTS_PROMPT_TEXT = $PromptText
$env:GUGA_TTS_PLAY_AUDIO = if ($NoAudio) { "0" } else { "1" }

Write-Host "[Guga Voice] endpoint=$env:GUGA_TTS_ENDPOINT"
Write-Host "[Guga Voice] ref_audio=$env:GUGA_TTS_REF_AUDIO_PATH"
Write-Host "[Guga Voice] play_audio=$env:GUGA_TTS_PLAY_AUDIO"
Write-Host ""

python src\voice_cli_chat.py
