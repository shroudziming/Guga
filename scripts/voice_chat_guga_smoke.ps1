param(
    [string]$Endpoint = "http://127.0.0.1:9880/tts",
    [string]$RefAudioPath = "D:\work\LLM\voice_dataset\guga_voice\wav_mono_24k\justme.wav",
    [string]$PromptText = "",
    [int]$SentenceMaxChars = 16,
    [switch]$NoAudio,
    [switch]$NoPrewarm,
    [switch]$WithTools
)

$ErrorActionPreference = "Stop"

function ConvertFrom-Utf8Base64([string]$Text) {
    return [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Text))
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $RefAudioPath)) {
    throw "Reference audio not found: $RefAudioPath"
}

if (-not $PromptText) {
    $PromptText = ConvertFrom-Utf8Base64 "5a+55ZGA77yM6ams5LiK5p2l77yM6ams5LiK5Yiw77yM5L2G5piv6YO95rKh5pyJ5Yiw77yM5omA5Lul5bCx5Y+q5pyJ5oiR5LiA5Liq5Lq65ZWm77yB"
}

$env:GUGA_TTS_ENDPOINT = $Endpoint
$env:GUGA_TTS_REF_AUDIO_PATH = $RefAudioPath
$env:GUGA_TTS_PROMPT_TEXT = $PromptText
$env:GUGA_TTS_PLAY_AUDIO = if ($NoAudio) { "0" } else { "1" }
$env:GUGA_TTS_SENTENCE_MAX_CHARS = "$SentenceMaxChars"
$env:GUGA_TTS_PREWARM = if ($NoPrewarm) { "0" } else { "1" }
$env:GUGA_VOICE_WITH_TOOLS = if ($WithTools) { "1" } else { "0" }
if ($WithTools) {
    $env:Guga_MAX_TOOL_ROUNDS = "3"
} else {
    $env:Guga_MAX_TOOL_ROUNDS = "0"
}

Write-Host "[Guga Voice] endpoint=$env:GUGA_TTS_ENDPOINT"
Write-Host "[Guga Voice] ref_audio=$env:GUGA_TTS_REF_AUDIO_PATH"
Write-Host "[Guga Voice] play_audio=$env:GUGA_TTS_PLAY_AUDIO"
Write-Host "[Guga Voice] sentence_max_chars=$env:GUGA_TTS_SENTENCE_MAX_CHARS"
Write-Host "[Guga Voice] prewarm=$env:GUGA_TTS_PREWARM"
Write-Host "[Guga Voice] tools=$env:GUGA_VOICE_WITH_TOOLS"
Write-Host ""

python src\voice_cli_chat.py
