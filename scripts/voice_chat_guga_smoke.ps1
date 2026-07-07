param(
    [string]$Endpoint = "http://127.0.0.1:9880/tts",
    [string]$RefAudioPath = "D:\work\LLM\voice_dataset\guga_voice\wav_mono_24k\justme.wav",
    [string]$PromptText = "",
    [string]$GptWeightPath = "D:\work\LLM\external\GPT-SoVITS\GPT_weights_v2\guga_voice_e8-e8.ckpt",
    [string]$SoVitsWeightPath = "D:\work\LLM\external\GPT-SoVITS\SoVITS_weights_v2\guga_voice_e8_e8_s800.pth",
    [string]$MediaType = "raw",
    [string]$StreamingMode = "1",
    [int]$SentenceMaxChars = 16,
    [switch]$NoAudio,
    [switch]$NoPrewarm,
    [switch]$WithTools,
    [switch]$SkipWeightSwitch
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

if (-not $SkipWeightSwitch) {
    & (Join-Path $PSScriptRoot "set_gpt_sovits_voice_model.ps1") `
        -Endpoint $Endpoint `
        -GptWeightPath $GptWeightPath `
        -SoVitsWeightPath $SoVitsWeightPath
}

$env:GUGA_TTS_ENDPOINT = $Endpoint
$env:GUGA_TTS_REF_AUDIO_PATH = $RefAudioPath
$env:GUGA_TTS_PROMPT_TEXT = $PromptText
$env:GUGA_TTS_PLAY_AUDIO = if ($NoAudio) { "0" } else { "1" }
$env:GUGA_TTS_SENTENCE_MAX_CHARS = "$SentenceMaxChars"
$env:GUGA_TTS_PREWARM = if ($NoPrewarm) { "0" } else { "1" }
$env:GUGA_TTS_MEDIA_TYPE = $MediaType
$env:GUGA_TTS_STREAMING_MODE = $StreamingMode
$env:GUGA_VOICE_WITH_TOOLS = if ($WithTools) { "1" } else { "0" }
if ($WithTools) {
    $env:Guga_MAX_TOOL_ROUNDS = "3"
} else {
    $env:Guga_MAX_TOOL_ROUNDS = "0"
}

Write-Host "[Guga Voice] endpoint=$env:GUGA_TTS_ENDPOINT"
Write-Host "[Guga Voice] ref_audio=$env:GUGA_TTS_REF_AUDIO_PATH"
Write-Host "[Guga Voice] gpt_weight=$GptWeightPath"
Write-Host "[Guga Voice] sovits_weight=$SoVitsWeightPath"
Write-Host "[Guga Voice] weight_switch=$(-not $SkipWeightSwitch)"
Write-Host "[Guga Voice] play_audio=$env:GUGA_TTS_PLAY_AUDIO"
Write-Host "[Guga Voice] sentence_max_chars=$env:GUGA_TTS_SENTENCE_MAX_CHARS"
Write-Host "[Guga Voice] prewarm=$env:GUGA_TTS_PREWARM"
Write-Host "[Guga Voice] media_type=$env:GUGA_TTS_MEDIA_TYPE"
Write-Host "[Guga Voice] streaming_mode=$env:GUGA_TTS_STREAMING_MODE"
Write-Host "[Guga Voice] tools=$env:GUGA_VOICE_WITH_TOOLS"
if ($WithTools) {
    Write-Host "[Guga Voice] tools note: speech starts before tools only when the model streams content before tool_calls"
}
Write-Host ""

python src\voice_cli_chat.py
