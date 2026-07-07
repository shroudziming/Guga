param(
    [string]$Endpoint = "http://127.0.0.1:9880/tts",
    [string]$GptWeightPath = "D:\work\LLM\external\GPT-SoVITS\GPT_weights_v2\guga_voice_e8-e8.ckpt",
    [string]$SoVitsWeightPath = "D:\work\LLM\external\GPT-SoVITS\SoVITS_weights_v2\guga_voice_e8_e8_s800.pth",
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

function Get-GptSoVitsBaseUrl([string]$EndpointValue) {
    $normalized = $EndpointValue.TrimEnd("/")
    if ($normalized.EndsWith("/tts")) {
        return $normalized.Substring(0, $normalized.Length - 4)
    }
    return $normalized
}

function Invoke-GptSoVitsWeightSwitch([string]$BaseUrl, [string]$Route, [string]$WeightPath) {
    $encodedPath = [uri]::EscapeDataString($WeightPath)
    $url = "$BaseUrl/$Route`?weights_path=$encodedPath"
    $response = Invoke-WebRequest -Uri $url -Method GET -UseBasicParsing -TimeoutSec $TimeoutSeconds
    if ($response.StatusCode -ne 200) {
        throw "$Route failed: HTTP $($response.StatusCode) $($response.Content)"
    }
    Write-Host "[GPT-SoVITS] $Route ok: $WeightPath"
}

if (-not (Test-Path -LiteralPath $GptWeightPath)) {
    throw "GPT weight not found: $GptWeightPath"
}
if (-not (Test-Path -LiteralPath $SoVitsWeightPath)) {
    throw "SoVITS weight not found: $SoVitsWeightPath"
}

$baseUrl = Get-GptSoVitsBaseUrl $Endpoint
Invoke-GptSoVitsWeightSwitch $baseUrl "set_gpt_weights" $GptWeightPath
Invoke-GptSoVitsWeightSwitch $baseUrl "set_sovits_weights" $SoVitsWeightPath
