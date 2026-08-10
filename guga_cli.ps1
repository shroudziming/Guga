[CmdletBinding()]
param(
    # Parse configuration and print non-sensitive values without starting a model.
    [switch]$ValidateOnly,
    [string]$EnvFile = "",
    [string]$ConfigFile = "",
    [string]$DesktopPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
if (-not $EnvFile) {
    $EnvFile = Join-Path $repoRoot ".env"
}
if (-not $ConfigFile) {
    $ConfigFile = Join-Path $repoRoot "config\guga_cli.env"
}

function Import-GugaEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [switch]$Required,
        [string[]]$AllowedKeys,
        [switch]$PreserveExisting
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        if ($Required) {
            throw "Configuration file does not exist: $Path"
        }
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $raw = $line.Trim()
        if (-not $raw -or $raw.StartsWith("#")) {
            continue
        }
        if (-not $raw.Contains("=")) {
            throw "Invalid configuration line in ${Path}: $raw"
        }
        $parts = $raw.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw "Invalid configuration key in ${Path}: $key"
        }
        if ($AllowedKeys -and $key -notin $AllowedKeys) {
            throw "Unsupported startup configuration key: $key"
        }
        $existing = [Environment]::GetEnvironmentVariable($key, "Process")
        if ($PreserveExisting -and $null -ne $existing) {
            continue
        }
        Set-Item -LiteralPath "Env:$key" -Value $value
    }
}

# Private API values come from .env; existing process values take precedence.
Import-GugaEnvFile -Path $EnvFile -PreserveExisting

# Tracked startup config may set only this non-sensitive allowlist.
$startupKeys = @(
    "Guga_CLI_MODEL_ROUTE"
    "Guga_CLI_API_MODEL_ID"
    "Guga_CLI_LOCAL_MODEL_ID"
    "Guga_CLI_LOCAL_CACHE_DIR"
    "Guga_CLI_DEFAULT_WORKSPACE"
    "Guga_CLI_ALLOW_CREATE_WORKSPACE"
    "Guga_ENABLE_WRITE_TOOL"
    "Guga_ENABLE_COMMAND_TOOL"
    "Guga_DEBUG"
)
Import-GugaEnvFile -Path $ConfigFile -Required -AllowedKeys $startupKeys

$booleanKeys = @(
    "Guga_CLI_ALLOW_CREATE_WORKSPACE"
    "Guga_ENABLE_WRITE_TOOL"
    "Guga_ENABLE_COMMAND_TOOL"
    "Guga_DEBUG"
)
foreach ($booleanKey in $booleanKeys) {
    $booleanValue = [Environment]::GetEnvironmentVariable($booleanKey, "Process")
    if ($booleanValue -notin @("0", "1")) {
        throw "$booleanKey must be 1 (enabled/allowed) or 0 (disabled)"
    }
}

$route = $env:Guga_CLI_MODEL_ROUTE.Trim().ToLowerInvariant()
switch ($route) {
    "api" {
        $env:Guga_MODEL_PROVIDER = "api"
        $env:Guga_MODEL_ID = $env:Guga_CLI_API_MODEL_ID
        if (-not ($env:Guga_API_KEY -or $env:OPENAI_API_KEY)) {
            throw "API route requires Guga_API_KEY or OPENAI_API_KEY in .env"
        }
        if (-not ($env:Guga_API_BASE_URL -or $env:OPENAI_BASE_URL)) {
            throw "API route requires Guga_API_BASE_URL or OPENAI_BASE_URL in .env"
        }
    }
    "local" {
        $env:Guga_MODEL_PROVIDER = "local"
        $env:Guga_MODEL_ID = $env:Guga_CLI_LOCAL_MODEL_ID
        $cache = [IO.Path]::GetFullPath(
            (Join-Path $repoRoot $env:Guga_CLI_LOCAL_CACHE_DIR)
        )
        $env:Guga_CACHE_DIR = $cache
    }
    default {
        throw "Unsupported Guga_CLI_MODEL_ROUTE: $route"
    }
}

if ($env:Guga_CLI_DEFAULT_WORKSPACE.Trim().ToLowerInvariant() -ne "desktop") {
    throw "Guga_CLI_DEFAULT_WORKSPACE currently supports only: desktop"
}
if (-not $DesktopPath) {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
}
if (-not $DesktopPath) {
    throw "Unable to resolve the Windows desktop directory"
}
$workspacePath = Join-Path ([IO.Path]::GetFullPath($DesktopPath)) "Guga"
New-Item -ItemType Directory -Path $workspacePath -Force | Out-Null
$env:Guga_CLI_DEFAULT_WORKSPACE_PATH = (Resolve-Path -LiteralPath $workspacePath).Path

if ($ValidateOnly) {
    [ordered]@{
        model_provider = $env:Guga_MODEL_PROVIDER
        model_id = $env:Guga_MODEL_ID
        cache_dir = $env:Guga_CACHE_DIR
        workspace = $env:Guga_CLI_DEFAULT_WORKSPACE_PATH
        allow_create_workspace = $env:Guga_CLI_ALLOW_CREATE_WORKSPACE
        write_tool = $env:Guga_ENABLE_WRITE_TOOL
        command_tool = $env:Guga_ENABLE_COMMAND_TOOL
        debug = $env:Guga_DEBUG
    } | ConvertTo-Json -Compress
    return
}

# Workspace changes live only in the child Python process and are never saved here.
Push-Location $repoRoot
try {
    & python -u "src\basic_cli_chat.py"
    $cliExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $cliExitCode
