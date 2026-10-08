param(
    [ValidateRange(1, 32)][int]$Parallel = 4,
    [ValidateRange(1024, 65535)][int]$Port = 11435,
    [string]$LogDirectory = (Join-Path $PSScriptRoot '..\Datas\ollama-parallel')
)
$ErrorActionPreference = 'Stop'
$ollamaExecutable = (Get-Command ollama -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$resolvedLogs = (Resolve-Path -LiteralPath $LogDirectory).Path
$pidFile = Join-Path $resolvedLogs "server-$Port.pid"
if (Test-Path -LiteralPath $pidFile) {
    $previousPid = [int](Get-Content -LiteralPath $pidFile)
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$previousPid"
    if ($previousProcess) {
        if ($previousProcess.ExecutablePath -ne $ollamaExecutable -or $previousProcess.CommandLine -notmatch '\bserve\b') {
            throw 'Saved PID belongs to another process; refusing to stop it.'
        }
        Stop-Process -Id $previousPid -ErrorAction Stop
        Start-Sleep -Milliseconds 1000
    }
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port $Port already has a listener; choose another port." }
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_NUM_PARALLEL = "$Parallel"
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '12288'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'f16'
$env:OLLAMA_KEEP_ALIVE = '30m'
$env:OLLAMA_MAX_QUEUE = '256'
$env:OLLAMA_GPU_OVERHEAD = '536870912'
if (-not $env:OLLAMA_MODELS) {
    $configuredModels = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
    if ($configuredModels) { $env:OLLAMA_MODELS = $configuredModels }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $resolvedLogs "server-$Parallel-$stamp.stdout.log"
$stderrLog = Join-Path $resolvedLogs "server-$Parallel-$stamp.stderr.log"
$serverProcess = Start-Process -FilePath $ollamaExecutable -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$serverProcess.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
[pscustomobject]@{Pid=$serverProcess.Id;Url="http://127.0.0.1:$Port";Parallel=$Parallel;Context=12288;Cache='f16';Stderr=$stderrLog} | ConvertTo-Json
