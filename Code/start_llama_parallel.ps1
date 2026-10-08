param(
    [ValidateRange(1, 32)][int]$Parallel = 8,
    [ValidateRange(1024, 65535)][int]$Port = 11436,
    [ValidateRange(4096, 131072)][int]$Context = 12288,
    [string]$Model = 'qwen3.5:9b-q8_0',
    [string]$LogDirectory = (Join-Path $PSScriptRoot '..\Datas\ollama-parallel')
)
$ErrorActionPreference = 'Stop'
$ollamaExecutable = (Get-Command ollama -ErrorAction Stop).Source
$serverExecutable = Join-Path (Split-Path $ollamaExecutable) 'lib\ollama\llama-server.exe'
if (-not (Test-Path -LiteralPath $serverExecutable)) { throw 'Ollama bundled llama-server was not found.' }
$cudaDirectory = Join-Path (Split-Path $serverExecutable) 'cuda_v12'
if (-not (Test-Path -LiteralPath (Join-Path $cudaDirectory 'ggml-cuda.dll'))) { throw 'Ollama CUDA 12 backend was not found.' }
$env:GGML_BACKEND_PATH = Join-Path $cudaDirectory 'ggml-cuda.dll'
$env:PATH = (Split-Path $serverExecutable)+';'+$cudaDirectory+';'+$env:PATH
$modelsRoot = $env:OLLAMA_MODELS
if (-not $modelsRoot) { $modelsRoot = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User') }
if (-not $modelsRoot) { $modelsRoot = Join-Path $env:USERPROFILE '.ollama\models' }
if ($Model -notmatch '^([a-zA-Z0-9._-]+):([a-zA-Z0-9._-]+)$') { throw 'Use a local library model name:tag.' }
$manifestPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\$($Matches[1])\$($Matches[2])"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$layer = @($manifest.layers | Where-Object mediaType -eq 'application/vnd.ollama.image.model')
if ($layer.Count -ne 1) { throw 'Expected exactly one model GGUF layer.' }
$modelFile = Join-Path $modelsRoot ('blobs\'+$layer[0].digest.Replace(':','-'))
if (-not (Test-Path -LiteralPath $modelFile)) { throw 'Model GGUF was not found.' }
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$resolvedLogs = (Resolve-Path -LiteralPath $LogDirectory).Path
$pidFile = Join-Path $resolvedLogs "llama-$Port.pid"
if (Test-Path -LiteralPath $pidFile) {
    $previousPid = [int](Get-Content -LiteralPath $pidFile)
    $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$previousPid"
    if ($previousProcess) {
        if ($previousProcess.ExecutablePath -ne $serverExecutable -or $previousProcess.CommandLine -notmatch "--port $Port\b") {
            throw 'Saved PID belongs to another process; refusing to stop it.'
        }
        Stop-Process -Id $previousPid -ErrorAction Stop
        Start-Sleep -Milliseconds 1000
    }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already used." }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stderrLog = Join-Path $resolvedLogs "llama-$Parallel-$stamp.stderr.log"
$stdoutLog = Join-Path $resolvedLogs "llama-$Parallel-$stamp.stdout.log"
$arguments = @('--model', ('"'+$modelFile+'"'), '--port', "$Port", '--host', '127.0.0.1', '--no-webui', '--offline',
    '--ctx-size', "$($Context*$Parallel)", '--parallel', "$Parallel", '--gpu-layers', '99', '--fit', 'off',
    '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--batch-size', '1024', '--ubatch-size', '1024',
    '--split-mode', 'none', '--main-gpu', '0', '--jinja', '--no-context-shift', '--cache-ram', '1024',
    '--log-verbosity', '4', '--no-log-prefix', '--no-log-timestamps')
$serverProcess = Start-Process -FilePath $serverExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$serverProcess.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
$config = [pscustomobject]@{Pid=$serverProcess.Id;Url="http://127.0.0.1:$Port";Parallel=$Parallel;ContextPerSlot=$Context;Cache='f16';Model=$Model;ModelDigest=$layer[0].digest;ModelFile=$modelFile;Executable=$serverExecutable;Arguments=$arguments;Stderr=$stderrLog;StartedAt=(Get-Date).ToString('o')}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedLogs "llama-$Port.json") -Encoding utf8
$config | ConvertTo-Json -Depth 4
