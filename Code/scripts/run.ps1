param(
    [string]$Date = "",
    [switch]$All,
    [switch]$UseLLM,
    [switch]$Offline,
    [string]$Config = "",
    [string]$CondaPath = "D:\APP\miniconda\Scripts\conda.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $CondaPath)) { throw "未找到 conda：$CondaPath" }
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not $Config) { $Config = Join-Path $projectRoot "config.toml" }
$arguments = @("-m", "fusion_agent", "--config", $Config, "run")
if ($Date -and $All) { throw "Date 和 All 不能同时指定" }
if ($UseLLM -and $Offline) { throw "UseLLM 和 Offline 不能同时指定" }
if ($Date) { $arguments += @("--date", $Date) }
elseif ($All) { $arguments += "--all" }
else { $arguments += "--nodes" }
if ($UseLLM) { $arguments += "--use-llm" }
if ($Offline) { $arguments += "--offline" }
& $CondaPath run --no-capture-output -n multimodal-agent python @arguments
if ($LASTEXITCODE -ne 0) { throw "Agent 执行失败，退出码 $LASTEXITCODE" }

