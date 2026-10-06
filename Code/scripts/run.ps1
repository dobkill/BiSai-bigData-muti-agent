param(
    [string]$Date = "",
    [switch]$All,
    [switch]$UseLLM,
    [switch]$Offline,
    [ValidateSet("ollama", "llama_cpp", "api")][string]$Provider,
    [string]$Model = "",
    [string]$BaseUrl = "",
    [ValidateRange(1,32)][int]$Concurrency = 1,
    [switch]$AdaptiveConcurrency,
    [switch]$FixedConcurrency,
    [switch]$StrictLLM,
    [int]$MaxLLMCalls,
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
if ($AdaptiveConcurrency -and $FixedConcurrency) { throw "AdaptiveConcurrency 和 FixedConcurrency 不能同时指定" }
if ($Date) { $arguments += @("--date", $Date) }
elseif ($All) { $arguments += "--all" }
else { $arguments += "--nodes" }
if ($UseLLM) { $arguments += "--use-llm" }
if ($Offline) { $arguments += "--offline" }
if ($Provider) { $arguments += @("--provider", $Provider) }
if ($Model) { $arguments += @("--model", $Model) }
if ($BaseUrl) { $arguments += @("--base-url", $BaseUrl) }
if ($PSBoundParameters.ContainsKey("Concurrency")) { $arguments += @("--concurrency", "$Concurrency") }
if ($AdaptiveConcurrency) { $arguments += "--adaptive-concurrency" }
if ($FixedConcurrency) { $arguments += "--fixed-concurrency" }
if ($StrictLLM) { $arguments += "--strict-llm" }
if ($PSBoundParameters.ContainsKey("MaxLLMCalls")) {
    if ($MaxLLMCalls -lt 1) { throw "MaxLLMCalls 必须大于零" }
    $arguments += @("--max-llm-calls", "$MaxLLMCalls")
}
& $CondaPath run --no-capture-output -n multimodal-agent python @arguments
if ($LASTEXITCODE -ne 0) { throw "Agent 执行失败，退出码 $LASTEXITCODE" }

