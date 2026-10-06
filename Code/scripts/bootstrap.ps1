param([string]$CondaPath = "", [string]$EnvironmentName = "multimodal-agent")
$ErrorActionPreference = "Stop"
if (-not $CondaPath) {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) { $CondaPath = $command.Source }
    elseif (Test-Path -LiteralPath "D:\APP\miniconda\Scripts\conda.exe") {
        $CondaPath = "D:\APP\miniconda\Scripts\conda.exe"
    }
    else { throw "未找到 conda，请使用 -CondaPath 指定 conda.exe 的路径。" }
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Push-Location -LiteralPath $projectRoot
try {
    $environmentInfo = & $CondaPath env list --json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "conda env list 失败" }
    $exists = @($environmentInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvironmentName }).Count -gt 0
    if (-not $exists) {
        & $CondaPath create -n $EnvironmentName --override-channels -c conda-forge python=3.11 pip -y
        if ($LASTEXITCODE -ne 0) { throw "Conda 环境创建失败" }
    }
    & $CondaPath run --no-capture-output -n $EnvironmentName python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Python 依赖安装失败" }
    & $CondaPath run --no-capture-output -n $EnvironmentName python -m fusion_agent doctor
    if ($LASTEXITCODE -ne 0) { throw "环境检查失败" }
}
finally { Pop-Location }

