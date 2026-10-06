@echo off
setlocal
if not defined FUSION_CONDA_EXE set "FUSION_CONDA_EXE=D:\APP\miniconda\Scripts\conda.exe"
if not exist "%FUSION_CONDA_EXE%" (
    echo Conda not found. Set FUSION_CONDA_EXE to the full path of conda.exe.
    exit /b 1
)
if not defined FUSION_CONFIG set "FUSION_CONFIG=%~dp0..\config.toml"
call "%FUSION_CONDA_EXE%" run --no-capture-output -n multimodal-agent python -m fusion_agent --config "%FUSION_CONFIG%" export-review %*
exit /b %errorlevel%
