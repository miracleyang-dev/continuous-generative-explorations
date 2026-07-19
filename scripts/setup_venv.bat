@echo off
REM ============================================================
REM  continuous-generative-explorations - one-shot venv setup
REM  for Windows CMD
REM ============================================================
REM  Usage (from repo root):
REM      scripts\setup_venv.bat
REM
REM  What it does:
REM    1. Creates .venv/ under the repo root if it doesn't exist.
REM    2. Upgrades pip inside the venv (avoids the GBK-decode bug on
REM       older pip versions running under Chinese Windows locales).
REM    3. Installs runtime + dev deps from requirements-dev.txt.
REM
REM  After running this, activate the venv in every new terminal with:
REM      .venv\Scripts\activate
REM  and deactivate with:
REM      deactivate
REM
REM  NOTE on PyTorch / CUDA:
REM    requirements.txt pins `torch` without a CUDA variant, so pip
REM    resolves whatever wheel matches your Python + OS (CPU-only on
REM    most Windows boxes). If you need CUDA, run this script first,
REM    then reinstall torch from the official index, e.g.:
REM      .venv\Scripts\python.exe -m pip install --upgrade \
REM        --index-url https://download.pytorch.org/whl/cu121 torch
REM ============================================================

setlocal

REM cd to repo root (parent of this script)
cd /d "%~dp0.."

if not exist ".venv" (
    echo [1/3] Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: failed to create venv. Is python on PATH?
        exit /b 1
    )
) else (
    echo [1/3] .venv already exists, skipping creation.
)

echo [2/3] Upgrading pip inside .venv ...
call .venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    exit /b 1
)

echo [3/3] Installing dev + runtime dependencies ...
call .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Done. To use the venv in a new terminal, run:
echo      .venv\Scripts\activate
echo  Then verify with:
echo      python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo ============================================================

endlocal
