@echo off
setlocal
title F1 Strategy Platform - Model Benchmark
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - MODEL-SELECTION BENCHMARK
echo ============================================================
echo.
echo  Reproducible comparison of LinearRegression vs tree baselines
echo  and mixed models, on known-track, new-stint and unseen-track
echo  contracts. Does NOT overwrite ml_models\ - informational only.
echo.

if not exist "scripts\benchmark_models.py" (
    echo  [ERROR] scripts\benchmark_models.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\benchmark_models.py

echo.
pause
