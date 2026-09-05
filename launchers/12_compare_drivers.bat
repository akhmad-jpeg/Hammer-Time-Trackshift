@echo off
setlocal
title F1 Strategy Platform - Compare Drivers
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - DRIVER COMPARISON
echo ============================================================
echo.
echo  Compares two drivers head-to-head using their own trained
echo  lap-time models: same track, tyre and age -> pace gap.
echo  (Requires per-driver models: run 05_train_model.bat first.)
echo.

if not exist "scripts\driver_comparison.py" (
    echo  [ERROR] scripts\driver_comparison.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

if not exist "ml_models\drivers" (
    echo  [ERROR] No per-driver models found in ml_models\drivers\.
    echo          Run 05_train_model.bat first - it now trains one
    echo          model per driver with enough laps.
    echo.
    pause
    exit /b 1
)

"%PYTHON%" scripts\driver_comparison.py %*

echo.
echo  ============================================================
echo   DONE - comparison chart saved to ml_models\comparisons\.
echo  ============================================================
pause
