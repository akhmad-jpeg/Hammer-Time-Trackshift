@echo off
setlocal
title F1 Strategy Platform - Performance Analysis
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - PERFORMANCE ANALYSIS
echo ============================================================
echo.
echo  Generates lap-time and tyre-degradation charts plus a summary
echo  report for a session you pick. Saved under analysis\ at the
echo  project root.
echo.

if not exist "scripts\analyze_performance.py" (
    echo  [ERROR] scripts\analyze_performance.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\analyze_performance.py

echo.
echo  DONE - reports saved under analysis\.
pause
