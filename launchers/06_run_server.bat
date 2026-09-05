@echo off
setlocal
title F1 Strategy Platform - Dashboard Server
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - DASHBOARD SERVER
echo ============================================================
echo.
echo  Starting the web dashboard. Press Ctrl+C to stop.
echo  Open the printed link (http://localhost:5000) in your browser.
echo  (If the model is missing, the dashboard still shows charts -
echo   run 05_train_model.bat to enable predictions/strategy advice.)
echo.

if not exist "scripts\run_server.py" (
    echo  [ERROR] scripts\run_server.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\run_server.py

echo.
echo  Server stopped.
pause
