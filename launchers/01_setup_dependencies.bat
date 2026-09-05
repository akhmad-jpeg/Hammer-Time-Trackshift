@echo off
setlocal
title F1 Strategy Platform - Setup Dependencies
cd /d "%~dp0.."

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - SETUP DEPENDENCIES
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  [SETUP] No virtual environment found - creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo  [ERROR] Could not create .venv.
        echo          Install Python 3.13 and make sure 'python' is on PATH.
        pause
        exit /b 1
    )
) else (
    echo  [SETUP] Using existing virtual environment at .venv
)

echo  [SETUP] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo  [SETUP] Installing requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [WARNING] pip install reported an error - check the output above.
)

echo.
echo  ============================================================
echo   DONE - dependencies installed into .venv
echo   Next: run 02_import_race.bat to bring in your first race.
echo  ============================================================
pause
