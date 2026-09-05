@echo off
setlocal
title F1 Strategy Platform - Audit Pit Events
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - PIT-EVENT CONSISTENCY CHECK
echo ============================================================
echo.
echo  The first pass is a dry run: it only reports problems and
echo  changes nothing in the database.
echo.

"%PYTHON%" scripts\cleanup_pit_events.py

echo.
set /p APPLY="Apply the fixes to the database now? (y/N): "
if /I "%APPLY%"=="y" (
    echo.
    echo  Applying fixes ...
    "%PYTHON%" scripts\cleanup_pit_events.py --apply
) else (
    echo.
    echo  Skipped - the database was not modified.
)

echo.
pause
