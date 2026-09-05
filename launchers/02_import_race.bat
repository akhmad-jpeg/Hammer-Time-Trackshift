@echo off
setlocal
title F1 Strategy Platform - Import Race
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - IMPORT ONE RACE
echo ============================================================
echo.
echo  You will be asked for: year, race, session type and driver.
echo  Data is pulled from FastF1 and written to your MySQL database.
echo  (The first run re-downloads the FastF1 cache - it can take a
echo   while. Later imports reuse the cached f1_cache\ folder.)
echo.

if not exist "scripts\import_f1_race.py" (
    echo  [ERROR] scripts\import_f1_race.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\import_f1_race.py

echo.
echo  ============================================================
echo   DONE - session imported (if no error above).
echo   Repeat this launcher for more races, or use
echo   03_import_dataset.bat to batch-import many races at once.
echo  ============================================================
pause
