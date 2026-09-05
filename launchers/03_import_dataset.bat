@echo off
setlocal
title F1 Strategy Platform - Import Dataset
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - BATCH DATASET IMPORT
echo ============================================================
echo.
echo  Imports one driver across many races / seasons into MySQL.
echo  You will be asked for: driver, years and races.
echo  The more races you import, the better the trained model.
echo  (First run re-downloads the FastF1 cache - it can take a while.)
echo.

if not exist "scripts\import_f1_dataset.py" (
    echo  [ERROR] scripts\import_f1_dataset.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\import_f1_dataset.py

echo.
echo  ============================================================
echo   DONE - batch import finished (see summary above).
echo   Next: 04_audit_pit_events.bat, then 05_train_model.bat.
echo  ============================================================
pause
