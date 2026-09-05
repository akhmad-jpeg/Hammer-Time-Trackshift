@echo off
setlocal
title F1 Strategy Platform - Predict Lap Times
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - LAP-TIME PREDICTOR
echo ============================================================
echo.
echo  Interactive predictor - requires a trained model in ml_models\
echo  (run 05_train_model.bat first if this fails).
echo.

if not exist "scripts\predict_lap_times.py" (
    echo  [ERROR] scripts\predict_lap_times.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\predict_lap_times.py

echo.
pause
