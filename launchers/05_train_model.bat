@echo off
setlocal
title F1 Strategy Platform - Train Model
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - TRAIN LAP-TIME MODEL
echo ============================================================
echo.
echo  Reads cleaned laps from the database, fits LinearRegression
echo  vs RandomForest and saves the winner to ml_models\.
echo  (Requires the database to contain imported races first.)
echo.

if not exist "scripts\ml_lap_predictions.py" (
    echo  [ERROR] scripts\ml_lap_predictions.py not found.
    echo          Keep this launcher inside the f1-stategy-platform folder.
    pause
    exit /b 1
)

"%PYTHON%" scripts\ml_lap_predictions.py

echo.
echo  ============================================================
echo   DONE - artifacts saved to ml_models\:
echo     best_model.pkl  feature_names.pkl  model_info.json
echo     model_info.txt  predictions_vs_actual.png
echo   Next: 06_run_server.bat to view the dashboard.
echo  ============================================================
pause
