@echo off
setlocal
title F1 Strategy Platform - Energy Simulation
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - ENERGY SIMULATION (synthetic ERS)
echo ============================================================
echo.
echo  Regenerates the race_state energy rows for one session under
echo  one of three deployment modes:
echo    balanced   - default trace: starts FULL, drains through the
echo                 opening laps, settles in the soft 30-80%% band
echo    push       - drains the store to an energy-limited low band
echo    liftcoast  - banks the battery toward ~90%%
echo.
echo  Session defaults to the latest 2026 race if left blank.
echo.

set "SID="
set /p SID="Enter session id [blank = latest 2026 race]: "

set "MODE=balanced"
set /p MODE="Enter mode [balanced]: "
if "%MODE%"=="" set "MODE=balanced"

if not exist "scripts\energy_simulator.py" (
    echo  [ERROR] scripts\energy_simulator.py not found.
    echo          Keep this launcher inside the repo folder.
    pause
    exit /b 1
)

set "SID_FLAG="
if not "%SID%"=="" set "SID_FLAG=--session %SID%"

"%PYTHON%" scripts\energy_simulator.py --mode %MODE% %SID_FLAG%

echo.
echo  ============================================================
echo   DONE - energy rows written to race_state for the session.
echo   View the trace on the dashboard's Energy Management chart,
echo   or use the SIMULATE button on the energy card to re-run
echo   another mode live.
echo  ============================================================
pause
