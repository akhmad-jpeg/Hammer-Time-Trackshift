@echo off
setlocal
title F1 Strategy Platform - Capture Telemetry
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 2017/2018 TELEMETRY CAPTURE - SESSION SETUP
echo ============================================================
echo.
echo  Tracks (must match a name in the tracks table or an alias):
echo    Examples: Sochi, Monaco, Spa, Silverstone, Bahrain, Monza, Suzuka
set /p TRACK="Enter Track Name: "

echo.
echo  Tyre Compounds (choose one):
echo    Hypersoft  Ultrasoft  Supersoft  Soft  Medium  Hard
echo    Intermediate  Wet
set /p TYRE="Enter Starting Tyre Compound: "

echo.
echo  Weather options: Dry  Wet  Mixed
echo    (Also accepted: Clear, Rain, Cloudy, Showers, Overcast)
set /p WEATHER="Enter Weather: "

echo.
echo  Heartbeat interval (seconds between live-data heartbeat lines):
echo    Smaller = faster tick, e.g. 1-2 for quick feedback.
echo    0 disables the heartbeat (uses the old packet-count status line).
set "HEARTBEAT=5"
set /p HEARTBEAT="Enter Heartbeat Interval in seconds [5]: "

echo.
echo  ============================================================
echo   SESSION CONFIGURATION
echo  ============================================================
echo    Track     : %TRACK%
echo    Tyre      : %TYRE%
echo    Weather   : %WEATHER%
echo    Heartbeat : every %HEARTBEAT%s
echo    Driver    : 0 (Player - game telemetry sentinel)
echo  ============================================================
echo.
echo  Starting telemetry capture... Press Ctrl+C to stop.
"%PYTHON%" scripts\capture_telemetry.py --track "%TRACK%" --tyre "%TYRE%" --weather "%WEATHER%" --heartbeat "%HEARTBEAT%"

echo.
echo  Capture stopped.
pause
