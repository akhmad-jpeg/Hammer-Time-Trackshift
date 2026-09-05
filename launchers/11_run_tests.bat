@echo off
setlocal
title F1 Strategy Platform - Run Tests
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

cls
echo ============================================================
echo  F1 STRATEGY PLATFORM - TEST SUITE
echo ============================================================
echo.
echo  Running all unit tests. No database needed - DB access is
echo  mocked, so this also works before any data is imported.
echo.

if not exist "tests" (
    echo  [INFO] No tests\ folder found in this checkout -
    echo         nothing to run. The test suite is optional and
    echo         should sit in a tests\ folder at the project
    echo         root; add it and re-run this launcher.
    echo.
    pause
    exit /b 0
)

"%PYTHON%" -m unittest discover -s tests

if errorlevel 1 (
    echo.
    echo  ============================================================
    echo   TESTS FAILED - see the output above.
    echo  ============================================================
) else (
    echo.
    echo  ============================================================
    echo   ALL TESTS PASSED.
    echo  ============================================================
)

pause
