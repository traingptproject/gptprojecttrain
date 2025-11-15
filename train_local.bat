@echo off
setlocal ENABLEDELAYEDEXPANSION
REM ========================================================================
REM dLNk GPT - Robust Background Training Starter
REM ========================================================================

REM Always run from this script's folder
cd /d %~dp0

echo.
echo ========================================================================
echo dLNk GPT - Starting Training (background)
echo ========================================================================
echo.

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate.bat
) else (
  echo [INFO] No venv found. Using system Python.
)

REM Safer defaults to reduce VRAM and noise
set TOKENIZERS_PARALLELISM=false
if "%USE_8BIT%"=="" set USE_8BIT=1
if "%RESUME%"=="" set RESUME=1

REM Prepare logs directory and file names
if not exist train_logs mkdir train_logs
set LOG_LATEST=train_logs\latest.log
set LOG_TIME=train_logs\train_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%_%TIME:~0,2%-%TIME:~3,2%-%TIME:~6,2%.log
set LOG_TIME=%LOG_TIME: =0%

REM Rotate previous latest
if exist "%LOG_LATEST%" move /Y "%LOG_LATEST%" "%LOG_TIME%" >nul

echo [INFO] Logging to %LOG_LATEST%

REM Start minimized in background, unbuffered output
start "dLNk GPT Training" /min cmd /c python -u train_local.py >> "%LOG_LATEST%" 2>>&1

REM Give it a moment and do a basic sanity check
timeout /t 3 /nobreak >nul
if not exist "%LOG_LATEST%" (
  echo [ERROR] Log file not created. Try running in foreground:  python -u train_local.py
  exit /b 1
)

for %%A in ("%LOG_LATEST%") do set SZ=%%~zA
if "!SZ!"=="0" (
  echo [WARN] Log file is empty yet. Process may still be starting.
) else (
  echo [OK] Training started. Tail log with:
  echo      PowerShell: Get-Content -Path "%LOG_LATEST%" -Wait
)

exit /b 0
