@echo off
setlocal
cd /d "%~dp0"
title OPT Log Diameter Checker - Setup

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if not %errorlevel%==0 goto :no_python
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the private app environment...
    %PYTHON_CMD% -m venv .venv || goto :failed
)

echo Installing or updating the app...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed

echo Starting OPT Log Diameter Checker...
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:no_python
echo Python is not installed or is not available in PATH.
echo Install Python 3.11 or newer from https://www.python.org/downloads/windows/
echo During installation, tick "Add python.exe to PATH", then run this file again.
pause
goto :end

:failed
echo Setup did not finish. Check your internet connection and try again.
pause

:end
endlocal

