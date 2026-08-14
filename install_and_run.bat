@echo off
setlocal
cd /d "%~dp0"
title OPT Log Diameter Checker - Setup

set "PYTHON_EXE="

where py >nul 2>nul
if %errorlevel%==0 (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %errorlevel%==0 (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not defined PYTHON_EXE goto :no_python

echo Using:
"%PYTHON_EXE%" --version

if not exist ".venv\Scripts\python.exe" (
    echo Creating the private app environment...
    "%PYTHON_EXE%" -m venv .venv || goto :failed
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
