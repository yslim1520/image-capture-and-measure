@echo off
setlocal
cd /d "%~dp0"
title OPT Log Diameter Checker
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup is required. Opening install_and_run.bat...
    call install_and_run.bat
    goto :end
)
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --browser.gatherUsageStats=false
:end
endlocal
