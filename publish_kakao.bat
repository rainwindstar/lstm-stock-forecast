@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" start_share.py --provider cloudflare
) else (
    python start_share.py --provider cloudflare
)
if errorlevel 1 pause
