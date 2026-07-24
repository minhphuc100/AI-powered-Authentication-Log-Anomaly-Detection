@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay .venv\Scripts\python.exe
    echo Hay tao moi truong va cai requirements.txt truoc khi chay.
    exit /b 2
)

".venv\Scripts\python.exe" -m src.deployment.terminal_demo %*
exit /b %ERRORLEVEL%
