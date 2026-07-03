@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set PYTHONPATH=%~dp0..

set PYTHON_EXE=

if exist ".jailbreak\Scripts\python.exe" (
    set PYTHON_EXE=.jailbreak\Scripts\python.exe
    goto :found
)
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    goto :found
)
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    goto :found
)

where python >nul 2>&1
if not errorlevel 1 (
    set PYTHON_EXE=python
    goto :found
)

echo [ERROR] Python not found. Install Python or create a venv:
echo         python -m venv .jailbreak
pause
exit /b 1

:found
echo Using Python: !PYTHON_EXE!
!PYTHON_EXE! Client\launcher.py
pause
endlocal
