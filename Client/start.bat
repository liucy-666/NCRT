@echo off
cd /d "%~dp0\.."
set PYTHONPATH=%~dp0..
.jailbreak\Scripts\python.exe Client\launcher.py
pause
