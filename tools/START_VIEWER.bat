@echo off
cd /d "%~dp0"
start /min python viewer_server.py
timeout /t 1 /nobreak >nul
start chrome http://localhost:8080/run_viewer.html
