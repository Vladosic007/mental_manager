@echo off
title Mental Manager WebApp
cd /d %~dp0webapp
echo ============================================
echo   Mental Manager - starting web server...
echo   Open in browser:  http://localhost:8000
echo   (first launch takes ~15 sec - loads the AI model)
echo   Press Ctrl+C to stop
echo ============================================
rem --- free port 8000 if something already uses it ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /PID %%p /F >nul 2>&1
python server.py
pause
