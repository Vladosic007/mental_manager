@echo off
title Mental Manager WebApp
cd /d %~dp0webapp
echo ============================================
echo   Mental Manager - web server starting...
echo   Open in browser:  http://localhost:8000
echo   Press Ctrl+C to stop
echo ============================================
python server.py
pause
