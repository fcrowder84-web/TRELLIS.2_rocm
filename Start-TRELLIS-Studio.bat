@echo off
setlocal
title Spirit Legacy TRELLIS Studio
set HOST=100.125.111.71
set USER=foster
set REPO=/home/foster/TRELLIS.2-ROCm
set PORT=7860

echo Starting Spirit Legacy TRELLIS Studio on %HOST%...
ssh %USER%@%HOST% "mkdir -p /home/foster/trellis2-outputs && if pgrep -f '[s]pirit_legacy_gui.py' >/dev/null; then echo TRELLIS Studio already running; else nohup %REPO%/tools/start_spirit_legacy_gui.sh >/home/foster/trellis2-outputs/trellis-studio.log 2>&1 </dev/null & echo TRELLIS Studio started; fi"
if errorlevel 1 (
  echo Failed to start TRELLIS Studio over SSH.
  pause
  exit /b 1
)
timeout /t 2 /nobreak >nul
start "" "http://%HOST%:%PORT%"
endlocal
