@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ClipMaster Build System (Full Pipeline)
echo ============================================================
echo.

echo [1/3] Checking FFmpeg binaries...
powershell -ExecutionPolicy Bypass -File "copy_ffmpeg.ps1"
if %errorlevel% neq 0 (
    echo Error while setting up FFmpeg binaries.
    exit /b %errorlevel%
)

echo.
echo [2/3] Building ClipMaster C# App...
powershell -ExecutionPolicy Bypass -File "build.ps1"
if %errorlevel% neq 0 (
    echo Error while building ClipMaster.
    exit /b %errorlevel%
)

echo.
echo [3/3] Build Completed successfully!
echo Binary directory: ..\bin\clip-master
echo.
pause
endlocal
