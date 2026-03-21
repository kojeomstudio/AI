@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo YoutubeExtractor Build System
echo ============================================================
echo.

echo Building YoutubeExtractor C# App...
powershell -ExecutionPolicy Bypass -File "build.ps1"
if %errorlevel% neq 0 (
    echo Error while building YoutubeExtractor.
    exit /b %errorlevel%
)

echo.
echo Build Completed successfully!
echo Binary directory: ..\bin\youtube-extractor
echo.
pause
endlocal
