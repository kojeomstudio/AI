@echo off
set ROOT_DIR=%~dp0
echo ==========================================
echo Building CascViewerWPF Full Tool
echo ==========================================

rem Check if dotnet is available
where dotnet >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: dotnet SDK is not installed or not in PATH.
    exit /b 1
)

echo Starting dotnet build...
dotnet build "%ROOT_DIR%mod\casc-viewer-wpf\CascViewerWPF.csproj" -c Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Tool build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo Output Directory: %ROOT_DIR%mod\casc-viewer-wpf\bin\Release
et8.0-windows
echo.
pause
