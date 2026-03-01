@echo off
set ROOT_DIR=%~dp0
set VERSION=v1.0.0
echo ==========================================
echo Building CascViewerWPF Full Tool %VERSION%
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
echo Archiving binaries to version %VERSION%
echo ==========================================
if not exist "%ROOT_DIR%archive\%VERSION%" mkdir "%ROOT_DIR%archive\%VERSION%"

copy /Y "%ROOT_DIR%build\CascViewerWPF.exe" "%ROOT_DIR%archive\%VERSION%\"
copy /Y "%ROOT_DIR%build\CascViewerWPF.dll" "%ROOT_DIR%archive\%VERSION%\"
copy /Y "%ROOT_DIR%build\CascViewerWPF.pdb" "%ROOT_DIR%archive\%VERSION%\"
copy /Y "%ROOT_DIR%build\CascLib.dll" "%ROOT_DIR%archive\%VERSION%\"

echo.
echo ==========================================
echo Build and Archive Successful!
echo ==========================================
echo Output Directory: %ROOT_DIR%build
echo Archive Directory: %ROOT_DIR%archive\%VERSION%
echo.
pause
