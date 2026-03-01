@echo off
set ROOT_DIR=%~dp0
set PROJECT_FILE="%ROOT_DIR%mod\casc-viewer-wpf\CascViewerWPF.csproj"

echo ==========================================
echo Detecting Project Version...
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "([xml](Get-Content %PROJECT_FILE%)).Project.PropertyGroup.Version"') do set VERSION=v%%i

if "%VERSION%"=="v" (
    set VERSION=v1.0.0
    echo [WARNING] Could not detect version, defaulting to v1.0.0
)

echo Building CascViewerWPF Full Tool %VERSION%
echo ==========================================

rem Check if dotnet is available
where dotnet >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: dotnet SDK is not installed or not in PATH.
    exit /b 1
)

echo Starting dotnet build...
dotnet build %PROJECT_FILE% -c Release

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
