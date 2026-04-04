@echo off
setlocal

echo ========================================
echo  QuadTerminal Build Script
echo ========================================
echo.

set "PROJECT_DIR=%~dp0src\QuadTerminal"
set "OUTPUT_DIR=%~dp0..\..\Bins\QuadTerminal"
set "CONFIG=Release"

if "%1"=="debug" set "CONFIG=Debug"
if "%1"=="Debug" set "CONFIG=Debug"

echo Configuration: %CONFIG%
echo Output: %OUTPUT_DIR%
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

dotnet publish "%PROJECT_DIR%\QuadTerminal.csproj" ^
    -c %CONFIG% ^
    -o "%OUTPUT_DIR%" ^
    --self-contained false ^
    -p:PublishTrimmed=false ^
    -p:DebugType=None ^
    -p:DebugSymbols=false

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  Build succeeded!
    echo  Output: %OUTPUT_DIR%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo  Build failed with error code %ERRORLEVEL%
    echo ========================================
)

endlocal
