@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0src\QuadTerminal"
set "OUTPUT_DIR=%~dp0..\..\Bins\QuadTerminal"
set "LOG_DIR=%OUTPUT_DIR%\logs"
set "CONFIG=Release"
set "RID=win-x64"

if "%1"=="debug" set "CONFIG=Debug"
if "%1"=="Debug" set "CONFIG=Debug"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "LOG_FILE=%LOG_DIR%\build_%DATE:/=-%_%TIME::=-%.log"
set "LOG_FILE=%LOG_FILE: =%"

call :log "========================================"
call :log " QuadTerminal Build Script"
call :log " Configuration: %CONFIG%"
call :log " RID: %RID%"
call :log " Output: %OUTPUT_DIR%"
call :log "========================================"

call :log "Running dotnet publish..."
dotnet publish "%PROJECT_DIR%\QuadTerminal.csproj" -c %CONFIG% -r %RID% -o "%OUTPUT_DIR%" --self-contained false -p:PublishTrimmed=false -p:DebugType=None -p:DebugSymbols=false 2>&1 | tee "%LOG_DIR%\build_output.tmp"

set "ERR=%ERRORLEVEL%"
type "%LOG_DIR%\build_output.tmp" >> "%LOG_FILE%" 2>nul
del "%LOG_DIR%\build_output.tmp" 2>nul

if %ERR% EQU 0 (
    call :log "Build succeeded."
) else (
    call :log "Build FAILED with error code %ERR%"
)

call :log "========================================"
exit /b %ERR%

:log
echo %~1
echo [%TIME%] %~1 >> "%LOG_FILE%"
goto :eof
