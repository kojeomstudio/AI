@echo off
REM ============================================
REM Agent Executor API - Windows Build Script
REM ============================================
REM This script builds a standalone executable using PyInstaller
REM with automatic virtual environment management

setlocal enabledelayedexpansion

REM Colors and formatting (using standard CMD escape sequences)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARN=[WARN]"
set "STEP=[STEP]"

echo.
echo ========================================================================
echo                    Agent Executor API - Build Script
echo ========================================================================
echo.
echo %INFO% Starting build process...
echo %INFO% Timestamp: %date% %time%
echo.

REM ============================================
REM Step 1: Check Python installation
REM ============================================
echo %STEP% [1/7] Checking Python installation...
echo ------------------------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python is not installed or not in PATH
    echo %ERROR% Please install Python 3.10 or higher
    echo %ERROR% Download from: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo %SUCCESS% Found %PYTHON_VERSION%
echo.

REM ============================================
REM Step 2: Check/Create virtual environment
REM ============================================
echo %STEP% [2/7] Checking virtual environment...
echo ------------------------------------------------------------------------

if exist "venv\" (
    echo %INFO% Virtual environment already exists at: %cd%\venv
    echo %INFO% Checking if it's valid...

    if exist "venv\Scripts\python.exe" (
        echo %SUCCESS% Virtual environment is valid
    ) else (
        echo %WARN% Virtual environment is corrupted, recreating...
        rmdir /s /q venv
        goto CREATE_VENV
    )
) else (
    :CREATE_VENV
    echo %INFO% Creating new virtual environment...
    python -m venv venv

    if errorlevel 1 (
        echo %ERROR% Failed to create virtual environment
        exit /b 1
    )

    echo %SUCCESS% Virtual environment created successfully
)
echo.

REM ============================================
REM Step 3: Activate virtual environment
REM ============================================
echo %STEP% [3/7] Activating virtual environment...
echo ------------------------------------------------------------------------

if not exist "venv\Scripts\activate.bat" (
    echo %ERROR% Virtual environment activation script not found
    exit /b 1
)

call venv\Scripts\activate.bat
echo %SUCCESS% Virtual environment activated
echo %INFO% Python location: !cd!\venv\Scripts\python.exe
echo.

REM ============================================
REM Step 4: Upgrade pip
REM ============================================
echo %STEP% [4/7] Upgrading pip...
echo ------------------------------------------------------------------------

python -m pip install --upgrade pip
if errorlevel 1 (
    echo %WARN% Failed to upgrade pip, continuing anyway...
) else (
    echo %SUCCESS% pip upgraded successfully
)

for /f "tokens=*" %%i in ('pip --version') do set PIP_VERSION=%%i
echo %INFO% %PIP_VERSION%
echo.

REM ============================================
REM Step 5: Install dependencies
REM ============================================
echo %STEP% [5/7] Installing dependencies...
echo ------------------------------------------------------------------------

echo %INFO% Reading requirements from: requirements.txt
echo %INFO% This may take a few minutes...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo %ERROR% Failed to install dependencies
    echo %ERROR% Check the error messages above
    exit /b 1
)

echo.
echo %SUCCESS% All dependencies installed successfully
echo.

REM ============================================
REM Step 6: Clean previous builds
REM ============================================
echo %STEP% [6/7] Cleaning previous builds...
echo ------------------------------------------------------------------------

if exist "dist\" (
    echo %INFO% Removing old dist directory...
    rmdir /s /q dist
    echo %SUCCESS% Old dist directory removed
)

if exist "build\" (
    echo %INFO% Removing old build directory...
    rmdir /s /q build
    echo %SUCCESS% Old build directory removed
)

echo %SUCCESS% Build directories cleaned
echo.

REM ============================================
REM Step 7: Build executable with PyInstaller
REM ============================================
echo %STEP% [7/7] Building executable with PyInstaller...
echo ------------------------------------------------------------------------

echo %INFO% Using spec file: agent-executor-api.spec
echo %INFO% Build mode: One folder (with dependencies)
echo %INFO% This may take several minutes...
echo.

pyinstaller --clean --noconfirm agent-executor-api.spec

if errorlevel 1 (
    echo.
    echo %ERROR% Build failed
    echo %ERROR% Check the error messages above
    exit /b 1
)

echo.
echo %SUCCESS% Build completed successfully
echo.

REM ============================================
REM Step 8: Verify build output
REM ============================================
echo ========================================================================
echo                         Build Verification
echo ========================================================================
echo.

if exist "dist\agent-executor-api\agent-executor-api.exe" (
    echo %SUCCESS% Executable created successfully!
    echo.
    echo Location: %cd%\dist\agent-executor-api\
    echo Executable: agent-executor-api.exe
    echo.

    REM Get file size
    for %%A in ("dist\agent-executor-api\agent-executor-api.exe") do (
        set SIZE=%%~zA
        set /a SIZE_MB=!SIZE! / 1048576
        echo File size: !SIZE_MB! MB
    )

    echo.
    echo %INFO% Additional files in distribution:
    dir /b "dist\agent-executor-api" | findstr /v "agent-executor-api.exe"

) else (
    echo %ERROR% Executable not found in expected location
    echo %ERROR% Build may have failed
    exit /b 1
)

echo.

REM ============================================
REM Step 9: Copy configuration files
REM ============================================
echo ========================================================================
echo                    Copying Configuration Files
echo ========================================================================
echo.

echo %INFO% Copying config.json.example to distribution...
copy /y "config.json.example" "dist\agent-executor-api\config.json.example" >nul
if errorlevel 1 (
    echo %WARN% Failed to copy config.json.example
) else (
    echo %SUCCESS% config.json.example copied
)

echo %INFO% Copying config.json file...
if exist "config.json" (
    copy /y "config.json" "dist\agent-executor-api\config.json" >nul
    if errorlevel 1 (
        echo %WARN% Failed to copy config.json
    ) else (
        echo %SUCCESS% config.json copied
    )
) else (
    echo %INFO% config.json not found, creating from example...
    copy /y "config.json.example" "dist\agent-executor-api\config.json" >nul
    if errorlevel 1 (
        echo %WARN% Failed to create config.json
    ) else (
        echo %SUCCESS% config.json created from example
    )
)

echo.
echo %INFO% Copying prompts directory to distribution...
if exist "prompts\" (
    mkdir "dist\agent-executor-api\prompts" 2>nul
    xcopy /y /e /i "prompts\*" "dist\agent-executor-api\prompts\" >nul
    if errorlevel 1 (
        echo %WARN% Failed to copy prompts directory
    ) else (
        echo %SUCCESS% prompts directory copied
    )
) else (
    echo %WARN% prompts directory not found, skipping
)

echo.

REM ============================================
REM Final Summary
REM ============================================
echo ========================================================================
echo                           Build Summary
echo ========================================================================
echo.
echo %SUCCESS% Build completed successfully!
echo.
echo Distribution directory: %cd%\dist\agent-executor-api\
echo.
echo To run the application:
echo   1. Navigate to: dist\agent-executor-api\
echo   2. Edit config.json as needed
echo   3. Run: agent-executor-api.exe
echo.
echo Configuration files:
echo   - config.json: Main configuration (host, port, agents, templates, logging, etc.)
echo   - prompts/: Prompt template files (customizable)
echo.
echo To test the build:
echo   cd dist\agent-executor-api
echo   agent-executor-api.exe
echo.
echo ========================================================================
echo.

REM Deactivate virtual environment
call deactivate 2>nul

endlocal
exit /b 0
