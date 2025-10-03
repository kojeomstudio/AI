@echo off

REM Exit immediately if a command exits with a non-zero status.
setlocal

REM Define paths relative to the script location
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%\..\..\..") do set "PROJECT_ROOT=%%~fi"
set "EMBED_TRAINER_DIR=%PROJECT_ROOT%\embedding\embed_trainer"
set "VENV_DIR=%EMBED_TRAINER_DIR%\.venv"
set "REQUIREMENTS_FILE=%EMBED_TRAINER_DIR%\requirements.txt"

REM Default config file (can be overridden by argument)
set "DEFAULT_CONFIG=%EMBED_TRAINER_DIR%\config.json"
set "CONFIG_FILE=%1"
if "%CONFIG_FILE%"=="" set "CONFIG_FILE=%DEFAULT_CONFIG%"


REM --- 1. Create and activate virtual environment ---
if not exist "%VENV_DIR%" (
    echo Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate"

REM --- 2. Install dependencies ---
echo Installing/updating dependencies from %REQUIREMENTS_FILE%...
python -m pip install --upgrade pip
python -m pip install -r "%REQUIREMENTS_FILE%"

REM --- 3. Run the training script ---
echo Running embed_trainer with config: %CONFIG_FILE%
REM Ensure the project root is in PYTHONPATH for module imports
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

python -m embedding.embed_trainer.train --config "%CONFIG_FILE%"

echo Script finished.

endlocal
