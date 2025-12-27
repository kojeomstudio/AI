@echo off
REM Agent Executor API - Windows Run Script

echo ========================================
echo Agent Executor API
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created successfully
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\Lib\site-packages\fastapi\" (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        exit /b 1
    )
    echo Dependencies installed successfully
    echo.
)

REM Check if .env exists
if not exist ".env" (
    echo .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env file if needed
    echo.
)

REM Start the server
echo Starting Agent Executor API...
echo Server will be available at http://localhost:8000
echo API Documentation at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m app.main

REM Deactivate virtual environment on exit
deactivate
