#!/bin/bash
# Agent Executor API - Linux/Mac Run Script

set -e

echo "========================================"
echo "Agent Executor API"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "Virtual environment created successfully"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -d "venv/lib/python*/site-packages/fastapi" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully"
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ".env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env file if needed"
    echo ""
fi

# Start the server
echo "Starting Agent Executor API..."
echo "Server will be available at http://localhost:8000"
echo "API Documentation at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m app.main

# Deactivate virtual environment on exit
deactivate
