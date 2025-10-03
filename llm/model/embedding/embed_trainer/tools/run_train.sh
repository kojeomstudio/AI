#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EMBED_TRAINER_DIR="$PROJECT_ROOT/embedding/embed_trainer"
VENV_DIR="$EMBED_TRAINER_DIR/.venv"
REQUIREMENTS_FILE="$EMBED_TRAINER_DIR/requirements.txt"

# Default config file (can be overridden by argument)
DEFAULT_CONFIG="$EMBED_TRAINER_DIR/config.json"
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

# --- 1. Create and activate virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- 2. Install dependencies ---
echo "Installing/updating dependencies from $REQUIREMENTS_FILE..."
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"

# --- 3. Run the training script ---
echo "Running embed_trainer with config: $CONFIG_FILE"
# Ensure the project root is in PYTHONPATH for module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python -m embedding.embed_trainer.train --config "$CONFIG_FILE"

echo "Script finished."
