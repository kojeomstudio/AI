#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
ENTRY="${ENTRY:-run_training.py}"
NAME="${NAME:-package_size_predict}"

if [ ! -d "$VENV_PATH" ]; then
  python -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
pip install -r "$REQ_FILE"

# Build single-file binary
pyinstaller --onefile --name "$NAME" \
  --hidden-import openpyxl \
  --hidden-import pandas \
  --collect-all pandas \
  --collect-all openpyxl \
  "$ENTRY"

echo "Build complete. Binary at ./dist/$NAME"

