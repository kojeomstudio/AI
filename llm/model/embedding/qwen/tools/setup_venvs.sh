#!/usr/bin/env bash
set -euo pipefail

# Recreates two Python virtual environments at the repo root:
# - .venv      from requirements.txt
# - .venv_gtx  from requirements_gtx.txt
# If the venv directory exists, it is deleted and recreated.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

req_base="$repo_root/requirements.txt"
req_gtx="$repo_root/requirements_gtx.txt"

venv_base="$repo_root/.venv"
venv_gtx="$repo_root/.venv_gtx"

log() { printf "\n[setup] %s\n" "$*"; }

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    log "Python not found. Please install Python 3.8+ and re-run."
    exit 1
  fi
}

create_venv() {
  local venv_path=$1
  local req_file=$2

  if [[ ! -f "$req_file" ]]; then
    log "Requirements file not found: $req_file"
    exit 1
  fi

  # Remove existing venv to ensure a clean environment
  if [[ -d "$venv_path" ]]; then
    log "Removing existing venv: $venv_path"
    rm -rf "$venv_path"
  fi

  local PYTHON
  PYTHON=$(find_python)

  log "Creating venv: $venv_path"
  "$PYTHON" -m venv "$venv_path"

  local pybin="$venv_path/bin/python"
  if [[ ! -x "$pybin" ]]; then
    # Fallback for environments that may use Windows-style layout
    pybin="$venv_path/Scripts/python"
  fi

  log "Upgrading pip"
  "$pybin" -m pip install --upgrade pip >/dev/null

  log "Installing dependencies from $(basename "$req_file")"
  "$pybin" -m pip install -r "$req_file"

  log "Validating installed packages (pip check)"
  "$pybin" -m pip check || true

  log "Python version in $venv_path: $($pybin -c 'import sys; print(sys.version)')"
}

log "Starting environment setup in: $repo_root"

create_venv "$venv_base" "$req_base"
create_venv "$venv_gtx" "$req_gtx"

log "All done. Created venvs: $venv_base and $venv_gtx"

