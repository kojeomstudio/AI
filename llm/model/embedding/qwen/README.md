**Environment Setup**
- Purpose: Recreate two Python virtual environments from the included requirements files for base and GTX variants.

**Virtualenvs**
- Base venv: `.venv` from `requirements.txt`.
- GTX venv: `.venv_gtx` from `requirements_gtx.txt`.

**Prerequisites**
- Python: 3.8+ available on PATH (`python3`, `python`, or Windows `py -3`).
- Pip: installed with Python (the scripts upgrade `pip`).

**macOS/Linux**
- Run: `bash tools/setup_venvs.sh`
- Result: Recreates `.venv` and `.venv_gtx`, installs dependencies, runs `pip check`.
- Activate base: `source .venv/bin/activate`
- Activate GTX: `source .venv_gtx/bin/activate`

**Windows (PowerShell)**
- Run: `pwsh -File tools/setup_venvs.ps1` (or `powershell -ExecutionPolicy Bypass -File tools/setup_venvs.ps1`)
- Activate base: `.\.venv\Scripts\Activate.ps1`
- Activate GTX: `.\.venv_gtx\Scripts\Activate.ps1`

**Behavior**
- If a virtual environment directory already exists, it is deleted and recreated (clean install).
- The scripts verify the requirements files exist and perform a basic dependency validation via `pip check`.

**Notes**
- If your system uses a proxy or private index, configure pip accordingly (e.g., `PIP_INDEX_URL`).
- Existing `.venv` in this repo is safe to remove; the scripts will recreate it.

