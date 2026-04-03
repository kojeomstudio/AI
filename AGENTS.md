# AGENTS.md

This is a personal AI playground monorepo containing ML studies, game automation bots, LLM agent experiments, workflow automation, and generative AI tools. Many directories are **git submodules** with their own build systems — check for `.git` files before editing.

## Repository Structure

```
/study/                  ML/DL study scripts (sklearn, tensorflow, keras, pytorch)
/bot/mabinogi-mobile/    YOLO-based game macro bot (Python, typer, pyautogui, win32api)
/tools/agent-executor-api-tool/  FastAPI HTTP proxy for coding agents (Pydantic, uvicorn)
/llm/                    LLM experiments (submodules: agents, MCP servers, RAG, search tools)
/workflow/n8n/           Workflow automation (submodule - TypeScript monorepo)
/image-generative/       ComfyUI, Stable Diffusion WebUI (submodules)
/tools/                  Utility scripts and tools (FFmpeg, image tools, YouTube extractor)
/utils/                  Shared utility directories
/web-servers/caddy/      Caddy reverse proxy config (submodule)
/reinforcement_learning/ RL study scripts
/datas/                  Static datasets (CSV, NPY, images)
/docs/                   Project documentation
```

## Build & Install Commands

### Python (root-level)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install root dependencies
pip install -r requirements.txt

# Platform-specific installs
pip install -r requirements_mac.txt    # macOS
pip install -r requirements_windows.txt # Windows
```

### Agent Executor API (`tools/agent-executor-api-tool/`)

```bash
cd tools/agent-executor-api-tool
pip install -r requirements.txt
python -m app.main
# Dev with hot reload:
uvicorn app.main:app --host 0.0.0.0 --port 9999 --reload
```

### Mabinogi Bot (`bot/mabinogi-mobile/`)

```bash
cd bot/mabinogi-mobile
pip install -r requirements.txt  # if present
python app.py run                # normal mode
python app.py run --test         # test mode
```

### Submodule Initialization

```bash
./update_submodules.sh          # macOS/Linux
.\update_submodules.ps1         # Windows
```

## Testing

No unified test runner at root level. Each subproject manages its own tests.

### Run a single test file

```bash
# Bot project (unittest)
python -m pytest bot/mabinogi-mobile/tests/test_action_processor.py
# Or with unittest directly:
python -m unittest bot.mabinogi-mobile.tests.test_action_processor

# Agent executor tool
python tools/agent-executor-api-tool/test_client.py
python tools/agent-executor-api-tool/test_agents.py

# Submodule tests (n8n example)
cd workflow/n8n && npm run test -- --filter="package-name"
```

### Run all tests in a subproject

```bash
cd bot/mabinogi-mobile && python -m pytest tests/
cd workflow/n8n && npm run test
```

## Lint & Format

There is **no root-level linter or formatter configured**. Key observations:

- No `pyproject.toml`, `ruff.toml`, `.flake8`, `.eslintrc`, or `prettier` config at root
- Some submodules have their own lint configs (e.g., `ruff.toml` in `llm/mcp/mcp-use/`)
- Submodules like `n8n` have eslint/vitest configs in their package directories

When adding Python code, follow the existing style. No automated linting is enforced at the monorepo level.

## Code Style Guidelines

### Python

**Imports**: Standard library → third-party → local. No strict alphabetical sorting observed, but group logically:

```python
import os
import sys
from typing import Dict, List

import numpy as np
from pydantic import BaseModel

from logger_helper import get_logger
from ui.base.element import ElementType
```

**Naming conventions**:
- `snake_case` for functions, variables, methods, module names, file names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants (e.g., `KEY_MAP`, `ACTION_CONFIG_SCHEMA`)
- Prefix private methods with underscore: `_load_config()`, `_signal_handler()`
- `snake_case` with `.py` extension for all Python files

**Type hints**: Use type hints for function signatures, especially in production code (`bot/`, `tools/`). Study scripts may omit them:

```python
def _execute_action(self, action_name: str, position: Optional[Tuple[int, int, int, int]] = None) -> bool:
def get_action_stats(self) -> Dict:
```

**Classes**: Use docstrings on classes and public methods. Private helper methods should also have brief docstrings:

```python
class ActionProcessor:
    """Processes detected elements and executes corresponding actions."""

    def _check_cooldown(self, action_name: str) -> bool:
        """Checks if the cooldown for a given action has passed."""
```

**String formatting**: Use f-strings (not `.format()` or `%` formatting):

```python
logger.info(f"YOLO model loaded successfully: {model_path}")
print(f"score_value (from test data) : {sgd_clf.score(x_test_scaled, y_test):.2f}")
```

**Error handling**: Catch specific exceptions, log with `exc_info=True` for unexpected errors:

```python
try:
    model = YOLO(model_path)
except FileNotFoundError:
    logger.error(f"YOLO model file not found: {model_path}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    sys.exit(1)
```

**Logging**: Use Python's `logging` module. Use descriptive log levels:
- `logger.debug()` for step-by-step tracing
- `logger.info()` for significant state changes and milestones
- `logger.warning()` for recoverable issues
- `logger.error()` with `exc_info=True` for unexpected exceptions

**Async/FastAPI**: Use `async def` for endpoint handlers. Use `@asynccontextmanager` for lifespan events. Return Pydantic models as response_model.

**CLI apps**: Use `typer` for CLI interfaces with `@app.command()` decorators.

**Configuration**: Load from JSON files with fallback defaults. Validate with `jsonschema` where applicable. For apps, use `pydantic-settings` (`BaseSettings`) for env-var-aware config.

### Language

This is a **bilingual codebase** (Korean + English):
- Comments and docstrings may be in Korean or English
- Variable/function names are always in English
- Log messages may be in either language (match the surrounding context)

### TypeScript/JavaScript (submodules)

Submodules (n8n, openclaw, open-agent, etc.) follow their own conventions. Check each submodule's `AGENTS.md`, `CONTRIBUTING.md`, or eslint config before modifying. Do not assume root-level TypeScript conventions apply to submodules.

## Important Notes

- **Never modify submodule code** unless explicitly asked — they are pinned to upstream versions. Check for a `.git` file in the directory.
- **No root-level CI/CD pipeline** exists. Each subproject handles its own build/test.
- **No root `package.json` or `pyproject.toml`** — dependency management is per-project.
- The `datas/` directory contains static datasets referenced by study scripts; do not modify or delete.
- API keys and secrets should never be committed. Use `.env` files (already gitignored via `.venv` pattern) or environment variables.
- When creating new Python files in `bot/` or `tools/`, follow the existing project structure and import patterns.
