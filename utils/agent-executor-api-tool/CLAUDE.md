# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Executor API is a FastAPI-based HTTP proxy server for executing local coding agents (Claude Code, Codex, Gemini, Cursor, Aider). It enables workflow automation tools like n8n to trigger code reviews via HTTP.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m app.main
# Or with hot reload:
uvicorn app.main:app --host 0.0.0.0 --port 9999 --reload

# Run via script (auto-creates venv)
run.bat

# Build standalone executable
build.bat  # Output: dist/agent-executor-api/

# Test API
python test_client.py
```

## Architecture

```
app/
├── main.py      # FastAPI app, endpoints, request handling
├── config.py    # Settings from config.json + env vars (priority: env > json > defaults)
├── executor.py  # AgentExecutor class - subprocess execution, prompt formatting
├── models.py    # Pydantic request/response models
prompts/         # Prompt template files (.txt) with {placeholder} syntax
```

### Key Components

**AgentExecutor** (`executor.py`):
- Executes agents via `subprocess.run()` with `shell=False` (security)
- Windows .CMD/.BAT handling: uses `shell=True` + `subprocess.list2cmdline()`
- `execute()`: Direct agent execution with raw command args
- `execute_code_review()`: Template-based code review with stdin support for Gemini

**Settings** (`config.py`):
- Multi-source config: env vars > `config.json` > defaults
- `get_prompt_template_for_agent()`: Agent-specific template lookup
- `format_prompt_template()`: Placeholder substitution ({context}, {build_id}, etc.)

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/agents` | GET | List supported agents |
| `/config` | GET | Current config (non-sensitive) |
| `/execute` | POST | Direct agent execution (JSON) |
| `/review/json` | POST | Code review with JSON body |
| `/review/form` | POST | Code review with form-data |
| `/review/raw` | POST | Code review with raw text body + query params |

### Prompt Template System

Templates in `prompts/` use `.md` (Markdown) format.

Placeholders: `{context}`, `{build_id}`, `{user_id}`, `{agent_type}`, `{timestamp}`, `{date}`, `{time}`

Template resolution order:
1. `custom_template` in request (if provided)
2. Agent-specific template from `agent_prompt_templates` config
3. Default `prompt_template_file`

### Prompt Strategy: Context Gathering

**Critical principle**: Code reviews must NOT be based on diff alone.

All templates enforce:
1. **Mandatory context gathering** - Open related files before judging
2. **Referenced files section** - List all files opened for context
3. **Line number requirement** - Issues must include `file:line` references
4. **Evidence-based review** - Explain WHY something is a problem

Required investigation before review:
- Header/interface files (class structure)
- 20-30 lines before/after changed code
- Called methods/classes (trace the call chain)
- Base class/interface (inherited behavior)
- Initialization/cleanup code (lifecycle)

## Configuration

Copy `config.json.example` to `config.json`. Key settings:
- `port`: Default 8000 (README mentions 9999 for manual uvicorn)
- `default_timeout`: 900s (15 min)
- `max_timeout`: 3600s (1 hour)
- `agent_commands`: Maps agent types to CLI executables
- `agent_prompt_templates`: Maps agent types to template files

## Error Codes

| Code | Description |
|------|-------------|
| `TIMEOUT_EXCEEDED` | Requested timeout > max_timeout |
| `UNSUPPORTED_AGENT` | Unknown agent_type |
| `EMPTY_CONTEXT` | Code review context missing |
| `COMMAND_NOT_FOUND` | Agent CLI not in PATH |
| `EXECUTION_TIMEOUT` | Process timed out |
