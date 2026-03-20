# Gemini Workspace Context: AI & Image Tools

This directory is a collection of utility tools focused on AI agent execution and image processing. It serves as a personal toolbox for various automation and data preparation tasks.

## Project Overview

The workspace is organized into several key areas:

### 1. Agent Executor API (`/agent-executor-api-tool`)
A FastAPI-based proxy server that allows executing locally installed coding agents (Claude Code, OpenAI Codex, Google Gemini, etc.) through an HTTP API.
- **Purpose**: Provides a standardized interface for calling various CLI-based agents.
- **Key Features**: Prompt templating system, read-only mode for safe code reviews, detailed logging, and cross-platform support.
- **Tech Stack**: FastAPI, Pydantic, Python.

### 2. Image Processing Tools (`/image`)
- **Remove Background (`/image/remove_bg`)**: Uses deep learning models (U2Net, ISNet) to remove backgrounds from general objects, humans, or anime characters.
- **Sprite Splitter (`/image/sprite`)**: An OpenCV-based GUI tool to automatically detect and split individual sprites from a sprite sheet.

### 3. Root Utilities
- **`ImageResizer.py`**: A Tkinter GUI application for bulk resizing images with optional renaming.
- **`numpy_img_file_maker.py`**: A utility for converting images into NumPy arrays, likely for machine learning training data.
- **`crlf2lf.py`**: A utility script for converting Windows line endings (CRLF) to Unix line endings (LF).
- **`wsl_windows_port_auto.ps1`**: A PowerShell script to automate port forwarding between Windows and WSL2.

## Building and Running

### Agent Executor API
- **Install Dependencies**:
  ```bash
  cd agent-executor-api-tool
  pip install -r requirements.txt
  ```
- **Configuration**: Copy `config.json.example` to `config.json` and adjust as needed.
- **Running**:
  - Windows: `run.bat`
  - Linux/Mac: `./run.sh`
  - Manual: `python -m app.main` or `uvicorn app.main:app --port 9999`
- **Building Executable**: Use `build.bat` or `build.sh` to create a standalone binary using PyInstaller.

### Image Tools
- **Remove Background**:
  ```bash
  python image/remove_bg/main.py <input_path> <model_type>
  ```
  *Models: `isnet-general-use`, `u2net_human_seg`, `isnet-anime`.*
- **Sprite Splitter**:
  ```bash
  python image/sprite/sprite_spliter.py
  ```

### Utility Scripts
- **Image Resizer**: `python ImageResizer.py`
- **Line Ending Conversion**: `python crlf2lf.py`

## Development Conventions

- **Python Version**: Python 3.x is the standard across all tools.
- **GUI Framework**: Tkinter is preferred for simple desktop utility interfaces.
- **Cross-Platform Compatibility**: Scripts often include logic to handle paths correctly when running as a script or as a bundled PyInstaller executable (see `file_path_helper.py`).
- **Path Handling**: Use `os.path.join` and the `get_base_path()` pattern to ensure tools work regardless of the current working directory.
- **Virtual Environments**: It is recommended to use local `venv` for managing tool-specific dependencies.

## Key Files Summary

- `agent-executor-api-tool/app/main.py`: Entry point for the Agent API.
- `agent-executor-api-tool/prompts/`: Directory containing Markdown templates for different agent behaviors.
- `file_path_helper.py`: Shared utility for robust path resolution across different execution environments.
- `image/sprite/sprite_spliter.py`: Main logic for OpenCV-based contour detection in sprite sheets.
