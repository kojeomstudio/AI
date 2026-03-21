## Project Overview

The workspace is organized into several key areas:

### 1. Agent Executor API (`/agent-executor-api-tool`)
A FastAPI-based proxy server for executing coding agents.

### 2. ClipMaster (`/clip-master`)
A tool for merging audio and video with fade effects. Supports both GUI and CLI.

### 3. Image Processing Tools (`/image`)
- **Remove Background (`/image/remove_bg`)**: AI-based background removal.
- **Sprite Splitter (`/image/sprite`)**: OpenCV-based sprite sheet splitter.

### 4. Utility Tools
- **CRLF to LF Converter (`/crlf2lf`)**: Utility for converting Windows line endings to Unix line endings.
- **Image Resizer (`/image-resizer`)**: Tkinter GUI application for bulk resizing images.
- **NumPy Image Maker (`/numpy-img-file-maker`)**: Utility for converting images into NumPy arrays.
- **WSL Utilities (`/wsl`)**: Automation for WSL port forwarding.
- **Shared Utilities (`/utils`)**: Common helper functions like `file_path_helper.py`.

## Development Rules

1.  **Binary Output**: All tools should be configured to generate their output binaries in the `tools/bin/<tool-name>` directory.
    *   Example: `tools/bin/clip-master`, `tools/bin/agent-executor-api`.
2.  **Build Scripts**: Each tool should provide build scripts (`build.ps1`, `build.bat`, or `build.sh`) that automate the process of building and deploying binaries to the standard output directory.
3.  **Logging**: Tools that perform background or batch processing should implement logging to a file within their executable directory for easier troubleshooting.
