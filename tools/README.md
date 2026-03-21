# AI Tools

This directory contains various utility tools for the AI project.

## Development Rules

1.  **Binary Output**: All tools should be configured to generate their output binaries in the `tools/bin/<tool-name>` directory.
    *   Example: `tools/bin/clip-master`, `tools/bin/agent-executor-api`.
2.  **Build Scripts**: Each tool should provide build scripts (`build.ps1`, `build.bat`, or `build.sh`) that automate the process of building and deploying binaries to the standard output directory.
3.  **Logging**: Tools that perform background or batch processing should implement logging to a file within their executable directory for easier troubleshooting.
