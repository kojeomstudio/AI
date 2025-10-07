#!/bin/bash

SCRIPT_PATH="/Users/kojeomstudio/stable-diffusion-webui/webui.sh"

if [ -f "$SCRIPT_PATH" ]; then
    echo "Executing $SCRIPT_PATH in API mode..."
    chmod +x "$SCRIPT_PATH"
    "$SCRIPT_PATH" --api
else
    echo "Error: $SCRIPT_PATH does not exist."
    exit 1
fi