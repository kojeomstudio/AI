#!/bin/bash

# 실행할 .sh 파일 경로를 변수에 저장
SCRIPT_PATH="/kojeomstudio/stable-diffusion-webui/webui.sh"

# 파일이 존재하는지 확인
if [ -f "$SCRIPT_PATH" ]; then
    echo "Executing $SCRIPT_PATH..."
    # 실행 권한 부여 후 실행
    chmod +x "$SCRIPT_PATH"
    "$SCRIPT_PATH"
else
    echo "Error: $SCRIPT_PATH does not exist."
    exit 1
fi
