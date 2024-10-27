#!/bin/zsh

# 현재 스크립트의 경로를 가져오기
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 가상환경 경로 설정 (스크립트 경로를 기준으로 상대 경로)
VENV_PATH="$SCRIPT_DIR/my_r2r_venv"

# 가상환경 존재 여부 확인
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 새로운 터미널 프로세스에서 가상환경 폴더로 이동하고 활성화
osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && source '$VENV_PATH/bin/activate'\""