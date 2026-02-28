#!/bin/bash
# 이 스크립트는 git root(AI 리포지토리)에서 실행하거나
# 현재 폴더(diablo2)에서 실행해도 상위 AI 리포지토리의 서브모듈을 초기화합니다.

# 현재 디렉토리 저장
PUSHED_DIR=$PWD

# git root 찾기
GIT_ROOT=$(git rev-parse --show-toplevel)

if [ -z "$GIT_ROOT" ]; then
    echo "Git root를 찾을 수 없습니다."
    exit 1
fi

echo "Git root: $GIT_ROOT"
cd "$GIT_ROOT"

echo "서브모듈 초기화 및 업데이트를 시작합니다... (Recursive)"
git submodule update --init --recursive

# 원래 디렉토리로 복귀
cd "$PUSHED_DIR"

echo "서브모듈 업데이트가 완료되었습니다."
