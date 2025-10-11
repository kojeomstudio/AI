#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------
# 1. 기본 설정
# -----------------------------------------
# TeamCity 빌드 에이전트에서 환경 변수로 입력받는 값
PROMPT_TEXT="${PROMPT_TEXT:-}"
PROFILE=$PROFILE  # codex cli가 사용하는 모델명

# -----------------------------------------
# 2. 환경 확인 및 로그
# -----------------------------------------
echo "=============================================="
echo "🔧 Codex CLI 실행 설정"
echo "----------------------------------------------"
echo "PROMPT_TEXT: ${PROMPT_TEXT}"
echo "PROFILE    : ${PROFILE}"
echo "=============================================="

# 3. Codex CLI 실행
# -----------------------------------------
echo "🚀 Codex CLI 실행 중..."
codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    --profile "${PROFILE}" \
    "${PROMPT_TEXT}"