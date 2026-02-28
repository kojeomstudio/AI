@echo off
setlocal

:: Git root 디렉토리 찾기
for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set GIT_ROOT=%%i

if "%GIT_ROOT%"=="" (
    echo Git root를 찾을 수 없습니다.
    exit /b 1
)

echo Git root: %GIT_ROOT%
cd /d "%GIT_ROOT%"

echo 서브모듈 초기화 및 업데이트를 시작합니다... (Recursive)
git submodule update --init --recursive

echo 서브모듈 업데이트가 완료되었습니다.
pause
