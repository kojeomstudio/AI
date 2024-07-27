@echo off
REM 코드 페이지를 UTF-8로 설정
chcp 65001

REM 경로 설정
set "env_path=llama_env\Scripts"
set "target_path=llama_Scripts"

REM 가상 환경 활성화
call "%env_path%\activate.bat"

REM 경로 이동
cd /d "%target_path%"

REM 명령 프롬프트가 종료되지 않게 유지
cmd /k "echo 가상 환경이 활성화되고, 경로가 %target_path%로 이동되었습니다."
