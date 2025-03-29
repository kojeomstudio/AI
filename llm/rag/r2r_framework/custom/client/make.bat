@echo off

chcp 65001

setlocal

REM 가상환경 이름 및 경로 설정
set VENV_DIR=chatbot_venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe
set PYINSTALLER=%VENV_DIR%\Scripts\pyinstaller.exe

REM 가상환경이 없으면 생성
if not exist %PYTHON_BIN% (
    echo [정보] 가상환경이 존재하지 않아 새로 생성합니다: %VENV_DIR%
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [오류] 가상환경 생성 실패.
        exit /b 1
    )
)

REM 필수 패키지 설치
echo [정보] 필요한 패키지를 설치합니다.
REM pip 업그레이드 (기존 방식에서 변경)
%PYTHON_BIN% -m pip install --upgrade pip
%PIP_BIN% install pyinstaller r2r==3.5.3
if errorlevel 1 (
    echo [오류] 패키지 설치 실패.
    exit /b 1
)

REM PyInstaller로 빌드 실행
%PYINSTALLER% --onefile --windowed --name xh_cheat_chatbot main.py
if errorlevel 1 (
    echo [오류] 빌드 실패.
    exit /b 1
)

REM config.json을 dist 폴더로 복사
echo [정보] config.json 복사 중...
if exist dist (
    copy /Y config.json dist\\
    if errorlevel 1 (
        echo [오류] config.json 복사 실패.
        exit /b 1
    )
)

echo [성공] 빌드가 완료되었습니다.
pause
