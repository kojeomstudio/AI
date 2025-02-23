@echo off
REM Markdown Converter 빌드 스크립트

chcp 65001 >nul

cd %~dp0

REM Python 실행 파일 확인
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python이 설치되어 있지 않습니다. 설치 후 다시 시도하세요.
    pause
    exit /b
)

REM 가상 환경 생성 및 활성화 (필요한 경우)
if not exist my_markitdown_venv (
    echo 가상 환경을 생성합니다...
    python -m venv my_markitdown_venv
)

REM 가상환경 활성화
call my_markitdown_venv\Scripts\activate

echo Current Path : %~dp0

REM PyInstaller가 설치되어 있는지 확인
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PyInstaller가 설치되어 있지 않습니다. 설치를 진행합니다...
    pip install pyinstaller
)

REM markitdown 패키지가 설치되어 있는지 확인
pip show markitdown >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo markitdown 패키지가 설치되어 있지 않습니다. 설치를 진행합니다...
    pip install markitdown
)

REM 스크립트 파일명 설정
set SCRIPT_NAME=anything2markdown.py

REM PyInstaller를 사용하여 실행 파일로 빌드
echo 프로그램을 빌드 중입니다...

REM --add-data "my_markitdown_venv/Lib/site-packages/markitdown;markitdown/"
call my_markitdown_venv\Scripts\pyinstaller --onefile --windowed --collect-all markitdown --name "MarkdownConverter" %SCRIPT_NAME%

REM 빌드된 실행 파일 확인
set BUILD_DIR=dist
set EXECUTABLE_NAME=MarkdownConverter.exe

if exist "%BUILD_DIR%\%EXECUTABLE_NAME%" (
    echo 빌드가 완료되었습니다! 실행 파일이 dist 폴더에 생성되었습니다.
    explorer %BUILD_DIR%
) else (
    echo 빌드된 파일을 찾을 수 없습니다. 빌드가 실패했을 수 있습니다.
)

echo 완료.
pause
