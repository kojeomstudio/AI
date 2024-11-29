@echo off
REM PyInstaller로 파이썬 스크립트를 단일 실행 파일로 빌드하는 배치 스크립트

chcp 65001

cd %~dp0

call my_doc_convert_venv\Scripts\activate

echo Current Path : %~dp0

REM PyInstaller가 설치되어 있는지 확인
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PyInstaller가 설치되어 있지 않습니다. 설치를 진행합니다...
    pip install pyinstaller
)

REM 스크립트 파일명 설정
set SCRIPT_NAME=doc_convertor_main.py

REM PyInstaller를 사용하여 단일 실행 파일로 빌드
echo 프로그램을 빌드 중입니다...


REM pyinstaller --onefile --hidden-import=pypdfium2 --windowed %SCRIPT_NAME%
REM pypdfium의 경우, dll 같은 라이브러리를 동적으로 로드하기 때문에 pyinstaller가 감지 못함.

pyinstaller --onefile ^
    --add-data "my_doc_convert_venv\Lib\site-packages\pypdfium2_raw\pdfium.dll;pypdfium2_raw" ^
    --add-data "my_doc_convert_venv\Lib\site-packages\pypdfium2_raw\version.json;pypdfium2_raw" ^
    --add-data "my_doc_convert_venv\Lib\site-packages\pypdfium2\version.json;pypdfium2" ^
    --windowed %SCRIPT_NAME%