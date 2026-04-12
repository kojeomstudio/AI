@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 마비노기 모바일 매크로 패키지 빌드 스크립트
echo ========================================
echo.

:: 현재 디렉토리 확인
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo 현재 작업 디렉토리: %CD%
echo.

:: Python 환경 확인
echo [1/8] Python 환경 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python이 설치되지 않았거나 PATH에 없습니다.
    echo Python을 설치하고 PATH에 추가한 후 다시 실행하세요.
    pause
    exit /b 1
)
echo Python 환경 확인 완료
echo.

:: pip 확인
echo [2/8] pip 확인 중...
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip가 설치되지 않았습니다.
    pause
    exit /b 1
)
echo pip 확인 완료
echo.

:: 가상환경 생성 (선택사항)
set /p "USE_VENV=가상환경을 생성하시겠습니까? (y/n): "
if /i "%USE_VENV%"=="y" (
    echo [3/8] 가상환경 생성 중...
    if exist "venv" (
        echo 기존 가상환경을 삭제합니다...
        rmdir /s /q "venv"
    )
    python -m venv venv
    if errorlevel 1 (
        echo 가상환경 생성 실패
        pause
        exit /b 1
    )
    echo 가상환경 생성 완료
    
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
    echo 가상환경 활성화 완료
    echo.
) else (
    echo 가상환경을 사용하지 않습니다.
    echo.
)

:: PyInstaller 설치
echo [4/8] PyInstaller 설치 중...
pip install pyinstaller
if errorlevel 1 (
    echo PyInstaller 설치 실패
    pause
    exit /b 1
)
echo PyInstaller 설치 완료
echo.

:: requirements.txt 설치
echo [5/8] 의존성 패키지 설치 중...
if exist "requirements.txt" (
    echo requirements.txt에서 패키지 설치 중...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 의존성 패키지 설치 실패
        pause
        exit /b 1
    )
    echo 의존성 패키지 설치 완료
) else (
    echo 기본 패키지만 설치합니다...
    pip install ultralytics pyautogui pywin32 opencv-python numpy pillow typer jsonschema psutil
    if errorlevel 1 (
        echo 기본 패키지 설치 실패
        pause
        exit /b 1
    )
    echo 기본 패키지 설치 완료
)
echo.

:: 기존 빌드 폴더 정리
echo [6/8] 기존 빌드 폴더 정리 중...
set "DIST_DIR=..\..\Bins\mabinogi-mobile"
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
if exist "*.spec" del /q "*.spec"
echo 빌드 폴더 정리 완료
echo.

:: PyInstaller 실행
echo [7/8] PyInstaller로 패키지 생성 중...
echo 이 과정은 몇 분 정도 소요될 수 있습니다...
echo.

pyinstaller ^
    --onefile ^
    --name "MabinogiMobileMacro" ^
    --distpath "%DIST_DIR%" ^
    --workpath "build" ^
    --add-data "config;config" ^
    --add-data "ml\training_output;ml\training_output" ^
    --add-data "assets;assets" ^
    --hidden-import "ultralytics" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "PIL" ^
    --hidden-import "pyautogui" ^
    --hidden-import "win32gui" ^
    --hidden-import "win32con" ^
    --hidden-import "win32api" ^
    --hidden-import "win32process" ^
    --hidden-import "ctypes" ^
    --hidden-import "psutil" ^
    --hidden-import "jsonschema" ^
    --hidden-import "typer" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    --exclude-module "matplotlib" ^
    --exclude-module "seaborn" ^
    --exclude-module "pandas" ^
    --exclude-module "scipy" ^
    --exclude-module "sympy" ^
    --exclude-module "networkx" ^
    app.py

if errorlevel 1 (
    echo PyInstaller 실행 실패
    echo.
    echo 오류 로그를 확인하세요.
    pause
    exit /b 1
)
echo PyInstaller 실행 완료
echo.

:: 실행 파일 확인
echo [8/8] 실행 파일 확인 중...
if exist "%DIST_DIR%\MabinogiMobileMacro.exe" (
    echo 실행 파일 생성 완료: %DIST_DIR%\MabinogiMobileMacro.exe
    echo.
    echo 파일 크기:
    for %%A in ("%DIST_DIR%\MabinogiMobileMacro.exe") do echo %%~zA bytes
    echo.
) else (
    echo 실행 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

:: 추가 파일 복사
echo 추가 파일 복사 중...
if exist "config" (
    echo config 폴더 복사 중...
    xcopy "config" "%DIST_DIR%\config\" /E /I /Y >nul
)
if exist "README_IMPROVED.md" (
    echo README 파일 복사 중...
    copy "README_IMPROVED.md" "%DIST_DIR%\" >nul
)
echo 추가 파일 복사 완료
echo.

:: 완료 메시지
echo ========================================
echo 패키지 빌드 완료!
echo ========================================
echo.
echo 실행 파일 위치: %DIST_DIR%\MabinogiMobileMacro.exe
echo.
echo 사용법:
echo 1. Bins\mabinogi-mobile 폴더의 모든 파일을 원하는 위치로 복사
echo 2. MabinogiMobileMacro.exe run (일반 모드)
echo 3. MabinogiMobileMacro.exe run --test (테스트 모드)
echo.

:: 가상환경 비활성화
if /i "%USE_VENV%"=="y" (
    deactivate
    echo 가상환경이 비활성화되었습니다.
)

pause
