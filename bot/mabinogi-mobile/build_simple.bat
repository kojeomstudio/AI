@echo off
chcp 65001 >nul

echo 마비노기 모바일 매크로 - 간단 빌드 스크립트
echo ===========================================
echo.

:: PyInstaller 설치 확인
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller 설치 중...
    pip install pyinstaller
)

:: 의존성 설치
echo 의존성 패키지 설치 중...
if exist "requirements.txt" (
    pip install -r requirements.txt
)

:: 기존 빌드 폴더 정리
set "DIST_DIR=..\..\Bins\mabinogi-mobile"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "*.spec" del /q "*.spec"

:: PyInstaller 실행
echo 패키지 생성 중... (잠시만 기다려주세요)
pyinstaller --onefile --name "MabinogiMacro" ^
    --distpath "%DIST_DIR%" ^
    --workpath "build" ^
    --add-data "config;config" ^
    --add-data "ml\training_output;ml\training_output" ^
    --hidden-import "ultralytics" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "pyautogui" ^
    --hidden-import "win32gui" ^
    --hidden-import "win32con" ^
    --hidden-import "win32api" ^
    app.py

if exist "%DIST_DIR%\MabinogiMacro.exe" (
    echo.
    echo 빌드 완료! 실행 파일: %DIST_DIR%\MabinogiMacro.exe
) else (
    echo 빌드 실패
)

pause
