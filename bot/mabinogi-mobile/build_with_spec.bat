@echo off
chcp 65001 >nul

echo 마비노기 모바일 매크로 - Spec 파일 빌드
echo ======================================
echo.

:: Python 환경 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

:: PyInstaller 설치 확인
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller 설치 중...
    pip install pyinstaller
    if errorlevel 1 (
        echo ❌ PyInstaller 설치 실패
        pause
        exit /b 1
    )
)

:: 의존성 설치
echo 의존성 패키지 설치 중...
pip install -r ml\requirements.txt
if errorlevel 1 (
    echo ⚠️ 일부 패키지 설치 실패, 계속 진행합니다...
)

:: 기존 빌드 폴더 정리
echo 기존 빌드 폴더 정리 중...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

:: spec 파일로 빌드
echo spec 파일로 패키지 생성 중...
pyinstaller mabinogi_macro.spec

if errorlevel 1 (
    echo ❌ 빌드 실패
    echo.
    echo 오류 로그를 확인하세요.
    pause
    exit /b 1
)

:: 결과 확인
if exist "dist\MabinogiMobileMacro.exe" (
    echo.
    echo ✅ 빌드 완료!
    echo 실행 파일: dist\MabinogiMobileMacro.exe
    echo.
    
    :: 파일 크기 표시
    for %%A in ("dist\MabinogiMobileMacro.exe") do (
        set /a size=%%~zA/1024/1024
        echo 파일 크기: !size! MB
    )
    
    echo.
    echo 사용법:
    echo - MabinogiMobileMacro.exe (일반 실행)
    echo - MabinogiMobileMacro.exe --test (테스트 모드)
    
) else (
    echo ❌ 실행 파일을 찾을 수 없습니다.
)

pause 