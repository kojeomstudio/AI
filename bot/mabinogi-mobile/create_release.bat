@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 마비노기 모바일 매크로 - 배포 패키지 생성
echo ========================================
echo.

:: 버전 정보 입력
set /p "VERSION=버전을 입력하세요 (예: 1.0.0): "
if "%VERSION%"=="" set "VERSION=1.0.0"

:: 빌드 실행
echo [1/3] 실행 파일 빌드 중...
call build_with_spec.bat
if errorlevel 1 (
    echo ❌ 빌드 실패
    pause
    exit /b 1
)

:: 배포 폴더 생성
echo [2/3] 배포 폴더 생성 중...
set "RELEASE_DIR=release\MabinogiMobileMacro_v%VERSION%"
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%"

:: 파일 복사
echo 파일 복사 중...
xcopy "dist\*" "%RELEASE_DIR%\" /E /I /Y >nul

:: 추가 파일 복사
if exist "README_IMPROVED.md" (
    copy "README_IMPROVED.md" "%RELEASE_DIR%\" >nul
)

:: 배포용 README 생성
echo [3/3] 배포용 README 생성 중...
(
echo # 마비노기 모바일 매크로 v%VERSION%
echo.
echo ## 설치 및 실행
echo.
echo 1. 이 폴더의 모든 파일을 원하는 위치에 압축 해제하세요.
echo 2. MabinogiMobileMacro.exe를 실행하세요.
echo 3. 테스트 모드: MabinogiMobileMacro.exe --test
echo.
echo ## 주의사항
echo.
echo - 마비노기 모바일 게임이 실행 중이어야 합니다.
echo - 관리자 권한으로 실행해야 할 수 있습니다.
echo - Windows Defender가 실행을 차단할 수 있습니다.
echo.
echo ## 설정
echo.
echo config 폴더의 파일을 수정하여 매크로 동작을 변경할 수 있습니다.
echo.
echo - config.json: 기본 설정
echo - action_config.json: 액션 설정
echo.
echo ## 문제 해결
echo.
echo 1. 게임 창을 찾을 수 없는 경우
echo    - 게임이 실행 중인지 확인
echo    - 창 제목이 "Mabinogi Mobile"인지 확인
echo.
echo 2. 입력이 전달되지 않는 경우
echo    - 관리자 권한으로 실행
echo    - 게임 창이 최소화되지 않았는지 확인
echo.
echo ## 버전 정보
echo.
echo - 버전: %VERSION%
echo - 빌드 날짜: %date% %time%
echo.
) > "%RELEASE_DIR%\README.txt"

:: 배포 패키지 압축
echo 배포 패키지 압축 중...
set "ZIP_FILE=release\MabinogiMobileMacro_v%VERSION%.zip"
if exist "%ZIP_FILE%" del "%ZIP_FILE%"

:: PowerShell을 사용한 압축 (Windows 10 이상)
powershell -command "Compress-Archive -Path '%RELEASE_DIR%' -DestinationPath '%ZIP_FILE%' -Force"

if exist "%ZIP_FILE%" (
    echo ✅ 배포 패키지 생성 완료!
    echo.
    echo 파일 위치:
    echo - 폴더: %RELEASE_DIR%
    echo - 압축: %ZIP_FILE%
    echo.
    
    :: 파일 크기 표시
    for %%A in ("%ZIP_FILE%") do (
        set /a size=%%~zA/1024/1024
        echo 압축 파일 크기: !size! MB
    )
    
) else (
    echo ❌ 압축 파일 생성 실패
)

echo.
echo 배포 준비 완료!
pause 