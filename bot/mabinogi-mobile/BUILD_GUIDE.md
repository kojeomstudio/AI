# 마비노기 모바일 매크로 - 빌드 가이드

## 개요

이 문서는 마비노기 모바일 매크로를 PyInstaller를 사용하여 실행 파일로 패키징하는 방법을 설명합니다.

## 사전 요구사항

### 1. Python 환경
- Python 3.8 이상 설치
- pip 패키지 관리자

### 2. 필요한 패키지
- PyInstaller
- ml/requirements.txt에 명시된 모든 패키지

## 빌드 방법

### 방법 1: 자동 빌드 (권장)

#### 1.1 완전 자동 빌드
```bash
# 가장 완전한 빌드 스크립트
build_package.bat
```
- Python 환경 확인
- 가상환경 생성 (선택사항)
- 의존성 자동 설치
- PyInstaller 실행
- 결과 확인

#### 1.2 간단 빌드
```bash
# 빠른 빌드
build_simple.bat
```
- 기본적인 빌드만 수행
- 빠른 실행

#### 1.3 Spec 파일 빌드
```bash
# spec 파일 사용 빌드
build_with_spec.bat
```
- 미리 정의된 설정 사용
- 안정적인 빌드

### 방법 2: 수동 빌드

#### 2.1 환경 준비
```bash
# PyInstaller 설치
pip install pyinstaller

# 의존성 설치
pip install -r ml/requirements.txt
```

#### 2.2 기본 빌드
```bash
pyinstaller --onefile --windowed --name "MabinogiMacro" main_improved.py
```

#### 2.3 완전한 빌드
```bash
pyinstaller --onefile --windowed --name "MabinogiMobileMacro" \
    --add-data "config;config" \
    --add-data "ml/training_output;ml/training_output" \
    --add-data "assets;assets" \
    --hidden-import "ultralytics" \
    --hidden-import "cv2" \
    --hidden-import "numpy" \
    --hidden-import "pyautogui" \
    --hidden-import "win32gui" \
    --hidden-import "win32con" \
    --hidden-import "win32api" \
    main_improved.py
```

#### 2.4 Spec 파일 사용
```bash
pyinstaller mabinogi_macro.spec
```

## 배포 패키지 생성

### 자동 배포 패키지 생성
```bash
create_release.bat
```
- 버전 정보 입력
- 실행 파일 빌드
- 배포 폴더 생성
- ZIP 압축 파일 생성

### 수동 배포 패키지 생성
```bash
# 1. 빌드 실행
pyinstaller mabinogi_macro.spec

# 2. 배포 폴더 생성
mkdir release
mkdir release\MabinogiMobileMacro_v1.0.0

# 3. 파일 복사
xcopy dist\* release\MabinogiMobileMacro_v1.0.0\ /E /I /Y

# 4. 압축 (PowerShell 사용)
powershell -command "Compress-Archive -Path 'release\MabinogiMobileMacro_v1.0.0' -DestinationPath 'release\MabinogiMobileMacro_v1.0.0.zip' -Force"
```

## 빌드 옵션 설명

### PyInstaller 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--onefile` | 단일 실행 파일로 생성 |
| `--windowed` | 콘솔 창 없이 GUI 모드로 실행 |
| `--name` | 실행 파일 이름 지정 |
| `--add-data` | 데이터 파일 포함 |
| `--hidden-import` | 숨겨진 모듈 명시적 포함 |
| `--exclude-module` | 불필요한 모듈 제외 |
| `--icon` | 실행 파일 아이콘 설정 |

### 파일 크기 최적화

#### 제외할 모듈
```bash
--exclude-module "matplotlib" \
--exclude-module "seaborn" \
--exclude-module "pandas" \
--exclude-module "scipy" \
--exclude-module "sympy" \
--exclude-module "networkx" \
--exclude-module "jupyter" \
--exclude-module "IPython"
```

#### UPX 압축 사용
```bash
--upx-dir "path/to/upx"
```

## 문제 해결

### 1. 빌드 실패

#### 모듈을 찾을 수 없는 경우
```bash
# 숨겨진 임포트 추가
--hidden-import "모듈명"
```

#### 데이터 파일 누락
```bash
# 데이터 파일 추가
--add-data "소스경로;대상경로"
```

### 2. 실행 파일 크기 문제

#### 파일이 너무 큰 경우
- 불필요한 모듈 제외
- UPX 압축 사용
- 데이터 파일 최적화

#### 파일이 너무 작은 경우
- 필요한 모듈이 누락되었을 수 있음
- hidden-import 확인

### 3. 실행 시 오류

#### DLL 오류
- Visual C++ Redistributable 설치
- Windows 업데이트 확인

#### 경로 오류
- 상대 경로 문제 확인
- 데이터 파일 경로 확인

## 빌드 결과

### 성공적인 빌드 후 생성되는 파일
```
dist/
├── MabinogiMobileMacro.exe    # 실행 파일
├── config/                     # 설정 파일
│   ├── config.json
│   └── action_config.json
├── ml/
│   └── training_output/        # YOLO 모델
└── assets/                     # 에셋 파일
```

### 배포 패키지 구조
```
release/
├── MabinogiMobileMacro_v1.0.0/     # 배포 폴더
│   ├── MabinogiMobileMacro.exe
│   ├── config/
│   ├── ml/
│   ├── assets/
│   └── README.txt
└── MabinogiMobileMacro_v1.0.0.zip  # 압축 파일
```

## 성능 최적화

### 1. 빌드 시간 단축
- 가상환경 사용
- 캐시 활용
- 불필요한 모듈 제외

### 2. 실행 파일 크기 최적화
- 필요한 모듈만 포함
- 데이터 파일 압축
- UPX 사용

### 3. 실행 성능 최적화
- 메모리 사용량 최적화
- 시작 시간 단축
- 리소스 사용량 최소화

## 보안 고려사항

### 1. 바이러스 백신 오탐
- PyInstaller로 생성된 파일은 가끔 오탐될 수 있음
- 신뢰할 수 있는 소스에서만 다운로드
- 바이러스 백신 예외 설정

### 2. 코드 서명
- 디지털 서명으로 신뢰성 향상
- 코드 서명 인증서 필요

## 추가 정보

### 유용한 명령어
```bash
# PyInstaller 버전 확인
pyinstaller --version

# 도움말
pyinstaller --help

# 분석 모드 (빌드하지 않고 분석만)
pyinstaller --analysis-only main_improved.py
```

### 디버깅
```bash
# 콘솔 모드로 빌드 (디버깅용)
pyinstaller --onefile --console main_improved.py

# 디버그 정보 포함
pyinstaller --onefile --debug all main_improved.py
```

### 로그 확인
- 빌드 로그는 콘솔에 출력
- 오류 발생 시 상세 로그 확인
- PyInstaller 로그 파일 확인 