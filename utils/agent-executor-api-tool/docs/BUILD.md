# Build Instructions

이 문서는 Agent Executor API를 독립 실행 파일로 빌드하는 방법을 설명합니다.

## 빌드 개요

PyInstaller를 사용하여 Python 애플리케이션을 실행 파일로 패키징합니다.

### 빌드 산출물

- **실행 파일**: `agent-executor-api.exe` (Windows) 또는 `agent-executor-api` (Linux/Mac)
- **위치**: `dist/agent-executor-api/`
- **타입**: One-folder mode (실행 파일 + 의존성 폴더)

## 빌드 방법

### Windows

```bash
# 빌드 스크립트 실행
build.bat
```

스크립트는 다음을 자동으로 수행합니다:
1. Python 설치 확인
2. 가상환경 확인/생성
3. 의존성 설치
4. 이전 빌드 정리
5. PyInstaller로 실행 파일 생성
6. 설정 파일 복사

### Linux/Mac

```bash
# 실행 권한 부여
chmod +x build.sh

# 빌드 스크립트 실행
./build.sh
```

## 빌드 결과 확인

빌드가 성공하면 다음과 같은 구조가 생성됩니다:

```
dist/
└── agent-executor-api/
    ├── agent-executor-api(.exe)  # 실행 파일
    ├── .env                        # 기본 설정 파일
    ├── .env.example                # 설정 예제
    └── [여러 .dll/.so 파일들]      # 의존성 라이브러리
```

## 빌드 실행 및 테스트

### Windows

```cmd
cd dist\agent-executor-api
agent-executor-api.exe
```

### Linux/Mac

```bash
cd dist/agent-executor-api
./agent-executor-api
```

서버가 시작되면:
- API: http://localhost:8000
- 문서: http://localhost:8000/docs

## 배포

빌드된 `dist/agent-executor-api` 폴더 전체를 대상 시스템에 복사합니다.

### 요구사항

대상 시스템에는 **Python이 설치되지 않아도 됩니다**. 모든 의존성이 포함되어 있습니다.

### 주의사항

1. **OS 호환성**: Windows에서 빌드한 실행 파일은 Windows에서만, Linux에서 빌드한 것은 Linux에서만 실행됩니다.
2. **설정 파일**: `.env` 파일을 수정하여 서버 설정을 변경할 수 있습니다.
3. **코딩 에이전트**: 실행할 에이전트(claude-code, codex 등)는 대상 시스템에 설치되어 있어야 합니다.

## 빌드 설정 수정

### PyInstaller 설정

`agent-executor-api.spec` 파일을 수정하여 빌드 옵션을 변경할 수 있습니다:

```python
# 숨겨진 import 추가
hidden_imports = [
    'your_module',
]

# 제외할 모듈 추가
excludes = [
    'unnecessary_module',
]

# 아이콘 설정 (Windows)
icon='path/to/icon.ico'
```

### 단일 파일 빌드

기본적으로 one-folder 모드를 사용하지만, 단일 파일로 빌드하려면:

1. `agent-executor-api.spec` 파일에서 다음을 변경:

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # 추가
    a.zipfiles,      # 추가
    a.datas,         # 추가
    [],
    name='agent-executor-api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)

# COLLECT 섹션 제거
```

2. 빌드 실행:

```bash
# Windows
build.bat

# Linux/Mac
./build.sh
```

## 빌드 문제 해결

### 빌드 실패

1. **의존성 오류**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

2. **가상환경 오류**:
   ```bash
   # 가상환경 삭제 후 재시도
   rm -rf venv  # Linux/Mac
   rmdir /s /q venv  # Windows
   ```

3. **PyInstaller 오류**:
   ```bash
   pip install --upgrade pyinstaller
   ```

### 실행 파일 크기 줄이기

1. UPX 압축 활성화 (이미 활성화됨):
   ```python
   upx=True
   ```

2. 불필요한 모듈 제외:
   ```python
   excludes=['matplotlib', 'numpy', ...]
   ```

3. 디버그 심볼 제거:
   ```python
   strip=True  # Linux/Mac
   ```

## 고급 옵션

### 디버그 빌드

디버그 정보를 포함하려면 `agent-executor-api.spec`에서:

```python
exe = EXE(
    ...
    debug=True,  # 디버그 모드 활성화
    console=True,
    ...
)
```

### 버전 정보 추가 (Windows)

버전 정보 파일 생성:

```python
# version_info.py
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    # ...
  ),
  # ...
)
```

spec 파일에서:

```python
exe = EXE(
    ...
    version='version_info.py',
    ...
)
```

## CI/CD 통합

### GitHub Actions 예제

```yaml
name: Build

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest, macos-latest]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Build
      run: |
        chmod +x build.sh
        ./build.sh  # or build.bat on Windows

    - name: Upload artifact
      uses: actions/upload-artifact@v2
      with:
        name: agent-executor-api-${{ matrix.os }}
        path: dist/agent-executor-api/
```

## 참고 자료

- [PyInstaller 공식 문서](https://pyinstaller.org/)
- [FastAPI 배포 가이드](https://fastapi.tiangolo.com/deployment/)
- [Uvicorn 설정](https://www.uvicorn.org/)
