# Getting Started

AI 프로젝트 시작 가이드입니다.

## 요구사항

- Git (서브모듈 지원)
- Python 3.8+
- Docker (선택사항)

## 설치

### 1. 저장소 클론

```bash
# 서브모듈 포함하여 클론
git clone --recurse-submodules git@github.com:kojeomstudio/AI.git
cd AI

# 이미 클론한 경우 서브모듈 초기화
git submodule update --init --recursive
```

### 2. Python 가상환경 설정

#### macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt
```

#### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements_windows.txt
```

### 3. 서브모듈 업데이트

최신 upstream 변경사항을 가져오려면:

```bash
# macOS / Linux
./update_submodules.sh

# Windows PowerShell
.\update_submodules.ps1

# Windows CMD
update_submodules.bat
```

## 주요 컴포넌트 시작하기

### R2R RAG System

```bash
cd llm/rag/r2r
# R2R 문서 참조

# Docker Compose로 실행
cd docker
docker-compose up -d
```

### R2R Client

```bash
cd llm/rag/r2r-client
# 클라이언트 설정 및 실행
```

### ComfyUI

```bash
cd image-generative/comfyUI
# ComfyUI 실행 스크립트 참조
```

### n8n Workflow

```bash
cd workflow/n8n
# n8n 설정 문서 참조

# Docker Compose로 실행
cd docker
docker-compose up -d
```

### OpenManus Agent

```bash
cd llm/agent/open-manus
# OpenManus 설정 및 실행

# Docker Compose로 실행
cd dev
docker-compose up -d
```

### 마비노기 모바일 봇

```bash
cd bot/mabinogi-mobile
# 필요한 패키지 설치
pip install -r requirements.txt

# 설정 파일 수정
# config/config.json

# 봇 실행
python app.py
```

## 환경 변수

필요한 환경 변수는 `change_env_var.ps1` 스크립트를 참조하세요.

## 문서

- [프로젝트 구조](./PROJECT_STRUCTURE.md)
- [서브모듈 관리](./SUBMODULES.md)
- [LLM 설정](../documents/llama_settings.txt)

## 추가 리소스

- [Docker Hub](https://hub.docker.com/repositories/kojeomstudio)
- [Hugging Face](https://huggingface.co/kojeomstudio)
