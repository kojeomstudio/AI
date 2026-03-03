# AI Project Playground

> 개인 프로젝트를 중심으로 다양한 인공지능 분야를 실험하고, 게임 개발 및 서브컬처를 AI 기술과 융합하는 데 초점을 맞추고 있습니다.

## 🚀 Overview

이 리포지토리는 기초적인 머신러닝 학습부터 최신 LLM 에이전트, RAG 시스템, 그리고 게임 자동화 봇에 이르기까지 폭넓은 AI 기술 스택을 다루는 실험실입니다.

- **LLM Ecosystem**: RAG (Retrieval-Augmented Generation), AI 에이전트, MCP 서버 통합 및 모델 튜닝.
- **Generative AI**: Stable Diffusion 및 ComfyUI를 활용한 이미지 생성 워크플로우.
- **Game & AI**: YOLO 기반 게임 봇 (마비노기 모바일) 및 게임 데이터 분석.
- **Automation**: n8n을 이용한 워크플로우 자동화 및 브라우저 기반 AI 도구.
- **Research & Study**: Scikit-learn, TensorFlow, PyTorch를 활용한 딥러닝/머신러닝 기초 연구.

---

## 📂 Project Structure

### 1. LLM & Agents (`/llm`)
- **Agent**: `OpenManus`, `OpenAgent`, `OpenClaw`, `Nanobot` 등 자율형 AI 에이전트.
- **MCP (Model Context Protocol)**: `Serena`, `mcp-use`, `playwright-mcp`, `tavily-mcp` 등 도구 연동 서버.
- **RAG**: `R2R (SciPhi)` 엔진 기반의 지식 기반 검색 및 대화 시스템.
- **Search & Tools**: `local-deep-research`, `browser-use`를 활용한 실시간 정보 수집 및 브라우저 제어.
- **Model**: 임베딩 모델 파인튜닝 (`embedding`) 및 패키지 사이즈 예측 모델 (`package_size_predict`).

### 2. Generative AI (`/image-generative`)
- **ComfyUI**: 노드 기반의 정교한 이미지 생성 워크플로우 관리.
- **Stable Diffusion WebUI**: 직관적인 인터페이스를 통한 이미지 생성 및 편집.

### 3. Game Automation (`/bot`, `/game`)
- **Mabinogi Mobile Bot**: YOLOv8(Ultralytics)을 활용한 객체 인식 및 `pywin32` 기반 입력 자동화.
- **Diablo 2 Utilities**: `CascLib`를 활용한 게임 데이터 라이브러리 연동.

### 4. Machine Learning Study (`/study`, `/reinforcement_learning`)
- **ML/DL Foundations**: 분류, 회귀, CNN, RNN, Attention 등 기초 알고리즘 구현.
- **Reinforcement Learning**: `Gridworld`, `Cartpole` 및 강화학습 기초 실험.

### 5. Tools & Infrastructure (`/tools`, `/web-servers`)
- **Agent API**: `agent-executor-api-tool` (FastAPI 기반 에이전트 프록시).
- **Image Tools**: 배경 제거 (`remove_bg`), 스프라이트 분할 (`sprite`), 이미지 리사이저.
- **Utils**: WSL 포트 포워딩, CRLF-LF 변환 등 개발 편의 도구.
- **Web Server**: `Caddy`를 활용한 리버스 프록시 및 서버 구성.

---

## 🔗 Submodules

이 프로젝트는 최신 오픈소스의 성능을 유지하기 위해 서브모듈을 적극 활용합니다.

| Category | Submodules |
|----------|------------|
| **LLM Agents** | OpenManus, OpenAgent, OpenClaw, Nanobot |
| **MCP Servers** | Serena, mcp-use, playwright-mcp, tavily-mcp, web-search-mcp, noapi-google-search-mcp, open-webSearch |
| **RAG System** | R2R, R2R-Dashboard |
| **Search/Tools** | local-deep-research, browser-use |
| **Image Generation** | ComfyUI, Stable Diffusion WebUI |
| **Workflow** | n8n |
| **Game Libs** | CascLib (Diablo 2) |
| **Web Server** | Caddy |

---

## 🛠 Quick Start

### Repository 복제 및 서브모듈 초기화
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:kojeomstudio/AI.git
cd AI

# 서브모듈 최신 상태로 업데이트
# macOS/Linux
./update_submodules.sh
# Windows PowerShell
.\update_submodules.ps1
```

### 가상 환경 구성
```bash
# 가상 환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

---

## 📝 Documentation

프로젝트별 상세 가이드는 `docs/` 폴더 또는 각 프로젝트의 `README.md`를 참조하세요.

- [시작 가이드 (Getting Started)](./docs/GETTING_STARTED.md)
- [프로젝트 구조 (Project Structure)](./docs/PROJECT_STRUCTURE.md)
- [서브모듈 관리 가이드 (Submodules Guide)](./docs/SUBMODULES.md)

---

## 🌐 External Links

- **Docker Hub**: [kojeomstudio Repositories](https://hub.docker.com/repositories/kojeomstudio)
- **Hugging Face**: [kojeomstudio Models](https://huggingface.co/kojeomstudio)
