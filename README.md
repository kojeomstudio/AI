# AI Project

> 개인 프로젝트를 중심으로 다양한 인공지능 분야를 실험하고, 게임 개발 및 서브컬처를 AI 기술과 융합하는 데 초점을 맞추고 있습니다.

## Overview

- RAG 시스템 구축 및 운영
- LLM 모델 튜닝 및 추가 학습
- AI 에이전트 및 MCP 서버 통합
- 이미지 생성 AI (Stable Diffusion, ComfyUI)
- 게임 개발과 AI 기술 융합
- 워크플로우 자동화 (n8n)

## Project Structure

```
AI/
├── llm/                     # LLM 관련 프로젝트
│   ├── agent/               # AI 에이전트 (OpenManus, OpenAgent)
│   ├── mcp/                 # MCP 서버들 (Serena, Playwright 등)
│   ├── rag/                 # RAG 시스템 (R2R)
│   ├── search/              # 검색 도구
│   └── tools/               # LLM 도구 (browser-use)
├── image-generative/        # 이미지 생성 AI
│   ├── comfyUI/             # ComfyUI
│   └── stable-diffusion-webui/
├── workflow/                # 워크플로우 자동화
│   └── n8n/                 # n8n
├── game-engine/             # 게임 엔진
├── game-server/             # 게임 서버
├── reinforcement_learning/  # 강화학습
├── study/                   # 학습 및 연구
└── docs/                    # 프로젝트 문서
```

## Submodules

이 프로젝트는 다양한 오픈소스 프로젝트를 서브모듈로 포함합니다.

| Category | Submodules |
|----------|------------|
| LLM Agent | OpenManus, OpenAgent |
| MCP | Serena, mcp-use, playwright-mcp, tavily-mcp, web-search-mcp, open-webSearch |
| RAG | R2R, R2R-Dashboard |
| Search/Tools | local-deep-research, browser-use |
| Image | ComfyUI, Stable Diffusion WebUI |
| Workflow | n8n |

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:kojeomstudio/AI.git
cd AI

# Update all submodules from upstream
./update_submodules.sh        # macOS/Linux
.\update_submodules.ps1       # Windows PowerShell
```

## Documentation

자세한 문서는 [docs/](./docs/) 폴더를 참조하세요.

- [Getting Started](./docs/GETTING_STARTED.md) - 시작 가이드
- [Project Structure](./docs/PROJECT_STRUCTURE.md) - 프로젝트 구조
- [Submodules Guide](./docs/SUBMODULES.md) - 서브모듈 관리

## External Links

- [Docker Hub](https://hub.docker.com/repositories/kojeomstudio)
- [Hugging Face](https://huggingface.co/kojeomstudio)