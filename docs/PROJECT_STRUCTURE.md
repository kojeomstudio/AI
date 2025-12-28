# Project Structure

AI 프로젝트의 전체 구조를 설명하는 문서입니다.

## 디렉토리 구조

```
AI/
├── automation/          # 자동화 스크립트 및 도구
├── bot/                 # 봇 관련 코드 (마비노기 모바일 등)
├── datas/               # 데이터 파일 저장소
├── documents/           # 기존 문서 (설치 가이드, 논문 등)
├── docs/                # 프로젝트 문서 (신규)
├── game-engine/         # 게임 엔진 관련 프로젝트
├── game-server/         # 게임 서버 관련 프로젝트
├── git-local-server/    # 로컬 Git 서버 설정
├── image-generative/    # 이미지 생성 AI (서브모듈 포함)
├── label/               # 데이터 라벨링 관련
├── llm/                 # LLM 관련 프로젝트 (서브모듈 포함)
├── models/              # 학습된 모델 저장소
├── nexus-repository/    # Nexus 저장소 설정
├── reinforcement_learning/ # 강화학습 관련 코드
├── study/               # 학습 및 연구 코드
├── utils/               # 유틸리티 스크립트
└── workflow/            # 워크플로우 자동화 (n8n 등)
```

## 주요 구성 요소

### LLM (Large Language Model)

```
llm/
├── agent/               # AI 에이전트
│   ├── open-manus/      # OpenManus 에이전트 (서브모듈)
│   └── open-agent/      # OpenAgent (서브모듈)
├── analyzer/            # 분석 도구
├── documents/           # LLM 관련 문서
├── mcp/                 # Model Context Protocol 서버들
│   ├── mcp-use/         # MCP 사용 라이브러리 (서브모듈)
│   ├── serena/          # Serena MCP 서버 (서브모듈)
│   ├── web-search-mcp/  # 웹 검색 MCP (서브모듈)
│   ├── playwright-mcp/  # Playwright MCP (서브모듈)
│   ├── open-webSearch/  # 웹 검색 도구 (서브모듈)
│   └── tavily-mcp/      # Tavily 검색 MCP (서브모듈)
├── model/               # 모델 관련 코드
├── ollama/              # Ollama 관련 설정
├── rag/                 # RAG 시스템
│   ├── r2r/             # R2R RAG 시스템 (서브모듈)
│   └── r2r-dashboard/   # R2R 대시보드 (서브모듈)
├── search/              # 검색 도구
│   └── local-deep-research/ # 로컬 딥 리서치 (서브모듈)
└── tools/               # LLM 도구
    └── browser-use/     # 브라우저 자동화 (서브모듈)
```

### Image Generative

```
image-generative/
├── comfyUI/                    # ComfyUI (서브모듈)
└── stable-diffusion-webui/     # Stable Diffusion WebUI (서브모듈)
```

### Workflow

```
workflow/
├── n8n/                 # n8n 워크플로우 자동화 (서브모듈)
└── stable-diffusion/    # Stable Diffusion 워크플로우
```

## 설정 파일

| 파일 | 설명 |
|------|------|
| `.gitmodules` | Git 서브모듈 설정 |
| `requirements.txt` | Python 의존성 (공통) |
| `requirements_mac.txt` | macOS용 Python 의존성 |
| `requirements_windows.txt` | Windows용 Python 의존성 |
| `update_submodules.sh` | 서브모듈 업데이트 스크립트 (macOS/Linux) |
| `update_submodules.ps1` | 서브모듈 업데이트 스크립트 (PowerShell) |
| `update_submodules.bat` | 서브모듈 업데이트 스크립트 (Windows CMD) |

## 외부 링크

- **Docker Hub**: https://hub.docker.com/repositories/kojeomstudio
- **Hugging Face**: https://huggingface.co/kojeomstudio
