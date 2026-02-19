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
├── web-servers/         # 웹 서버 (Caddy 등)
└── workflow/            # 워크플로우 자동화 (n8n 등)
```

## 프로젝트 개요

이 프로젝트는 AI/ML 기술을 활용한 다양한 애플리케이션과 도구를 포함하는 통합 AI 개발 플랫폼입니다. 주요 구성 요소는 다음과 같습니다:

- **LLM 및 RAG 시스템**: 대규모 언어 모델 및 검색 증강 생성 시스템
- **이미지 생성 AI**: ComfyUI, Stable Diffusion 등 이미지 생성 도구
- **AI 에이전트**: 다양한 자동화 에이전트 (OpenManus, OpenAgent, OpenClaw, NanoBot 등)
- **MCP (Model Context Protocol)**: 다양한 MCP 서버 및 도구
- **게임 자동화**: 마비노기 모바일 게임 자동화 봇
- **워크플로우 자동화**: n8n 기반 워크플로우 자동화
- **학습 및 연구**: 다양한 ML/DL 알고리즘 및 연구 코드

## 주요 구성 요소

### LLM (Large Language Model)

```
llm/
├── agent/               # AI 에이전트
│   ├── open-manus/      # OpenManus 에이전트 (서브모듈)
│   ├── open-agent/      # OpenAgent (서브모듈)
│   ├── openclaw/        # OpenClaw 에이전트 (서브모듈)
│   └── nanobot/         # NanoBot 에이전트 (서브모듈)
├── analyzer/            # 분석 도구
├── documents/           # LLM 관련 문서
├── mcp/                 # Model Context Protocol 서버들
│   ├── mcp-use/         # MCP 사용 라이브러리 (서브모듈)
│   ├── serena/          # Serena MCP 서버 (서브모듈)
│   ├── web-search-mcp/  # 웹 검색 MCP (서브모듈)
│   ├── playwright-mcp/  # Playwright MCP (서브모듈)
│   ├── open-webSearch/  # 웹 검색 도구 (서브모듈)
│   ├── tavily-mcp/      # Tavily 검색 MCP (서브모듈)
│   └── noapi-google-search-mcp/ # Google 검색 MCP (서브모듈)
├── model/               # 모델 관련 코드
├── ollama/              # Ollama 관련 설정
├── rag/                 # RAG 시스템
│   ├── r2r/             # R2R RAG 시스템 (서브모듈)
│   ├── r2r-client/      # R2R 클라이언트
│   ├── r2r-dashboard/   # R2R 대시보드 (서브모듈)
│   ├── r2r_framework/   # R2R 프레임워크
│   ├── local_rag/       # 로컬 RAG 구현
│   ├── document_convertor/ # 문서 변환 도구
│   └── util/            # RAG 유틸리티
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
├── n8n_sample/          # n8n 샘플 워크플로우
├── n8n_templates/       # n8n 템플릿 모음
├── docker/              # Docker Compose 설정
│   └── n8n/             # n8n Docker 설정
├── comfyui/             # ComfyUI 워크플로우
├── stable-diffusion/    # Stable Diffusion 워크플로우
└── CICD/                # CI/CD 설정
```

### Web Servers

```
web-servers/
└── caddy/               # Caddy 웹 서버 (서브모듈)
```

### Bot (Game Automation)

```
bot/
└── mabinogi-mobile/     # 마비노기 모바일 게임 자동화 봇
    ├── app.py           # 메인 애플리케이션
    ├── action_processor.py  # 액션 처리기
    ├── config_manager.py    # 설정 관리자
    ├── input_manager.py     # 입력 관리자
    ├── ocr_helper.py        # OCR 헬퍼
    ├── logger_helper.py     # 로그 헬퍼
    ├── ui/               # UI 요소
    ├── utils/            # 유틸리티
    ├── ml/               # 머신러닝 모델
    ├── assets/           # 리소스 파일
    ├── config/           # 설정 파일
    └── tests/            # 테스트 파일
```

### Study & Research

```
study/
├── algorithm/           # 알고리즘 연습
├── llama_llm_sample_scripts/  # LLM 샘플 스크립트
├── pytorch/             # PyTorch 예제
├── simple_game/         # 간단한 게임 예제
├── classification_dctree_iris.py  # 의사결정 트리 분류
├── classification_heart_data.py   # 심장 데이터 분류
├── classification_mnist_mlp_keras.py  # MNIST MLP 분류
├── classification_mnist_sklearn.py    # MNIST sklearn 분류
├── classification_wine.py   # 와인 데이터 분류
├── clustering_fruits.py   # 과일 클러스터링
├── cnn_mnist_sample.py   # CNN MNIST 샘플
├── huggingface_model_downloader.py  # Hugging Face 모델 다운로더
├── mlp_regressor.py      # MLP 회귀
├── perceptron.py         # 퍼셉트론
├── regression_housing_mlp.py   # 주택 가격 회귀
├── regression_medicalcost.py   # 의료비 회귀
├── regression_wide_and_deep_housing.py  # Wide & Deep 회귀
├── reinforcement_cartpole.py  # 강화학습 CartPole
├── simple_bidir_rnn_with_attention.py  # 양방향 RNN + 어텐션
├── simple_nn_translator_eng_to_spanish.py  # 번역기
├── simple_rnn_imdb.py    # RNN IMDB
└── simple_rnn_shakespeare.py  # RNN 셰익스피어
```

### Reinforcement Learning

```
reinforcement_learning/
└── my_first_rl.py        # 첫 번째 강화학습 예제
```

## 설정 파일

| 파일 | 설명 |
|------|------|
| `.gitmodules` | Git 서브모듈 설정 |
| `.gitattributes` | Git 속성 설정 |
| `.gitignore` | Git 무시 파일 설정 |
| `requirements.txt` | Python 의존성 (공통) |
| `requirements_mac.txt` | macOS용 Python 의존성 |
| `requirements_windows.txt` | Windows용 Python 의존성 |
| `update_submodules.sh` | 서브모듈 업데이트 스크립트 (macOS/Linux) |
| `update_submodules.ps1` | 서브모듈 업데이트 스크립트 (PowerShell) |
| `update_submodules.bat` | 서브모듈 업데이트 스크립트 (Windows CMD) |
| `change_env_var.ps1` | 환경 변수 변경 스크립트 |
| `start_my_ai_venv.command` | AI 가상환경 시작 스크립트 (macOS) |
| `start_my_r2r_venv.command` | R2R 가상환경 시작 스크립트 (macOS) |
| `GEMINI.md` | Gemini 관련 문서 |
| `LICENSE` | 라이선스 파일 |
| `submodule_update.log` | 서브모듈 업데이트 로그 |
| `processed_files.db` | 처리된 파일 데이터베이스 |
| `my_shakespeare_model.keras` | 셰익스피어 모델 파일 |
| `scikit_learn_ml_map.png` | Scikit-learn ML 맵 이미지 |

## 외부 링크

- **Docker Hub**: https://hub.docker.com/repositories/kojeomstudio
- **Hugging Face**: https://huggingface.co/kojeomstudio
