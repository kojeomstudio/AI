# Git Submodules Guide

이 프로젝트에서 사용하는 Git 서브모듈 관리 가이드입니다.

## 서브모듈 목록

### LLM Agent

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| OpenManus | `llm/agent/open-manus` | main | FoundationAgents/OpenManus |
| OpenAgent | `llm/agent/open-agent` | master | - |

### MCP (Model Context Protocol)

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| mcp-use | `llm/mcp/mcp-use` | main | - |
| Serena | `llm/mcp/serena` | main | oraios/serena |
| web-search-mcp | `llm/mcp/web-search-mcp` | main | - |
| playwright-mcp | `llm/mcp/playwright-mcp` | main | - |
| open-webSearch | `llm/mcp/open-webSearch` | main | - |
| tavily-mcp | `llm/mcp/tavily-mcp` | main | - |

### RAG System

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| R2R | `llm/rag/r2r` | main | SciPhi-AI/R2R |
| R2R Dashboard | `llm/rag/r2r-dashboard` | main | - |

### Search & Tools

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| local-deep-research | `llm/search/local-deep-research` | main | - |
| browser-use | `llm/tools/browser-use` | main | - |

### Image Generative

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| ComfyUI | `image-generative/comfyUI` | master | - |
| Stable Diffusion WebUI | `image-generative/stable-diffusion-webui` | master | - |

### Workflow

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| n8n | `workflow/n8n` | master | n8n-io/n8n |

### Web Servers

| 서브모듈 | 경로 | 브랜치 | 업스트림 |
|----------|------|--------|----------|
| Caddy | `web-servers/caddy` | master | caddyserver/caddy |

## 서브모듈 업데이트

### 자동 업데이트 스크립트

프로젝트 루트에 3가지 플랫폼용 업데이트 스크립트가 있습니다:

#### macOS / Linux
```bash
./update_submodules.sh
```

#### Windows PowerShell
```powershell
.\update_submodules.ps1
```

#### Windows CMD
```cmd
update_submodules.bat
```

### 스크립트 동작 방식

1. **브랜치 자동 감지**: 각 서브모듈의 현재 브랜치 또는 기본 브랜치(main/master) 자동 감지
2. **upstream 처리**:
   - upstream 리모트가 있는 경우: `upstream` -> `local` -> `origin` 워크플로우
   - upstream 리모트가 없는 경우: `origin` -> `local` 워크플로우
3. **에러 처리**: 개별 서브모듈 실패 시 로그 기록 후 다음 서브모듈로 진행
4. **자동 커밋**: 모든 서브모듈 업데이트 후 부모 레포지토리에 자동 커밋 및 푸시

### 수동 업데이트

특정 서브모듈만 업데이트하려면:

```bash
# 서브모듈 디렉토리로 이동
cd llm/mcp/serena

# upstream에서 최신 변경사항 가져오기
git fetch upstream
git merge upstream/main

# origin으로 푸시
git push origin main

# 부모 레포지토리로 돌아가서 변경사항 커밋
cd ../../..
git add llm/mcp/serena
git commit -m "chore: Update serena submodule"
git push origin main
```

## upstream 리모트 설정

새로운 upstream을 설정하려면:

```bash
cd <서브모듈 경로>
git remote add upstream https://github.com/<원본-소유자>/<원본-레포>.git
git fetch upstream
```

## 서브모듈 초기화

클론 후 서브모듈을 초기화하려면:

```bash
# 모든 서브모듈 초기화 및 업데이트
git submodule update --init --recursive

# 또는 클론 시 함께 가져오기
git clone --recurse-submodules <repository-url>
```

## 트러블슈팅

### Detached HEAD 상태

서브모듈이 detached HEAD 상태인 경우:

```bash
cd <서브모듈 경로>
git checkout main  # 또는 master
```

### 병합 충돌

upstream 병합 시 충돌이 발생한 경우:

```bash
# 충돌 해결 후
git add <충돌-해결-파일>
git commit -m "Resolve merge conflict"

# 또는 병합 취소
git merge --abort
```

### 로그 확인

업데이트 중 발생한 에러는 `submodule_update.log` 파일에서 확인할 수 있습니다:

```bash
cat submodule_update.log
```
