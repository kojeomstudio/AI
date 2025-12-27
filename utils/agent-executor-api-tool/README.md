# Agent Executor API

FastAPI 기반 코딩 에이전트 실행 프록시 서버입니다. 로컬에 설치된 상용 코딩 에이전트(Claude Code, Codex, Gemini 등)를 HTTP API를 통해 실행할 수 있습니다.

## 특징

- FastAPI 기반 고성능 웹 서버
- 코딩 에이전트를 터미널 프로세스로 실행
- **다양한 입력 형식 지원**: JSON, form-data, raw text
- **코드 리뷰 전용 엔드포인트**: context, build_id, user_id 파라미터
- **프롬프트 템플릿 시스템**: 에이전트별 커스텀 프롬프트 지원
- **Read-Only 모드**: 코드 리뷰 시 파일 수정 방지
- **상세한 콘솔 로깅**: 요청 추적 ID, 타임스탬프, 실행 상태
- **상세한 에러 응답**: 에러 코드, 에러 메시지, 상세 정보
- **config.json 설정 파일**: 환경변수와 함께 사용 가능
- 실행 타임아웃 설정 가능
- 작업 디렉토리 지정 가능
- 실행 결과 및 에러 캡처
- CORS 지원

## 지원하는 에이전트

| 에이전트 | 식별자 | CLI 명령어 | 비고 |
|----------|--------|-----------|------|
| Claude Code | `claude-code` | `claude -p "prompt"` | 비대화형 print 모드 |
| OpenAI Codex | `codex` | `codex exec "prompt"` | 비대화형 exec 모드 |
| Google Gemini | `gemini` | `gemini -p "prompt"` | 비대화형 모드 |
| Cursor | `cursor` | `cursor "prompt"` | 기본 프롬프트 모드 |
| Aider | `aider` | `aider --message "prompt"` | 비대화형 메시지 모드 |

## 설치

### 1. 가상환경 생성 (선택사항)

```bash
python -m venv venv

# Windows
venv\Scriptsctivate

# Linux/Mac
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 설정

```bash
cp config.json.example config.json
```

설정을 확인하려면:

```bash
python check_config.py
```

## 실행

### 방법 1: 직접 실행

```bash
python -m app.main
```

### 방법 2: Uvicorn 사용

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9999 --reload
```

### 방법 3: 실행 스크립트 사용

```bash
# Windows
run.bat

# Linux/Mac
chmod +x run.sh && ./run.sh
```

서버가 시작되면 다음 주소에서 접근할 수 있습니다:
- API 서버: http://localhost:9999
- API 문서: http://localhost:9999/docs
- ReDoc: http://localhost:9999/redoc

## 독립 실행 파일 빌드

Python 설치 없이 실행 가능한 바이너리 파일을 생성할 수 있습니다.

```bash
# Windows
build.bat

# Linux/Mac
chmod +x build.sh && ./build.sh
```

### 빌드 결과

```
dist/agent-executor-api/
├── agent-executor-api(.exe)  # 실행 파일
├── config.json               # JSON 설정 파일
├── prompts/                  # 프롬프트 템플릿 폴더
└── [의존성 라이브러리들]
```

**자세한 빌드 옵션과 문제 해결은 [docs/BUILD.md](docs/BUILD.md)를 참조하세요.**

## API 사용법

### 1. 헬스 체크

```bash
curl http://localhost:9999/health
```

### 2. 지원하는 에이전트 목록 조회

```bash
curl http://localhost:9999/agents
```

### 3. 에이전트 실행

```bash
curl -X POST http://localhost:9999/execute   -H "Content-Type: application/json"   -d '{
    "agent_type": "claude-code",
    "command_args": "-p \"Create a hello world function\"",
    "timeout": 300
  }'
```

### 4. 코드 리뷰 (JSON 형식)

```bash
curl -X POST http://localhost:9999/review/json   -H "Content-Type: application/json"   -d '{
    "agent_type": "claude-code",
    "context": "def hello():
    print(\"hello\")",
    "build_id": "build-12345",
    "user_id": "user-abc123"
  }'
```

**자세한 API 사용법은 [docs/API_USAGE.md](docs/API_USAGE.md)를 참조하세요.**

## 프롬프트 템플릿 시스템

코드 리뷰 요청 시 프롬프트 템플릿을 사용하여 에이전트에 전달되는 프롬프트를 구성합니다.

### 템플릿 파일 위치

```
prompts/
├── code_review.md            # 기본 코드 리뷰 템플릿
├── code_review_claude.md     # Claude 전용 템플릿
├── code_review_ue4.md        # UE4 코드 리뷰 템플릿 (한국어)
└── code_review_simple.md     # 간단한 템플릿
```

> 템플릿 파일은 `.md` (Markdown) 형식을 사용합니다.

### Read-Only 모드

모든 프롬프트 템플릿에는 **Read-Only 지시사항**이 포함되어 있습니다:
- 코딩 에이전트가 파일을 수정하지 않도록 강력히 지시
- 리뷰 결과는 텍스트 출력으로만 제공
- 코드 수정 제안은 응답 내 예시 코드로만 표시

**자세한 템플릿 설정은 [docs/PROMPT_TEMPLATES.md](docs/PROMPT_TEMPLATES.md)를 참조하세요.**

## 환경 설정

설정은 `config.json` 파일을 통해 관리됩니다. 필요한 경우 환경변수로 오버라이드할 수 있습니다.

### 설정 우선순위

1. 환경 변수 (시스템 레벨) - 최우선
2. `config.json` 파일
3. 코드의 기본값

### 주요 설정 항목 (config.json)

```json
{
    "app_name": "Agent Executor API",
    "app_version": "1.2.0",
    "host": "0.0.0.0",
    "port": 9999,
    "default_timeout": 900,
    "max_timeout": 3600,
    "agent_commands": {
        "claude-code": "claude",
        "codex": "codex",
        "gemini": "gemini"
    },
    "prompt_template_file": "code_review.md",
    "agent_prompt_templates": {
        "claude-code": "code_review_claude.md"
    },
    "log_level": "INFO"
}
```

## 프로젝트 구조

```
agent-executor-api/
├── app/                         # 애플리케이션 코드
│   ├── main.py                  # FastAPI 애플리케이션
│   ├── models.py                # Pydantic 모델
│   ├── executor.py              # 에이전트 실행 로직
│   └── config.py                # 설정 관리
├── prompts/                     # 프롬프트 템플릿 폴더
├── docs/                        # 상세 문서
│   ├── BUILD.md
│   ├── API_USAGE.md
│   └── PROMPT_TEMPLATES.md
├── config.json                  # JSON 설정 파일
├── config.json.example          # 설정 파일 예제
├── requirements.txt             # Python 의존성
├── build.bat / build.sh         # 빌드 스크립트
└── run.bat / run.sh             # 실행 스크립트
```

## 보안 고려사항

1. **명령어 주입 방지**: `shlex.split()`을 사용하여 안전하게 인자를 파싱합니다.
2. **Shell 비활성화**: `subprocess.run()`에서 `shell=False`로 설정하여 쉘 인젝션을 방지합니다.
3. **타임아웃 설정**: 무한 실행을 방지하기 위해 타임아웃을 설정합니다.
4. **Read-Only 코드 리뷰**: 프롬프트에 읽기 전용 지시사항을 포함하여 파일 수정을 방지합니다.

## 에러 코드

| 에러 코드 | 설명 |
|-----------|------|
| `TIMEOUT_EXCEEDED` | 요청된 타임아웃이 최대값을 초과 |
| `UNSUPPORTED_AGENT` | 지원하지 않는 에이전트 타입 |
| `EMPTY_CONTEXT` | 코드 리뷰 컨텍스트가 비어있음 |
| `COMMAND_NOT_FOUND` | 에이전트 실행 파일을 찾을 수 없음 |
| `EXECUTION_TIMEOUT` | 실행 중 타임아웃 발생 |
| `PERMISSION_DENIED` | 실행 권한 없음 |

## 라이선스

MIT

## 기여

이슈와 풀 리퀘스트를 환영합니다!
