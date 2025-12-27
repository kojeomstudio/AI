# Agent Executor API - 상세 사용 가이드

이 문서는 Agent Executor API의 모든 엔드포인트에 대한 상세한 사용 예시를 제공합니다.

## 목차

1. [헬스 체크 엔드포인트](#1-헬스-체크-엔드포인트)
2. [에이전트 목록 조회](#2-에이전트-목록-조회)
3. [설정 조회](#3-설정-조회)
4. [일반 에이전트 실행 (/execute)](#4-일반-에이전트-실행-execute)
5. [코드 리뷰 - JSON 형식 (/review/json)](#5-코드-리뷰---json-형식-reviewjson)
6. [코드 리뷰 - Form-Data 형식 (/review/form)](#6-코드-리뷰---form-data-형식-reviewform)
7. [코드 리뷰 - Raw Text 형식 (/review/raw)](#7-코드-리뷰---raw-text-형식-reviewraw)
8. [에러 응답 처리](#8-에러-응답-처리)
9. [프로그래밍 언어별 예제](#9-프로그래밍-언어별-예제)

---

## 1. 헬스 체크 엔드포인트

서버 상태를 확인합니다.

### GET /

```bash
curl http://localhost:8000/
```

### GET /health

```bash
curl http://localhost:8000/health
```

### 응답

```json
{
  "status": "healthy",
  "version": "1.1.0",
  "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

## 2. 에이전트 목록 조회

지원되는 코딩 에이전트 목록을 조회합니다.

### GET /agents

```bash
curl http://localhost:8000/agents
```

### 응답

```json
{
  "supported_agents": [
    "claude-code",
    "codex",
    "gemini",
    "cursor",
    "aider"
  ],
  "description": "List of supported coding agent types"
}
```

---

## 3. 설정 조회

현재 서버 설정을 조회합니다 (민감하지 않은 정보만).

### GET /config

```bash
curl http://localhost:8000/config
```

### 응답

```json
{
  "config_directory": "D:\\ai-repo\\llm\\tools\\agent-executor-api",
  "config_json_loaded": true,
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "default_timeout": 300,
  "max_timeout": 1800,
  "agent_commands": {
    "claude-code": "claude",
    "codex": "codex",
    "gemini": "gemini",
    "cursor": "cursor",
    "aider": "aider"
  }
}
```

---

## 4. 일반 에이전트 실행 (/execute)

코딩 에이전트를 직접 실행합니다.

### POST /execute

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `agent_type` | string | ✅ | - | 실행할 에이전트 타입 |
| `command_args` | string | ✅ | - | 에이전트에 전달할 명령줄 인자 |
| `timeout` | integer | ❌ | 300 | 실행 타임아웃 (초) |
| `working_directory` | string | ❌ | null | 작업 디렉토리 경로 |

### 예시 1: 기본 실행 (필수 파라미터만)

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "claude-code",
    "command_args": "-p \"Create a hello world function in Python\""
  }'
```

### 예시 2: 타임아웃 설정

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "claude-code",
    "command_args": "-p \"Refactor this codebase to use async/await\"",
    "timeout": 600
  }'
```

### 예시 3: 작업 디렉토리 지정

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "claude-code",
    "command_args": "-p \"Fix the bug in main.py\"",
    "timeout": 300,
    "working_directory": "D:\\projects\\my-python-project"
  }'
```

### 예시 4: 모든 옵션 사용

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "codex",
    "command_args": "generate --language python --task \"implement binary search\"",
    "timeout": 120,
    "working_directory": "D:\\projects\\algorithms"
  }'
```

### 예시 5: Aider 에이전트 사용

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "aider",
    "command_args": "--message \"Add unit tests for the Calculator class\" --yes",
    "timeout": 600,
    "working_directory": "D:\\projects\\calculator"
  }'
```

### 성공 응답

```json
{
  "success": true,
  "output": "def hello_world():\n    print(\"Hello, World!\")\n\nif __name__ == \"__main__\":\n    hello_world()",
  "error": null,
  "exit_code": 0,
  "execution_time": 5.23
}
```

### 실패 응답 (에이전트 실행 실패)

```json
{
  "success": false,
  "output": "",
  "error": "Agent executable not found: claude. Make sure claude-code is installed and in PATH.",
  "exit_code": -1,
  "execution_time": 0.01
}
```

---

## 5. 코드 리뷰 - JSON 형식 (/review/json)

JSON body로 코드 리뷰를 요청합니다. 가장 일반적인 사용 방식입니다.

### POST /review/json

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `context` | string | ✅ | - | 코드 리뷰 컨텍스트 (코드, diff, 지시사항 등) |
| `build_id` | string | ✅ | - | 빌드 식별자 (추적용) |
| `user_id` | string | ✅ | - | 알림을 받을 사용자 ID |
| `agent_type` | string | ❌ | "claude-code" | 사용할 에이전트 타입 |
| `timeout` | integer | ❌ | 600 | 실행 타임아웃 (초) |
| `working_directory` | string | ❌ | null | 작업 디렉토리 경로 |
| `additional_args` | string | ❌ | null | 추가 명령줄 인자 |

### 예시 1: 기본 코드 리뷰 (필수 파라미터만)

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "다음 코드를 리뷰해주세요:\n\ndef add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b",
    "build_id": "build-12345",
    "user_id": "user-abc123"
  }'
```

### 예시 2: Git Diff 리뷰

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "다음 Git diff를 리뷰해주세요:\n\n--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1,5 +1,8 @@\n class Calculator:\n     def add(self, a, b):\n-        return a + b\n+        \"\"\"Add two numbers.\"\"\"\n+        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):\n+            raise TypeError(\"Arguments must be numbers\")\n+        return a + b",
    "build_id": "build-67890",
    "user_id": "developer-kim"
  }'
```

### 예시 3: 에이전트 타입 지정

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Review this TypeScript code for potential issues:\n\nfunction fetchUser(id: string): Promise<User> {\n  return fetch(`/api/users/${id}`).then(r => r.json());\n}",
    "build_id": "frontend-build-001",
    "user_id": "frontend-team",
    "agent_type": "gemini"
  }'
```

### 예시 4: 타임아웃 및 작업 디렉토리 설정

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "대규모 리팩토링 코드를 검토해주세요. 보안 취약점과 성능 이슈에 집중해주세요.\n\n[긴 코드 내용...]",
    "build_id": "refactor-sprint-42",
    "user_id": "security-team",
    "agent_type": "claude-code",
    "timeout": 1200,
    "working_directory": "D:\\projects\\enterprise-app"
  }'
```

### 예시 5: 추가 인자 포함 (모든 옵션 사용)

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Review the following Python module:\n\nimport os\nimport subprocess\n\ndef run_command(cmd):\n    return subprocess.check_output(cmd, shell=True)\n\ndef read_config(path):\n    with open(path) as f:\n        return eval(f.read())",
    "build_id": "security-audit-2025-01",
    "user_id": "security-admin",
    "agent_type": "claude-code",
    "timeout": 900,
    "working_directory": "D:\\projects\\legacy-system",
    "additional_args": "--verbose --no-cache"
  }'
```

### 예시 6: Perforce Changelist 리뷰 (빌드 시스템 연동)

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Perforce CL #123456 변경사항을 리뷰해주세요:\n\n//depot/main/Source/Engine/Core/Math/Vector.h\n@@ -45,6 +45,12 @@\n+inline FVector operator*(const FVector& V, float Scale)\n+{\n+    return FVector(V.X * Scale, V.Y * Scale, V.Z * Scale);\n+}\n",
    "build_id": "p4-cl-123456",
    "user_id": "engine-team-lead",
    "agent_type": "claude-code",
    "timeout": 600
  }'
```

### 성공 응답

```json
{
  "success": true,
  "build_id": "build-12345",
  "user_id": "user-abc123",
  "agent_type": "claude-code",
  "output": "## 코드 리뷰 결과\n\n### 긍정적인 점\n- 함수가 간결하고 명확합니다.\n- 적절한 함수명을 사용했습니다.\n\n### 개선 제안\n1. 타입 힌트 추가를 권장합니다.\n2. 입력 검증이 필요합니다.\n\n### 수정 예시\n```python\ndef add(a: float, b: float) -> float:\n    return a + b\n```",
  "error": null,
  "exit_code": 0,
  "execution_time": 12.45,
  "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

## 6. 코드 리뷰 - Form-Data 형식 (/review/form)

HTML 폼이나 multipart/form-data로 코드 리뷰를 요청합니다. 웹 폼에서 직접 제출할 때 유용합니다.

### POST /review/form

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `context` | string | ✅ | - | 코드 리뷰 컨텍스트 |
| `build_id` | string | ✅ | - | 빌드 식별자 |
| `user_id` | string | ✅ | - | 알림을 받을 사용자 ID |
| `agent_type` | string | ❌ | "claude-code" | 사용할 에이전트 타입 |
| `timeout` | integer | ❌ | 600 | 실행 타임아웃 (초) |
| `working_directory` | string | ❌ | null | 작업 디렉토리 경로 |
| `additional_args` | string | ❌ | null | 추가 명령줄 인자 |

### 예시 1: 기본 Form-Data 요청 (필수 파라미터만)

```bash
curl -X POST http://localhost:8000/review/form \
  -F "context=def hello():\n    print('hello')" \
  -F "build_id=build-12345" \
  -F "user_id=user-abc123"
```

### 예시 2: 에이전트 타입 지정

```bash
curl -X POST http://localhost:8000/review/form \
  -F "context=function greet(name) { return 'Hello, ' + name; }" \
  -F "build_id=js-build-001" \
  -F "user_id=frontend-dev" \
  -F "agent_type=gemini"
```

### 예시 3: 타임아웃 설정

```bash
curl -X POST http://localhost:8000/review/form \
  -F "context=Review this complex algorithm implementation..." \
  -F "build_id=algo-build-42" \
  -F "user_id=algorithm-team" \
  -F "agent_type=claude-code" \
  -F "timeout=900"
```

### 예시 4: 모든 옵션 사용

```bash
curl -X POST http://localhost:8000/review/form \
  -F "context=class UserService:\n    def get_user(self, id):\n        return db.query(f'SELECT * FROM users WHERE id={id}')" \
  -F "build_id=security-scan-001" \
  -F "user_id=security-team" \
  -F "agent_type=claude-code" \
  -F "timeout=600" \
  -F "working_directory=D:\projects\web-app" \
  -F "additional_args=--focus security"
```

### 예시 5: 파일에서 컨텍스트 읽기

```bash
# 먼저 리뷰할 코드를 파일로 준비
echo "def calculate(x, y): return x / y" > /tmp/code_to_review.txt

# Form-Data로 전송 (파일 내용을 context로)
curl -X POST http://localhost:8000/review/form \
  -F "context=<code_to_review.txt" \
  -F "build_id=file-review-001" \
  -F "user_id=reviewer"
```

### 예시 6: Windows에서 파일 내용 전송

```powershell
# PowerShell에서 파일 내용을 Form-Data로 전송
$code = Get-Content -Path "D:\projects\app\main.py" -Raw
$body = @{
    context = $code
    build_id = "ps-build-001"
    user_id = "windows-dev"
    agent_type = "claude-code"
    timeout = 600
}

Invoke-RestMethod -Uri "http://localhost:8000/review/form" -Method Post -Form $body
```

### 성공 응답

```json
{
  "success": true,
  "build_id": "build-12345",
  "user_id": "user-abc123",
  "agent_type": "claude-code",
  "output": "코드 리뷰 결과입니다...",
  "error": null,
  "exit_code": 0,
  "execution_time": 8.92,
  "timestamp": "2025-01-15T10:35:00.654321"
}
```

---

## 7. 코드 리뷰 - Raw Text 형식 (/review/raw)

HTTP body에 순수 텍스트로 컨텍스트를 전송합니다. 다른 파라미터는 쿼리 스트링으로 전달합니다.
파이프라인이나 스크립트에서 stdout을 직접 전송할 때 유용합니다.

### POST /review/raw

#### Query 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `build_id` | string | ✅ | - | 빌드 식별자 |
| `user_id` | string | ✅ | - | 알림을 받을 사용자 ID |
| `agent_type` | string | ❌ | "claude-code" | 사용할 에이전트 타입 |
| `timeout` | integer | ❌ | 600 | 실행 타임아웃 (초) |
| `working_directory` | string | ❌ | null | 작업 디렉토리 경로 |
| `additional_args` | string | ❌ | null | 추가 명령줄 인자 |

#### Request Body

- Content-Type: `text/plain`
- Body: 코드 리뷰 컨텍스트 (UTF-8 텍스트)

### 예시 1: 기본 Raw 요청 (필수 파라미터만)

```bash
curl -X POST "http://localhost:8000/review/raw?build_id=build-12345&user_id=user-abc123" \
  -H "Content-Type: text/plain" \
  -d 'def hello():
    print("hello world")'
```

### 예시 2: 에이전트 타입 지정

```bash
curl -X POST "http://localhost:8000/review/raw?build_id=build-001&user_id=dev-team&agent_type=gemini" \
  -H "Content-Type: text/plain" \
  -d 'const fetchData = async () => {
  const response = await fetch("/api/data");
  return response.json();
};'
```

### 예시 3: 타임아웃 설정

```bash
curl -X POST "http://localhost:8000/review/raw?build_id=complex-review&user_id=senior-dev&timeout=1200" \
  -H "Content-Type: text/plain" \
  -d '매우 긴 코드 내용...'
```

### 예시 4: 모든 옵션 사용

```bash
curl -X POST "http://localhost:8000/review/raw?build_id=full-review&user_id=team-lead&agent_type=claude-code&timeout=900&working_directory=D%3A%5Cprojects%5Capp&additional_args=--verbose" \
  -H "Content-Type: text/plain" \
  -d 'class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def connect(self):
        # TODO: implement connection logic
        pass'
```

### 예시 5: 파이프라인에서 Git diff 전송

```bash
# Git diff 결과를 직접 API로 전송
git diff HEAD~1 | curl -X POST "http://localhost:8000/review/raw?build_id=git-diff-review&user_id=git-user" \
  -H "Content-Type: text/plain" \
  --data-binary @-
```

### 예시 6: 파일 내용을 Raw로 전송

```bash
# 파일 내용을 그대로 body로 전송
curl -X POST "http://localhost:8000/review/raw?build_id=file-review&user_id=file-user&agent_type=claude-code" \
  -H "Content-Type: text/plain" \
  --data-binary "@D:/projects/app/main.py"
```

### 예시 7: 여러 파일 결합하여 전송

```bash
# 여러 파일을 결합하여 전송
(echo "=== file1.py ===" && cat file1.py && echo -e "\n=== file2.py ===" && cat file2.py) | \
curl -X POST "http://localhost:8000/review/raw?build_id=multi-file&user_id=reviewer" \
  -H "Content-Type: text/plain" \
  --data-binary @-
```

### 예시 8: n8n 워크플로우에서 사용

n8n HTTP Request 노드 설정:
- Method: POST
- URL: `http://localhost:8000/review/raw?build_id={{$node["Build"].json["id"]}}&user_id={{$node["User"].json["id"]}}`
- Body Content Type: Raw
- Body: `{{$node["CodeDiff"].json["diff"]}}`

### 성공 응답

```json
{
  "success": true,
  "build_id": "build-12345",
  "user_id": "user-abc123",
  "agent_type": "claude-code",
  "output": "## 코드 리뷰 결과\n\n리뷰 내용...",
  "error": null,
  "exit_code": 0,
  "execution_time": 10.12,
  "timestamp": "2025-01-15T10:40:00.789012"
}
```

---

## 8. 에러 응답 처리

### 에러 응답 형식

모든 에러는 일관된 JSON 형식으로 반환됩니다:

```json
{
  "success": false,
  "error_code": "ERROR_CODE",
  "error_message": "사람이 읽을 수 있는 에러 메시지",
  "detail": "추가 상세 정보 (선택)",
  "timestamp": "2025-01-15T10:30:00.123456",
  "request_id": "abc12345",
  "build_id": "build-12345",
  "user_id": "user-abc123"
}
```

### 에러 코드 목록

| 에러 코드 | HTTP 상태 | 설명 | 해결 방법 |
|-----------|-----------|------|-----------|
| `TIMEOUT_EXCEEDED` | 400 | 요청 타임아웃이 최대값 초과 | `max_timeout` 설정 확인, 더 작은 값 사용 |
| `UNSUPPORTED_AGENT` | 400 | 지원하지 않는 에이전트 타입 | `/agents` 엔드포인트로 지원 목록 확인 |
| `EMPTY_CONTEXT` | 400 | 코드 리뷰 컨텍스트가 비어있음 | context 파라미터에 내용 추가 |
| `INVALID_ENCODING` | 400 | 요청 본문이 유효한 UTF-8이 아님 | UTF-8로 인코딩된 텍스트 전송 |
| `VALIDATION_ERROR` | 400 | 요청 데이터 검증 실패 | 요청 파라미터 형식 확인 |
| `COMMAND_NOT_FOUND` | 400 | 에이전트 실행 파일을 찾을 수 없음 | 에이전트 설치 및 PATH 확인 |
| `INVALID_WORKING_DIRECTORY` | 400 | 작업 디렉토리가 존재하지 않음 | 경로 확인 후 재시도 |
| `EXECUTION_TIMEOUT` | 400 | 실행 중 타임아웃 발생 | 더 긴 타임아웃 설정 |
| `PERMISSION_DENIED` | 400 | 실행 권한 없음 | 파일 권한 확인 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 | 서버 로그 확인, 관리자 문의 |

### 에러 처리 예시

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/review/json",
    json={
        "context": "review this code",
        "build_id": "build-123",
        "user_id": "user-456"
    }
)

data = response.json()

if not data.get("success", True):
    error_code = data.get("error_code", "UNKNOWN")
    error_message = data.get("error_message", "Unknown error")
    detail = data.get("detail", "")

    print(f"Error [{error_code}]: {error_message}")
    if detail:
        print(f"Detail: {detail}")

    # 에러 코드별 처리
    if error_code == "TIMEOUT_EXCEEDED":
        print("Tip: Try reducing the timeout value")
    elif error_code == "UNSUPPORTED_AGENT":
        print("Tip: Check supported agents at /agents endpoint")
else:
    print(f"Success! Output: {data['output']}")
```

#### JavaScript

```javascript
async function reviewCode(context, buildId, userId) {
  try {
    const response = await fetch('http://localhost:8000/review/json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        context,
        build_id: buildId,
        user_id: userId
      })
    });

    const data = await response.json();

    if (!data.success) {
      console.error(`Error [${data.error_code}]: ${data.error_message}`);
      if (data.detail) {
        console.error(`Detail: ${data.detail}`);
      }
      return null;
    }

    return data.output;
  } catch (error) {
    console.error('Network error:', error);
    return null;
  }
}
```

---

## 9. 프로그래밍 언어별 예제

### Python (requests 라이브러리)

```python
import requests
from typing import Optional, Dict, Any

class AgentExecutorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def list_agents(self) -> list:
        """지원 에이전트 목록"""
        response = requests.get(f"{self.base_url}/agents")
        return response.json()["supported_agents"]

    def execute(
        self,
        agent_type: str,
        command_args: str,
        timeout: int = 300,
        working_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """에이전트 직접 실행"""
        payload = {
            "agent_type": agent_type,
            "command_args": command_args,
            "timeout": timeout
        }
        if working_directory:
            payload["working_directory"] = working_directory

        response = requests.post(
            f"{self.base_url}/execute",
            json=payload
        )
        return response.json()

    def review_code(
        self,
        context: str,
        build_id: str,
        user_id: str,
        agent_type: str = "claude-code",
        timeout: int = 600,
        working_directory: Optional[str] = None,
        additional_args: Optional[str] = None
    ) -> Dict[str, Any]:
        """코드 리뷰 요청"""
        payload = {
            "context": context,
            "build_id": build_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "timeout": timeout
        }
        if working_directory:
            payload["working_directory"] = working_directory
        if additional_args:
            payload["additional_args"] = additional_args

        response = requests.post(
            f"{self.base_url}/review/json",
            json=payload
        )
        return response.json()


# 사용 예시
if __name__ == "__main__":
    client = AgentExecutorClient()

    # 헬스 체크
    print(client.health_check())

    # 코드 리뷰
    result = client.review_code(
        context="def add(a, b): return a + b",
        build_id="python-example-001",
        user_id="developer",
        timeout=300
    )

    if result["success"]:
        print(f"리뷰 결과:\n{result['output']}")
    else:
        print(f"에러: {result['error_message']}")
```

### JavaScript/TypeScript (fetch API)

```typescript
interface ReviewResult {
  success: boolean;
  build_id: string;
  user_id: string;
  agent_type: string;
  output: string;
  error: string | null;
  exit_code: number;
  execution_time: number;
  timestamp: string;
}

interface ReviewOptions {
  agentType?: string;
  timeout?: number;
  workingDirectory?: string;
  additionalArgs?: string;
}

class AgentExecutorClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async healthCheck(): Promise<object> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async listAgents(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/agents`);
    const data = await response.json();
    return data.supported_agents;
  }

  async reviewCode(
    context: string,
    buildId: string,
    userId: string,
    options: ReviewOptions = {}
  ): Promise<ReviewResult> {
    const payload = {
      context,
      build_id: buildId,
      user_id: userId,
      agent_type: options.agentType || 'claude-code',
      timeout: options.timeout || 600,
      ...(options.workingDirectory && { working_directory: options.workingDirectory }),
      ...(options.additionalArgs && { additional_args: options.additionalArgs })
    };

    const response = await fetch(`${this.baseUrl}/review/json`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    return response.json();
  }
}

// 사용 예시
async function main() {
  const client = new AgentExecutorClient();

  // 헬스 체크
  console.log(await client.healthCheck());

  // 코드 리뷰
  const result = await client.reviewCode(
    'function add(a, b) { return a + b; }',
    'js-example-001',
    'frontend-dev',
    { timeout: 300 }
  );

  if (result.success) {
    console.log('리뷰 결과:', result.output);
  } else {
    console.error('에러:', result.error);
  }
}

main();
```

### PowerShell

```powershell
# Agent Executor API 클라이언트 함수들

function Get-AgentHealth {
    param([string]$BaseUrl = "http://localhost:8000")

    $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
    return $response
}

function Get-SupportedAgents {
    param([string]$BaseUrl = "http://localhost:8000")

    $response = Invoke-RestMethod -Uri "$BaseUrl/agents" -Method Get
    return $response.supported_agents
}

function Invoke-CodeReview {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Context,

        [Parameter(Mandatory=$true)]
        [string]$BuildId,

        [Parameter(Mandatory=$true)]
        [string]$UserId,

        [string]$AgentType = "claude-code",
        [int]$Timeout = 600,
        [string]$WorkingDirectory = $null,
        [string]$AdditionalArgs = $null,
        [string]$BaseUrl = "http://localhost:8000"
    )

    $body = @{
        context = $Context
        build_id = $BuildId
        user_id = $UserId
        agent_type = $AgentType
        timeout = $Timeout
    }

    if ($WorkingDirectory) {
        $body.working_directory = $WorkingDirectory
    }

    if ($AdditionalArgs) {
        $body.additional_args = $AdditionalArgs
    }

    $jsonBody = $body | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "$BaseUrl/review/json" `
        -Method Post `
        -ContentType "application/json" `
        -Body $jsonBody

    return $response
}

# 사용 예시
$health = Get-AgentHealth
Write-Host "Server Status: $($health.status)"

$agents = Get-SupportedAgents
Write-Host "Supported Agents: $($agents -join ', ')"

$code = @"
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total
"@

$result = Invoke-CodeReview -Context $code -BuildId "ps-build-001" -UserId "admin"

if ($result.success) {
    Write-Host "리뷰 결과:"
    Write-Host $result.output
} else {
    Write-Host "에러: $($result.error_message)" -ForegroundColor Red
}
```

### cURL 스크립트 (Bash)

```bash
#!/bin/bash

# Agent Executor API 클라이언트 스크립트

BASE_URL="${AGENT_API_URL:-http://localhost:8000}"

# 헬스 체크
health_check() {
    curl -s "$BASE_URL/health" | jq .
}

# 에이전트 목록
list_agents() {
    curl -s "$BASE_URL/agents" | jq -r '.supported_agents[]'
}

# 코드 리뷰 (JSON)
review_code() {
    local context="$1"
    local build_id="$2"
    local user_id="$3"
    local agent_type="${4:-claude-code}"
    local timeout="${5:-600}"

    curl -s -X POST "$BASE_URL/review/json" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg ctx "$context" \
            --arg bid "$build_id" \
            --arg uid "$user_id" \
            --arg agent "$agent_type" \
            --argjson timeout "$timeout" \
            '{
                context: $ctx,
                build_id: $bid,
                user_id: $uid,
                agent_type: $agent,
                timeout: $timeout
            }'
        )" | jq .
}

# 파일 기반 코드 리뷰
review_file() {
    local file_path="$1"
    local build_id="$2"
    local user_id="$3"

    if [ ! -f "$file_path" ]; then
        echo "Error: File not found: $file_path"
        return 1
    fi

    local content=$(cat "$file_path")
    review_code "$content" "$build_id" "$user_id"
}

# Git diff 리뷰
review_git_diff() {
    local build_id="$1"
    local user_id="$2"
    local commit_range="${3:-HEAD~1..HEAD}"

    local diff=$(git diff "$commit_range")

    if [ -z "$diff" ]; then
        echo "No changes found in $commit_range"
        return 1
    fi

    review_code "$diff" "$build_id" "$user_id"
}

# 사용 예시
case "$1" in
    health)
        health_check
        ;;
    agents)
        list_agents
        ;;
    review)
        review_code "$2" "$3" "$4" "$5" "$6"
        ;;
    review-file)
        review_file "$2" "$3" "$4"
        ;;
    review-diff)
        review_git_diff "$2" "$3" "$4"
        ;;
    *)
        echo "Usage: $0 {health|agents|review|review-file|review-diff}"
        echo ""
        echo "Commands:"
        echo "  health                          - Check server health"
        echo "  agents                          - List supported agents"
        echo "  review <context> <build_id> <user_id> [agent_type] [timeout]"
        echo "  review-file <file_path> <build_id> <user_id>"
        echo "  review-diff <build_id> <user_id> [commit_range]"
        exit 1
        ;;
esac
```

---

## 부록: 설정 참조

### config.json 전체 옵션

```json
{
    "app_name": "Agent Executor API",
    "app_version": "1.1.0",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,

    "default_timeout": 300,
    "max_timeout": 1800,
    "default_working_directory": null,

    "agent_commands": {
        "claude-code": "claude",
        "codex": "codex",
        "gemini": "gemini",
        "cursor": "cursor",
        "aider": "aider"
    },

    "code_review_prompt_template": "Review the following code changes. Build ID: {build_id}. User to notify: {user_id}.\n\nContext:\n{context}",

    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    "log_to_file": false,
    "log_file_path": "agent_executor.log",
    "log_max_bytes": 10485760,
    "log_backup_count": 5,

    "allow_origins": "*",
    "allow_credentials": true,
    "allow_methods": "*",
    "allow_headers": "*"
}
```

### 환경 변수로 설정 덮어쓰기

```bash
# 서버 설정
export HOST=127.0.0.1
export PORT=9000
export LOG_LEVEL=DEBUG

# 타임아웃 설정
export DEFAULT_TIMEOUT=600
export MAX_TIMEOUT=3600

# CORS 설정
export ALLOW_ORIGINS=http://localhost:3000,https://myapp.com
```
