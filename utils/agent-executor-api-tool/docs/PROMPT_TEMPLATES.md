# Prompt Template System

Agent Executor API는 코드 리뷰 요청 시 프롬프트 템플릿 시스템을 사용하여 에이전트에 전달되는 프롬프트를 구성합니다.

## 개요

프롬프트 템플릿은 HTTP 요청의 인자들을 조합하여 최종 프롬프트를 생성하는 데 사용됩니다. 이를 통해:
- 일관된 프롬프트 형식 유지
- 에이전트별 최적화된 프롬프트 사용
- 사용자 정의 프롬프트 템플릿 지원

## 디렉토리 구조

```
agent-executor-api/
├── prompts/
│   ├── code_review.md            # 기본 코드 리뷰 템플릿
│   ├── code_review_claude.md     # Claude 전용 템플릿
│   ├── code_review_ue4.md        # UE4 코드 리뷰 템플릿 (한국어)
│   ├── code_review_simple.md     # 간단한 템플릿
│   └── custom_template.example.md  # 사용자 정의 템플릿 예시
├── config.json                   # 템플릿 설정 포함
└── ...
```

## 파일 형식

템플릿 파일은 `.md` (Markdown) 형식을 사용합니다.

## 사용 가능한 플레이스홀더

템플릿에서 다음 플레이스홀더를 사용할 수 있습니다:

| 플레이스홀더 | 설명 | 예시 |
|-------------|------|------|
| `{context}` | 리뷰할 코드/컨텍스트 | `def hello(): print('hi')` |
| `{build_id}` | 빌드 식별자 | `build-12345` |
| `{user_id}` | 알림 대상 사용자 ID | `user-abc123` |
| `{agent_type}` | 사용되는 에이전트 타입 | `claude-code` |
| `{timestamp}` | 현재 타임스탬프 (ISO) | `2025-01-15T10:30:00.123456` |
| `{date}` | 현재 날짜 | `2025-01-15` |
| `{time}` | 현재 시간 | `10:30:00` |

## 템플릿 우선순위

1. **요청에서 제공된 custom_template** (최우선)
2. **에이전트별 템플릿 파일** (`agent_prompt_templates` 설정)
3. **기본 템플릿 파일** (`prompt_template_file` 설정)


## 설정 방법

### config.json 설정

```json
{
    "prompt_template_file": "code_review.md",
    "prompt_templates_dir": "prompts",
    "agent_prompt_templates": {
        "claude-code": "code_review_claude.md",
        "codex": "code_review.md",
        "gemini": "code_review.md"
    }
}
```

### 설정 항목 설명

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `prompt_template_file` | `code_review.md` | 기본 템플릿 파일명 |
| `prompt_templates_dir` | `prompts` | 템플릿 디렉토리 |
| `agent_prompt_templates` | (에이전트별 설정) | 에이전트별 템플릿 파일 매핑 |

## 템플릿 파일 예시

### 기본 템플릿 (code_review.md)

```
You are an expert code reviewer. Please review the following code changes thoroughly.

Build ID: {build_id}
User to notify: {user_id}

=== Code/Context to Review ===
{context}
=== End of Code/Context ===

Please provide detailed feedback on:
1. Code quality and readability
2. Potential bugs or logical errors
3. Security vulnerabilities
4. Performance considerations
5. Best practices and coding standards
6. Suggestions for improvement
```

### Claude 전용 템플릿 (code_review_claude.md)

```
You are an expert code reviewer. Please review the following code changes thoroughly.

Build ID: {build_id}
User to notify: {user_id}

=== Code/Context to Review ===
{context}
=== End of Code/Context ===

Please provide detailed feedback covering:

## Code Quality
- Readability and maintainability
- Naming conventions
- Code structure and organization

## Potential Issues
- Bugs or logical errors
- Edge cases not handled
- Error handling gaps

## Security Analysis
- SQL injection vulnerabilities
- XSS risks
- Command injection possibilities

## Performance
- Algorithm efficiency
- Memory usage concerns

## Recommendations
- Specific improvements with code examples
- Best practices to follow
```

## API에서 커스텀 템플릿 사용

### JSON 요청에서 커스텀 템플릿 사용

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "def add(a, b): return a + b",
    "build_id": "build-123",
    "user_id": "dev-team",
    "custom_template": "프로젝트: MyProject\n빌드: {build_id}\n담당자: {user_id}\n\n다음 코드를 검토해주세요:\n{context}\n\n버그와 보안 이슈에 집중해주세요."
  }'
```

### 추가 플레이스홀더 사용 (extra_template_args)

```bash
curl -X POST http://localhost:8000/review/json \
  -H "Content-Type: application/json" \
  -d '{
    "context": "def add(a, b): return a + b",
    "build_id": "build-123",
    "user_id": "dev-team",
    "custom_template": "프로젝트: {project_name}\n리뷰어: {reviewer}\n빌드: {build_id}\n\n{context}",
    "extra_template_args": {
      "project_name": "MyAwesomeProject",
      "reviewer": "Senior Developer Kim"
    }
  }'
```

### Form-Data에서 커스텀 템플릿 사용

```bash
curl -X POST http://localhost:8000/review/form \
  -F "context=def add(a, b): return a + b" \
  -F "build_id=build-123" \
  -F "user_id=dev-team" \
  -F "custom_template=빌드 {build_id}의 코드를 검토해주세요:\n\n{context}"
```

### Query Parameter에서 커스텀 템플릿 사용

```bash
curl -X POST "http://localhost:8000/review/raw?build_id=build-123&user_id=dev-team&custom_template=Review+for+{user_id}:%0A{context}" \
  -H "Content-Type: text/plain" \
  -d 'def add(a, b): return a + b'
```

## 사용자 정의 템플릿 만들기

1. `prompts/` 디렉토리에 새 `.md` 파일 생성
2. 플레이스홀더 사용하여 템플릿 작성
3. `config.json`에서 템플릿 파일 지정

### 예시: 보안 중심 템플릿

`prompts/security_review.md`:
```
=== SECURITY CODE REVIEW ===
Date: {date} {time}
Build: {build_id}
Reviewer: {user_id}

Focus Areas:
- Input validation
- SQL injection
- XSS vulnerabilities
- Authentication/Authorization
- Data encryption
- Sensitive data exposure

CODE TO REVIEW:
{context}

Please identify ALL security vulnerabilities and provide:
1. Severity level (Critical/High/Medium/Low)
2. Description of the vulnerability
3. Recommended fix with code example
```

`config.json`에 추가:
```json
{
    "agent_prompt_templates": {
        "claude-code": "security_review.md"
    }
}
```

## 프로그래밍 언어별 예제

### Python

```python
import requests

# 커스텀 템플릿으로 코드 리뷰 요청
response = requests.post(
    "http://localhost:8000/review/json",
    json={
        "context": code_to_review,
        "build_id": "build-123",
        "user_id": "python-dev",
        "custom_template": """
{project_name} 코드 리뷰 요청
================================
빌드: {build_id}
날짜: {date}
담당자: {user_id}

코드:
{context}

다음 항목을 검토해주세요:
1. Python best practices
2. Type hints
3. 예외 처리
""",
        "extra_template_args": {
            "project_name": "MyPythonProject"
        }
    }
)
```

### JavaScript

```javascript
const reviewWithCustomTemplate = async (code) => {
  const response = await fetch('http://localhost:8000/review/json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      context: code,
      build_id: 'frontend-build-001',
      user_id: 'js-team',
      custom_template: `
Frontend Code Review
====================
Build: {build_id}
Team: {user_id}
Date: {timestamp}

Code:
{context}

Please check:
- React best practices
- Performance optimizations
- Accessibility issues
`
    })
  });
  return response.json();
};
```

## 문제 해결

### 템플릿 파일을 찾을 수 없음

로그 확인:
```
[CONFIG] Template not found: my_template (tried: ['D:\\path\\to\\prompts\\my_template.md', 'D:\\path\\to\\prompts\\my_template.txt'])
```

해결 방법:
1. 파일 경로 확인
2. 파일명 철자 확인
3. `prompt_templates_dir` 설정 확인

### 플레이스홀더가 치환되지 않음

원인: 템플릿에 잘못된 플레이스홀더 사용

로그 확인:
```
[CONFIG] Warning: Missing placeholder in template: 'unknown_placeholder'
```

해결 방법:
1. 지원되는 플레이스홀더 목록 확인
2. `extra_template_args`로 커스텀 플레이스홀더 제공

### 인코딩 문제

템플릿 파일은 반드시 **UTF-8** 인코딩이어야 합니다.

Windows에서 새 파일 생성 시:
```powershell
# PowerShell
Set-Content -Path "prompts\my_template.md" -Value "템플릿 내용" -Encoding UTF8
```
