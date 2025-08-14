# 마비노기 모바일 매크로 - 개선된 버전

## 개요

기존 마비노기 모바일 매크로를 개선하여 더 안정적이고 유연한 입력 시스템을 구현했습니다. 이 버전은 다양한 입력 메서드를 지원하며, 다른 프로세스에 안정적으로 입력을 전달할 수 있습니다.

## 주요 개선사항

### 1. 다중 입력 메서드 지원
- **pyautogui**: 기존 방식, 윈도우 포커스 후 입력
- **win32api**: Windows API 직접 사용
- **PostMessage**: 윈도우 메시지 비동기 전송
- **SendMessage**: 윈도우 메시지 동기 전송

### 2. 설정 기반 액션 시스템
- JSON 설정 파일로 액션 정의
- 우선순위 규칙 시스템
- 조건부 액션 실행
- 쿨다운 시스템

### 3. 안정성 향상
- 오류 처리 강화 (루프 단위 예외 처리 추가)
- 로깅 시스템 개선
- 시그널 핸들러 추가
- 통계 수집 (액션별 실행 횟수/시간 추적)

## 파일 구조

```
mabinogi-mobile/
├── main_improved.py          # 개선된 메인 매크로
├── input_manager.py          # 입력 관리 시스템
├── action_processor.py       # 액션 처리 시스템
├── test_input.py            # 입력 시스템 테스트
├── config/
│   ├── config.json          # 기본 설정
│   └── action_config.json   # 액션 설정
└── README_IMPROVED.md       # 이 파일
```

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install ultralytics pyautogui pywin32 opencv-python numpy
```

### 2. 기본 실행
```bash
python main_improved.py
```

### 3. 테스트 모드 실행
```bash
python main_improved.py --test
```

### 4. 입력 시스템 테스트
```bash
python test_input.py
```

## 설정 파일 설명

### action_config.json

```json
{
  "input_method": "postmessage",  // 기본 입력 메서드
  "default_delay": 0.5,           // 기본 지연 시간
  "actions": {
    "COAL_VEIN": {
      "type": "key",              // 액션 타입 (key/click)
      "key": "space",             // 키보드 키
      "description": "석탄 광맥 채굴",
      "delay": 1.0,               // 액션별 지연 시간
      "conditions": ["UI_MINING"] // 실행 조건
    }
  },
  "priority_rules": [
    {
      "name": "작업 중 대기",
      "conditions": ["UI_WORKING"],
      "action": "wait",
      "description": "작업 중일 때는 대기"
    }
  ]
}
```

## 입력 메서드 비교

| 메서드 | 장점 | 단점 | 권장 사용 |
|--------|------|------|-----------|
| pyautogui | 간단, 안정적 | 윈도우 포커스 필요 | 일반적인 용도 |
| win32api | 빠름, 직접적 | 복잡함 | 성능이 중요한 경우 |
| PostMessage | 비동기, 안정적 | 좌표 변환 필요 | 권장 |
| SendMessage | 동기식, 확실함 | 블로킹 가능 | 중요한 액션 |

## 사용법

### 1. 기본 사용
```python
from input_manager import InputManager
from action_processor import ActionProcessor

# 입력 매니저 초기화
input_manager = InputManager("Mabinogi Mobile")

# 액션 프로세서 초기화
action_processor = ActionProcessor("config/action_config.json", input_manager)

# 액션 실행
action_processor.process_detected_elements(matched_elements)
```

### 2. 직접 입력 전송
```python
# 클릭
input_manager.click(x, y, method='postmessage')

# 키보드 입력
input_manager.send_key('space', method='postmessage')
```

### 3. 입력 메서드 테스트
```python
# 모든 메서드 테스트
input_manager.test_all_methods(x, y)
```

## 문제 해결

### 1. 윈도우를 찾을 수 없는 경우
- 게임 창이 실행 중인지 확인
- 창 제목이 정확한지 확인 (`config.json`의 `window_title`)

### 2. 입력이 전달되지 않는 경우
- 다른 입력 메서드 시도
- 관리자 권한으로 실행
- 게임 창이 최소화되지 않았는지 확인

### 3. 성능 문제
- `tick_interval` 조정
- 더 빠른 입력 메서드 사용 (win32api)
- 불필요한 액션 비활성화

## 보안 고려사항

1. **관리자 권한**: 일부 입력 메서드는 관리자 권한이 필요할 수 있습니다.
2. **게임 정책**: 게임사의 이용약관을 확인하고 준수하세요.
3. **개인정보**: 로그 파일에 민감한 정보가 포함되지 않도록 주의하세요.

## 개발자 정보

### 확장 방법

1. **새로운 액션 추가**:
   - `action_config.json`에 액션 정의
   - 필요한 경우 `ElementType`에 새 타입 추가

2. **새로운 입력 메서드 추가**:
   - `InputManager`에 새 메서드 구현
   - `action_processor.py`에서 지원 추가

3. **새로운 조건 추가**:
   - `priority_rules`에 새 규칙 추가
   - 조건 로직 구현

### 디버깅

1. **로깅 레벨 조정**: `logger_helper.py`에서 로그 레벨 변경
2. **테스트 모드**: `--test` 플래그로 입력 메서드 테스트
3. **통계 확인**: 프로그램 종료 시 액션 통계 출력

## 라이선스

이 프로젝트는 교육 및 연구 목적으로만 사용하세요. 상업적 사용은 금지됩니다.

## 기여

버그 리포트나 개선 제안은 이슈로 등록해 주세요. 