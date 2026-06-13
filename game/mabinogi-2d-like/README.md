# mabinogi-2d-like

2D 미니멀 MMO (마비노기 스타일). 올-C# 스택: Godot 4(.NET) 클라이언트 + ASP.NET Core 권위 서버.

설계·기술 검토: [`docs/technical-review.md`](docs/technical-review.md)

## 구조

```
mabinogi-2d-like/
├── Mabinogi2D.slnx              # .NET 솔루션 (서버 + 공유 라이브러리)
├── shared/Mabinogi2D.Shared/    # 프로토콜 DTO/상수 (서버·클라 공유, MessagePack)
├── server/Mabinogi2D.Server/    # 단일 서버: 인증 REST + JWT + 게임 WebSocket + 20Hz 틱 루프
│   ├── Data/                    #   EF Core 엔티티/DbContext
│   ├── Auth/                    #   JWT, 로그인/캐릭터 REST
│   └── Game/                    #   GameWorld(권위 시뮬), 틱 루프, WebSocket
├── client/                      # (예정) Godot 4 .NET 프로젝트
├── tools/                       # 정적 데이터 변환 등
└── docker-compose.yml           # Postgres (lite, 선택)
```

서버를 두 프로세스(인증/게임)로 나누지 않고 **단일 프로세스**로 둔 것은 초반 테스트 단순화를 위한 의도적 선택. 규모가 커지면 분리한다.

## 요구 사항

- .NET SDK 8+ (현재 머신: .NET 10)
- Godot 4 **.NET 에디션** — `C:\workspaces\Godot_v4.6.3-stable_mono_win64\`
  (소스 빌드 `C:\workspaces\godot`는 C# glue 생성이 hang되므로 사용하지 않음)
- Docker — Postgres로 전환할 때만 필요 (기본은 SQLite라 불필요)

## 데이터베이스

개발 기본값은 **SQLite**(Docker 불필요, 즉시 실행). 설정 한 줄로 Docker Postgres 전환:

```jsonc
// server/Mabinogi2D.Server/appsettings.json
"Database": { "Provider": "Sqlite" }   // "Postgres"로 바꾸면 docker compose의 DB 사용
```

## 보안 설정 (필독)

- **JWT 서명 키는 소스/설정에 커밋하지 않는다.** 환경변수 `Jwt__Key`(또는 .NET user-secrets)로 32바이트 이상 키를 주입한다.
  - Development: 키를 안 주면 매 실행 임의 키를 자동 생성한다(편의용, 재시작 시 기존 토큰 무효).
  - 비-Development: 키가 없으면 **기동 실패**(fail-closed). 토큰 위조 = 인증 우회이므로 안전한 기본값을 두지 않는다.
  ```bash
  # 예시 (운영/공유 환경)
  export Jwt__Key="$(openssl rand -base64 48)"
  ```
- **배포 시 TLS 필수.** 현재 REST/WebSocket은 `http`/`ws`(평문)다. 외부 노출 시 `https`/`wss`로 토큰 가로채기를 막을 것.

## 실행

```bash
# 서버 (SQLite로 바로 기동, 스키마 자동 생성)
dotnet run --project server/Mabinogi2D.Server

# Postgres로 쓰려면
docker compose up -d           # DB 컨테이너
# 그리고 appsettings.json의 Provider를 "Postgres"로
```

REST 빠른 확인:

```bash
curl -X POST localhost:<port>/auth/register -H "Content-Type: application/json" -d '{"username":"hero","password":"secret"}'
curl -X POST localhost:<port>/auth/login    -H "Content-Type: application/json" -d '{"username":"hero","password":"secret"}'
# 반환된 token으로 Authorization: Bearer <token> 헤더를 붙여 /characters 호출
```

빌드 산출물은 레포 규칙대로 `Bins/mabinogi-2d-like-*/`로 출력한다 (`AGENTS.md` 참고).

## 진행 상태

- [x] 솔루션 골격 + `Mabinogi2D.Shared` 프로토콜(접속/이동/스냅샷/전투/채팅)
- [x] `docker-compose.yml` (Postgres, lite)
- [x] EF Core 엔티티/DbContext (SQLite 기본 ↔ Postgres 전환)
- [x] 인증 REST + JWT (회원가입/로그인/캐릭터) — 스모크 테스트 통과
- [x] 단일 서버 프로세스: 20Hz 틱 루프 + WebSocket(`/ws`), 이동 동기화 서버측
- [x] Godot 클라이언트(`client/`) — 로그인→WS 입장→스냅샷 수신/보간/입력 전송. 빌드·헤드리스 임포트·실서버 입장까지 검증
- [x] 통합 실행 스크립트 `run-dev.ps1` (서버 + 클라 N개, 인자로 캐릭터 분리) — 메커니즘 검증 완료
- [x] 몬스터 + 기본 전투 — 고정 몬스터 스폰, 근접 공격(사거리·쿨다운·데미지), 처치 시 exp + 부활. 봇으로 처치·부활 검증
- [ ] 멀티플레이 육안 확인 (`.\run-dev.ps1` 실행해 두 창에서 이동/전투 확인)
- [ ] 그리드 스프라이트 시트 임포터 + ComfyUI/SD 생성 파이프라인 (`docs/asset-pipeline.md`)

조작: 화살표=이동, Space=가장 가까운 몬스터 공격. 클라 인자 `--bot`을 주면 자동으로 몬스터를 사냥(테스트/부하용).

## 통합 실행 (권장)

서버 + 클라이언트 N개를 한 번에 빌드·실행하고, 창을 닫으면 서버까지 정리한다:

```powershell
.\run-dev.ps1                 # 서버 + 클라 2개 (hero1/hero2, 서로 다른 캐릭터)
.\run-dev.ps1 -Clients 3      # 클라 3개
.\run-dev.ps1 -NoBuild        # 빌드 생략
.\run-dev.ps1 -Godot "D:\Godot\Godot.exe"   # Godot 경로 지정 (또는 GODOT_BIN 환경변수)
```

클라이언트는 화살표키로 이동. 각 인스턴스는 `-- --user/--char/--server` 인자로 서로 다른 캐릭터로 입장한다.

## 수동 실행 (개별)

```powershell
# 서버
dotnet run --project server/Mabinogi2D.Server        # http://localhost:5080
# 클라이언트(게임 모드, 다른 캐릭터로)
& "C:\workspaces\Godot_v4.6.3-stable_mono_win64\Godot_v4.6.3-stable_mono_win64.exe" `
    --path client -- --user hero2 --char Hero2 --server http://localhost:5080
# 에디터로 열려면: ... --editor --path client
```
