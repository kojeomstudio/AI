# 아트 파이프라인 & Godot 플러그인

> 2D 스프라이트/이미지 생성과 Godot 임포트 전략. 이 레포가 이미 가진 AI 생성 자산을 적극 재사용한다.

## 핵심 원칙: 생성은 기존 자산, 임포트는 검증된 플러그인

이 레포에는 이미 강력한 이미지 생성 스택이 있다 (`image-generative/comfyUI`, `image-generative/stable-diffusion-webui`)
— **에디터 내 AI 플러그인보다 이쪽을 생성 백엔드로 쓰는 게 품질·제어·비용 모두 유리**하다.
Godot 플러그인은 "생성"이 아니라 "임포트/애니메이션"에 집중해서 채택한다.

```
[ComfyUI / SD WebUI]  →  [remove_bg + 스프라이트 분할]  →  [PNG 시트]  →  [Godot 임포트 플러그인]  →  AnimatedSprite2D
   (기존, 로컬 무료)        (기존 tools/image/*)            (규격화)        (Aseprite Wizard 등)
```

## 1) 생성 — 기존 ComfyUI / SD WebUI 재사용

- 캐릭터/몬스터/타일셋을 픽셀아트 또는 2D 스타일 모델·LoRA로 생성. 둘 다 HTTP API 제공 → 배치 스크립트로 자동화 가능.
- 후처리는 레포 기존 도구 활용: 배경 제거 `tools/image/remove_bg`, 스프라이트 분할(README에 언급된 sprite 도구).
- 산출 규격을 **고정 그리드 PNG 스프라이트 시트**(예: 한 프레임 N×N, 행=애니메이션, 열=프레임)로 표준화 → 임포트가 단순해진다.

(선택) 에디터 안에서 빠른 컨셉만 뽑고 싶을 때: **AI Horde Client**(크라우드소싱 SD, 무료) 또는 **Sprite Pipeline**(OpenAI 키) 같은 인-에디터 플러그인도 있으나, 제어/품질이 낮아 보조용으로만.

## 2) 임포트 — 채택 후보 (아트 워크플로우에 따라 택1)

| 플러그인 | 적합한 경우 | 비고 |
|----------|-------------|------|
| **Aseprite Wizard** (viniciusgerevini) | Aseprite로 직접 그리거나 애니메이션 편집 | 가장 성숙·인기. 태그→애니메이션, 레이어 regex 필터, ms→FPS 변환, AnimatedSprite2D/SpriteFrames/AnimationPlayer 자동 생성 |
| **Importality** (nklbdev) | 여러 도구 혼용 (Aseprite/Krita/Piskel/Pixelorama) | 멀티 포맷 임포터 번들 |
| 고정 그리드 PNG 시트 + 소형 EditorImportPlugin | AI 생성 PNG 시트가 주력이고 Aseprite를 안 씀 | 의존성 최소, 우리가 규격 통제 |

**권장 기본값**: 캐릭터/몬스터 애니메이션은 **Aseprite Wizard**(아트 작업을 Aseprite로 한다면), AI 생성 시트가 주력이면 **고정 그리드 + 임포트 스크립트**. 타일맵은 Godot 내장 TileMapLayer.

## 3) 나중에 볼 RPG 보조 플러그인 (지금은 보류)

- **Dialogue Manager** (Nathan Hoad) — NPC 대화/퀘스트 (마비노기는 대화 비중이 큼)
- **LimboAI** 또는 **Beehave** — 몬스터/NPC 행동 트리 (전투 증분 때)
- **Phantom Camera** — 2D 추적 카메라 부드럽게

## 결정 대기

- 아트를 **Aseprite로 작업하는지** 여부 → 임포터 선택(Aseprite Wizard vs 그리드 시트 파이프라인)이 갈림.
- Godot 클라이언트 스캐폴딩 시 선택한 임포터를 `addons/`에 넣고, ComfyUI/SD → 시트 → 임포트 예시 1개를 함께 구성한다.
