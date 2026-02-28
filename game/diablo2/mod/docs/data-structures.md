# 데이터 구조 및 주요 파일 (Data Structures)

D2R 모딩에서 다루는 주요 파일 형식과 그 역할입니다.

## 1. `.txt` 파일 (Excel Data)
게임의 밸런스, 아이템, 스킬, 드랍률 등을 결정하는 핵심 파일입니다. 대부분 `data/global/excel` 폴더에 위치합니다.

*   **inventory.txt:** 인벤토리 크기 및 배치 설정.
*   **itemstatcost.txt:** 아이템 능력치 정의.
*   **treasureclassex.txt:** 아이템 드랍 테이블 (TC).
*   **skills.txt:** 기술의 효과 및 데미지 수치.
*   **monstats.txt:** 몬스터 능력치.
*   **runes.txt:** 룬워드 조합법.

## 2. `.json` 파일
레저렉션에서 도입된 현대적인 설정 파일입니다. 주로 UI 레이아웃, 폰트, 그리고 일부 엔진 설정을 담당합니다.

*   `data/hd/ui`: UI 레이아웃 및 디자인 설정.
*   아이템의 3D 모델 연결 설정 등도 JSON으로 관리됩니다.

## 3. `.json` (Localization)
텍스트 데이터는 `data/local/lng` 폴더의 JSON 파일에 저장됩니다. 아이템 이름, 설명, 대사 등을 수정할 때 사용합니다.

## 4. 그래픽 파일
*   **.sprite:** 2D 스프라이트 데이터 (아이콘, 버튼 등).
*   **.texture:** 3D 텍스처 데이터.
*   **.model:** 3D 모델 데이터.

## 5. `.d2s` (Save Files)
싱글 플레이어 캐릭터 세이브 파일입니다. `Saved Games\Diablo II Resurrected` 폴더에 위치합니다.
