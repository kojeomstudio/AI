# D2R CASC Explorer & Modding Guide

이 문서는 **CASC Viewer** 도구의 구조, 기반 라이브러리인 **CascLib**의 아키텍처, 그리고 실제 **디아블로 2 레저렉션(D2R)**에 모드를 적용하는 방법을 통합하여 설명합니다.

---

## 1. 도구 아키텍처 (Casc Viewer Architecture)

본 프로젝트의 `casc-viewer-wpf`는 D2R의 자산(Asset)을 탐색하고 추출하기 위해 설계된 Windows 데스크톱 애플리케이션입니다.

### 1.1 기술 스택
- **Framework**: .NET 6.0 / WPF (Windows Presentation Foundation)
- **Pattern**: MVVM (Model-View-ViewModel)
- **Interoperability**: P/Invoke를 통한 C++ (CascLib.dll) 연동
- **Storage**: JSON 기반 로컬 설정 캐싱 (SettingsService)

### 1.2 주요 컴포넌트
- **View (MainWindow)**: 트리 뷰(TreeView)를 통해 CASC 내부의 가상 파일 시스템을 시각화합니다.
- **ViewModel (MainViewModel)**: 비동기 작업(Task)을 통해 CASC 스토리지를 열고, 파일 목록을 스캔하며 추출 로직을 제어합니다.
- **Wrapper (CascLibWrapper)**: C++로 작성된 `CascLib.dll`의 함수들을 C#에서 사용할 수 있도록 매핑한 브릿지 클래스입니다.
- **Services**:
    - `LogService`: 실시간 작업 로그 기록 및 파일 저장.
    - `SettingsService`: 마지막 사용 경로 및 검색 마스크 자동 저장/불러오기.

---

## 2. 기반 라이브러리 아키텍처 (CascLib Architecture)

`CascLib`은 블리자드 게임에서 사용하는 **CASC (Content Addressable Storage Container)** 형식을 읽기 위한 오픈 소스 C++ 라이브러리입니다.

### 2.1 CASC의 핵심 개념
CASC는 기존의 MPQ 방식과 달리 파일 이름이 아닌 **콘텐츠의 해시(Hash)** 값을 주소로 사용합니다.
- **Index Files**: 데이터 파일 내의 실제 위치를 가리키는 인덱스 정보를 담고 있습니다.
- **Encoding File**: 파일의 가상 경로(Name)와 실제 데이터의 해시(Content ID)를 연결합니다.
- **Root File**: 게임 클라이언트가 사용하는 논리적인 파일 트리를 정의합니다.

### 2.2 CascLib의 처리 흐름
1.  **Storage Open**: 지정된 경로에서 `build.info` 및 인덱스 파일들을 로드합니다.
2.  **Mounting**: 인코딩 테이블과 루트 핸들러를 구성하여 가상 파일 시스템을 구축합니다.
3.  **File Search**: `CascFindFirstFile` / `CascFindNextFile`을 통해 해시 기반의 데이터를 논리적 경로로 변환하여 탐색합니다.
4.  **Extraction**: 데이터 조각(BLTE chunks)을 찾아 압축을 해제하고 원본 데이터를 복원합니다.

---

## 3. D2R 모딩 적용 가이드 (D2R Modding Guide)

추출한 데이터를 수정하여 실제 게임에 적용하는 방법입니다.

### 3.1 모드 환경 구축
D2R 설치 경로(`Diablo II Resurrected/`)에 다음과 같이 폴더를 생성합니다.
- 경로: `mods/[모드이름]/[모드이름].mpq/data/`
- **주의**: `.mpq`는 실제 파일이 아니라 **폴더 이름의 접미사**입니다.

### 3.2 주요 수정 대상 파일
- **로직/수치**: `data/global/excel/*.json` (또는 `.txt`)
- **외형/그래픽**: `data/hd/items/.../*.json` (에셋 경로 및 모델 설정)
- **텍스트/이름**: `data/local/lng/*.json` (다국어 지원)

### 3.3 실행 및 활성화
1.  Battle.net 런처 -> D2R 설정 -> 게임 설정 -> **명령줄 인수 추가**.
2.  `-mod [모드이름] -txt` 입력. (예: `-mod MyMod -txt`)
3.  게임을 실행하여 오프라인(로컬) 모드에서 적용 여부를 확인합니다.

---

## 4. 모딩 팁 및 주의 사항
- **오프라인 전용**: 모드를 적용한 상태로 배틀넷 멀티플레이어 접속 시 제재를 받을 수 있으므로 반드시 오프라인 캐릭터로 테스트하십시오.
- **백업**: 원본 데이터를 수정하기 전 반드시 백업본을 유지하십시오.
- **대소문자**: 윈도우에서는 구분이 없으나, 가급적 소문자 경로 사용을 권장합니다.

---
*문서 최종 수정일: 2026-03-02*
