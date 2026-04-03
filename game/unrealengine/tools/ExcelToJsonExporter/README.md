# Excel to JSON Exporter

엑셀(.xlsx) 데이터 시트에 작성된 게임 데이터를 JSON 형태로 추출하는 도구입니다.
C# .NET 8.0 WPF 기반으로 작성되었으며, GUI 모드와 CLI 모드를 모두 지원합니다.

## 개요

| 항목 | 설명 |
|------|------|
| 런타임 | .NET 8.0 (Windows) |
| UI 프레임워크 | WPF |
| 엑셀 라이브러리 | EPPlus 7.x |
| JSON 라이브러리 | Newtonsoft.Json 13.x |
| 출력 형식 | JSON (UTF-8, Indented) |

## 동작 원리

### 1. 입력 형식

엑셀 파일의 각 시트(Sheet)를 하나의 데이터 테이블로 처리합니다.

**헤더 행 (1행)**: `타입:컬럼명` 형식으로 정의합니다.

| int:Id | string:Name | float:Damage | bool:Stackable |
|--------|-------------|--------------|----------------|

타입을 생략하면 기본값인 `string`으로 처리됩니다.

| Name | Level | Description |
|------|-------|-------------|

### 2. 지원 데이터 타입

| 타입 | JSON 출력 | 설명 |
|------|-----------|------|
| `string` | `"text"` | 기본값. 문자열 그대로 출력 |
| `int` / `int32` | `42` | 32비트 정수 |
| `long` / `int64` | `9999999999` | 64비트 정수 |
| `float` / `single` | `3.14` | 단정밀도 부동소수점 |
| `double` | `3.14159265` | 배정밀도 부동소수점 |
| `bool` / `boolean` | `true` / `false` | 불리언 (`1`/`0`도 인식) |

### 3. 데이터 파싱 규칙

- **1행**: 헤더 (`타입:컬럼명`)
- **2행**: 구분자 행 (`---`, `===` 등)이 있으면 자동으로 건너뜀
- **3행~**: 실제 데이터 행 (빈 행은 자동 건너뜀)
- 파싱 실패한 값은 원본 문자열 그대로 JSON에 포함 (데이터 손실 방지)

### 4. 출력 형식

각 시트별로 `{시트명}.json` 파일이 생성됩니다.

```json
[
  {
    "Id": 1,
    "Name": "Short Sword",
    "Damage": 5.5,
    "Stackable": false
  },
  {
    "Id": 2,
    "Name": "Health Potion",
    "Damage": 0.0,
    "Stackable": true
  }
]
```

## 사용법

### GUI 모드 (WPF)

```
dotnet run --project tools/ExcelToJsonExporter
```

1. **엑셀 파일 선택**: "찾아보기..." 버튼으로 `.xlsx` 파일 선택
2. **출력 폴더 지정**: 기본값은 입력 파일과 동일한 디렉토리
3. **시트 미리보기**: 시트 목록에서 시트를 선택하면 데이터를 미리볼 수 있음
4. **내보내기**: "JSON 내보내기" 버튼 또는 `Ctrl+E`로 변환 실행

### CLI 모드

```
ExcelToJsonExporter --cli -i <입력.xlsx> [-o <출력디렉토리>]
```

**옵션:**

| 옵션 | 설명 |
|------|------|
| `-c`, `--cli` | CLI 모드로 실행 |
| `-i`, `--input` | 입력 엑셀 파일 경로 (필수) |
| `-o`, `--output` | 출력 디렉토리 (기본: 입력 파일 위치) |

**사용 예:**

```bash
# 기본 출력 (입력 파일과 같은 위치)
ExcelToJsonExporter --cli -i GameData.xlsx

# 출력 위치 지정
ExcelToJsonExporter --cli -i GameData.xlsx -o ./output

# dotnet run으로 직접 실행
dotnet run --project tools/ExcelToJsonExporter -- --cli -i GameData.xlsx -o ./GameData
```

## 프로젝트 구조

```
ExcelToJsonExporter/
├── ExcelToJsonExporter.csproj   # 프로젝트 설정
├── App.xaml / App.xaml.cs       # WPF 애플리케이션 진입점
├── Program.cs                   # CLI/GUI 듀얼 모드 진입점
├── MainWindow.xaml              # 메인 윈도우 XAML 레이아웃
├── MainWindow.xaml.cs           # 메인 윈도우 코드비하인드
├── ExcelToJsonCore.cs           # CLI용 핵심 변환 로직
├── Models/
│   ├── ColumnDefinition.cs      # 컬럼 정의 (타입 + 이름)
│   ├── SheetPreview.cs          # 시트 미리보기 데이터 모델
│   └── ExportResult.cs          # 내보내기 결과 모델
├── Services/
│   ├── ExcelReader.cs           # 엑셀 파일 읽기 서비스
│   └── JsonExporter.cs          # JSON 변환/출력 서비스
└── ViewModels/
    ├── ViewModelBase.cs         # MVVM 기본 클래스
    ├── MainViewModel.cs         # 메인 화면 뷰모델
    └── RelayCommand.cs          # ICommand 구현
```

## DataGenerator와의 관계

기존 `DataGenerator` 도구는 Markdown 테이블(`Template/*.md`)을 JSON으로 변환합니다.
`ExcelToJsonExporter`는 동일한 `타입:컬럼명` 헤더 규약을 사용하므로, 엑셀로 작성한 데이터를
동일한 JSON 구조로 추출할 수 있습니다.

| 도구 | 입력 | 출력 |
|------|------|------|
| DataGenerator | Markdown 테이블 (`.md`) | JSON + UE5 C++ 헤더 |
| **ExcelToJsonExporter** | **엑셀 시트 (`.xlsx`)** | **JSON** |

## 빌드

```bash
dotnet build tools/ExcelToJsonExporter
```

## 라이선스

EPPlus는 Polyform Noncommercial License 1.0.0 하에 배포됩니다.
상업적 사용이 필요한 경우 EPPlus 라이선스를 확인하십시오.
