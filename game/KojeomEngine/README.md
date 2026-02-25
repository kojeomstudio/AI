# KojeomEngine

DirectX 11 기반의 경량화된 C++ 게임 엔진입니다. 모듈성과 확장성을 중시하여 설계되었으며, 간결한 코드 스타일과 쉬운 사용법을 목표로 합니다.

## 🏗️ 엔진 아키텍처

### 📁 프로젝트 구조
```
KojeomEngine/
├── Engine/                 # 엔진 코어
│   ├── Core/              # 핵심 시스템
│   │   └── Engine.h/cpp   # 메인 엔진 클래스
│   ├── Graphics/          # 그래픽스 시스템
│   │   ├── GraphicsDevice.h/cpp  # DirectX 11 디바이스 관리
│   │   ├── Camera.h/cpp          # 3D 카메라 시스템
│   │   ├── Renderer.h/cpp        # 통합 렌더링 시스템
│   │   ├── Shader.h/cpp          # 셰이더 관리 시스템
│   │   ├── Mesh.h/cpp            # 메시 렌더링 시스템
│   │   └── Texture.h/cpp         # 텍스처 관리 시스템
│   └── Utils/             # 유틸리티
│       ├── Common.h       # 공통 헤더 및 매크로
│       └── Logger.h       # 로깅 시스템
├── Examples/              # 예제 코드
│   ├── BasicExample.cpp   # 기본 사용 예제
│   ├── TriangleExample.cpp # 3D 렌더링 예제
│   └── AdvancedExample.cpp # 통합 렌더링 시스템 예제
├── Renderer/              # 기존 렌더러 (레거시)
└── KojeomEngine/          # 기존 프로젝트 (레거시)
```

## ✨ 주요 특징

### 🎯 경량화된 설계
- **최소한의 의존성**: DirectX 11과 표준 C++ 라이브러리만 사용
- **효율적인 메모리 관리**: ComPtr을 사용한 자동 리소스 관리
- **모듈화된 구조**: 필요한 컴포넌트만 사용 가능

### 🔧 핵심 컴포넌트

#### Engine 클래스
```cpp
class Engine
{
public:
    HRESULT Initialize(HINSTANCE hInstance, const std::wstring& windowTitle, 
                      UINT width, UINT height);
    int Run();
    virtual void Update(float deltaTime);
    virtual void Render();
    void Shutdown();
};
```

#### GraphicsDevice 클래스
- DirectX 11 디바이스 및 스왑체인 관리
- 자동 리소스 정리
- 윈도우 크기 변경 대응

#### Camera 클래스
- 3D 카메라 시스템
- 뷰/프로젝션 매트릭스 관리
- 원근/직교 투영 지원

#### Renderer 클래스
- 통합 렌더링 시스템
- 모든 그래픽스 컴포넌트 통합 관리
- 간단한 렌더링 인터페이스 제공

#### Shader 시스템
- 런타임 셰이더 컴파일
- 파일 및 문자열 소스 지원
- 셰이더 프로그램 관리

#### Mesh 시스템
- 3D 메시 렌더링
- 내장 프리미티브 (삼각형, 큐브, 구체 등)
- 효율적인 버퍼 관리

#### Texture 시스템
- 2D 텍스처 관리
- 런타임 텍스처 생성
- 캐싱 및 리소스 관리

#### Logger 시스템
- 디버그 빌드에서 콘솔 및 Visual Studio 출력 창 지원
- 릴리즈 빌드에서 최소 오버헤드
- HRESULT 에러 로깅 지원

## 🚀 사용법

### 기본 애플리케이션 작성

```cpp
#include "Engine/Core/Engine.h"

class MyGame : public Engine
{
public:
    void Update(float deltaTime) override
    {
        Engine::Update(deltaTime);
        // 게임 로직 업데이트
    }

    void Render() override
    {
        auto renderer = GetRenderer();
        auto camera = GetCamera();
        
        float clearColor[4] = { 0.0f, 0.3f, 0.6f, 1.0f };
        renderer->BeginFrame(camera, clearColor);
        
        // 3D 메시 렌더링
        auto cube = renderer->CreateCubeMesh();
        XMMATRIX world = XMMatrixIdentity();
        renderer->RenderMeshBasic(cube, world);
        
        renderer->EndFrame(true);
    }
};

int WINAPI wWinMain(HINSTANCE hInstance, ...)
{
    auto app = std::make_unique<MyGame>();
    
    if (FAILED(app->Initialize(hInstance, L"My Game", 1024, 768)))
        return -1;
    
    int result = app->Run();
    app->Shutdown();
    
    return result;
}
```

### 카메라 사용

```cpp
Camera* camera = GetCamera();

// 카메라 위치 설정
camera->SetPosition(0.0f, 5.0f, -10.0f);

// 특정 지점 바라보기
camera->LookAt(XMFLOAT3(0.0f, 0.0f, 0.0f));

// 원근 투영 설정
camera->SetPerspective(XM_PIDIV4, 16.0f/9.0f, 0.1f, 1000.0f);

// 매트릭스 가져오기
XMMATRIX viewMatrix = camera->GetViewMatrix();
XMMATRIX projMatrix = camera->GetProjectionMatrix();
```

### 렌더링 시스템 사용

```cpp
// 렌더러 가져오기
Renderer* renderer = GetRenderer();

// 다양한 메시 생성
auto triangleMesh = renderer->CreateTriangleMesh();
auto cubeMesh = renderer->CreateCubeMesh();
auto sphereMesh = renderer->CreateSphereMesh(32, 16);

// 렌더링 루프에서
void Render() override
{
    auto camera = GetCamera();
    renderer->BeginFrame(camera);
    
    // 기본 렌더링
    XMMATRIX world = XMMatrixRotationY(XMConvertToRadians(45.0f));
    renderer->RenderMeshBasic(cubeMesh, world);
    
    // 커스텀 텍스처와 함께 렌더링
    auto texture = renderer->GetTextureManager()->GetCheckerboardTexture();
    renderer->RenderMesh(sphereMesh, world, texture);
    
    renderer->EndFrame(true);
}
```

## 🛠️ 빌드 요구사항

- **컴파일러**: Visual Studio 2019 이상 (C++17 지원)
- **플랫폼**: Windows 10/11
- **DirectX**: DirectX 11 SDK
- **추가 라이브러리**: 없음 (Windows SDK 포함)

## 📝 개발 가이드라인

### 코딩 스타일
- **명명 규칙**: PascalCase (클래스, 함수), camelCase (변수), m_ prefix (멤버 변수)
- **주석**: 중요한 기능에는 Doxygen 스타일 주석 사용
- **에러 처리**: HRESULT 기반 에러 처리, 로거 시스템 활용

### 성능 고려사항
- ComPtr 사용으로 메모리 안전성 확보
- 델타 타임 기반 업데이트 루프
- V-Sync 지원으로 안정적인 프레임 레이트

## 🎮 예제 실행

`Examples/BasicExample.cpp`를 컴파일하고 실행하면:
- 파란색 배경의 윈도우 생성
- 실시간 FPS 표시
- ESC 키로 종료
- 윈도우 크기 조절 지원

## 🔄 기존 Renderer 폴더 구조와의 비교

### 🆕 새로운 구조의 장점
1. **완전한 책임 분리**: 렌더링, 윈도우 관리, 카메라, 셰이더, 메시, 텍스처가 독립된 클래스로 분리
2. **안전한 리소스 관리**: ComPtr 사용으로 메모리 누수 완전 방지
3. **높은 확장성**: 상속과 컴포지션을 통한 유연한 기능 확장
4. **일관된 에러 처리**: HRESULT 기반 에러 처리 및 로깅 시스템
5. **모듈성**: 필요한 컴포넌트만 선택적으로 사용 가능
6. **사용 편의성**: 간단한 API로 복잡한 렌더링 기능 제공
7. **성능 최적화**: 효율적인 리소스 캐싱 및 관리

### 🔧 기존 구조에서 통합된 기능들
- **PrimitiveModel.cpp** → **Mesh.cpp**: 확장된 메시 시스템으로 발전
- **ShaderModel.cpp** → **Shader.cpp**: 향상된 셰이더 관리 시스템
- **CTexture.cpp** → **Texture.cpp**: 완전히 새로운 텍스처 관리 시스템
- **RendererCore.cpp** → **Renderer.cpp**: 모든 기능을 통합하는 상위 렌더러

### ⚠️ 이전 구조의 문제점들
- RendererCore에 너무 많은 책임 집중 (단일 책임 원칙 위반)
- 수동 리소스 관리로 인한 메모리 누수 위험
- 확장이 어려운 단일 클래스 구조
- 일관성 없는 에러 처리
- 하드코딩된 셰이더 파일 의존성

## 📈 향후 계획

### ✅ 완료된 기능
- [x] 셰이더 관리 시스템
- [x] 텍스처 관리자
- [x] 기본 메시 렌더링 시스템
- [x] 통합 렌더러 시스템

### 🚧 개발 예정
- [ ] 3D 모델 로딩 시스템 (.obj, .fbx 지원)
- [ ] 이미지 파일 로딩 (.png, .jpg, .dds 지원)
- [ ] 입력 시스템 (키보드, 마우스)
- [ ] 오디오 시스템
- [ ] 씬 그래프 및 Transform 시스템
- [ ] 애니메이션 시스템
- [ ] 조명 시스템 (Phong, PBR)
- [ ] 포스트 프로세싱 효과
- [ ] ImGui 통합
- [ ] 물리 엔진 통합 (Bullet Physics)

## 📜 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조
