#include "Engine.h"

// Global instance definition
KEngine* KEngine::Instance = nullptr;

// Window procedure forward declaration
LRESULT CALLBACK WindowProc(HWND WindowHandle, UINT Message, WPARAM WParam, LPARAM LParam);

KEngine::KEngine()
    : InstanceHandle(nullptr)
    , WindowHandle(nullptr)
    , WindowTitle(L"KojeomEngine")
    , WindowWidth(EngineConstants::DEFAULT_WINDOW_WIDTH)
    , WindowHeight(EngineConstants::DEFAULT_WINDOW_HEIGHT)
    , bIsRunning(false)
    , bIsInitialized(false)
    , DeltaTime(0.0f)
    , TotalTime(0.0f)
    , FrameCount(0)
    , FrameTime(0.0f)
    , FPS(0.0f)
{
    // Set global instance
    Instance = this;

    // Initialize high-performance timer
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&LastTime);

    LOG_INFO("Engine constructor called");
}

KEngine::~KEngine()
{
    Shutdown();
    Instance = nullptr;
    LOG_INFO("Engine destructor called");
}

HRESULT KEngine::Initialize(HINSTANCE InInstanceHandle, const std::wstring& InWindowTitle, UINT32 InWidth, UINT32 InHeight)
{
    LOG_INFO("Engine initialization starting...");

    InstanceHandle = InInstanceHandle;
    WindowTitle = InWindowTitle;
    WindowWidth = InWidth;
    WindowHeight = InHeight;

    // Initialize window
    HRESULT hr = InitializeWindow(InInstanceHandle, InWindowTitle);
    if (FAILED(hr))
    {
        LOG_ERROR("Window initialization failed");
        return hr;
    }

    // Initialize graphics system
    hr = InitializeGraphics();
    if (FAILED(hr))
    {
        LOG_ERROR("Graphics system initialization failed");
        return hr;
    }

    // Initialize camera
    Camera = std::make_unique<KCamera>();
    Camera->SetPerspective(
        EngineConstants::DEFAULT_FOV,
        static_cast<float>(InWidth) / static_cast<float>(InHeight),
        EngineConstants::DEFAULT_NEAR_PLANE,
        EngineConstants::DEFAULT_FAR_PLANE
    );

    bIsInitialized = true;
    LOG_INFO("Engine initialization completed");

    return S_OK;
}

int32 KEngine::Run()
{
    if (!bIsInitialized)
    {
        LOG_ERROR("Engine has not been initialized!");
        return -1;
    }

    LOG_INFO("Engine main loop starting");
    bIsRunning = true;

    // Main game loop
    while (bIsRunning)
    {
        // Process window messages
        ProcessMessages();

        if (bIsRunning)
        {
            // Update timer
            UpdateTimer();

            // Update game logic
            Update(DeltaTime);

            // Render
            Render();

            // Calculate frame statistics
            CalculateFrameStats();
        }
    }

    LOG_INFO("Engine main loop ended");
    return 0;
}

void KEngine::Shutdown()
{
    LOG_INFO("Engine shutdown starting...");

    bIsRunning = false;

    // Cleanup components
    if (Renderer)
    {
        Renderer->Cleanup();
        Renderer.reset();
    }

    if (Camera)
    {
        Camera.reset();
    }

    if (GraphicsDevice)
    {
        GraphicsDevice->Cleanup();
        GraphicsDevice.reset();
    }

    // Cleanup window
    if (WindowHandle)
    {
        DestroyWindow(WindowHandle);
        WindowHandle = nullptr;
    }

    if (InstanceHandle)
    {
        UnregisterClass(L"KojeomEngineWindow", InstanceHandle);
    }

    bIsInitialized = false;
    LOG_INFO("Engine shutdown completed");
}

void KEngine::Update(float InDeltaTime)
{
    // Basic update logic
    // Override in derived classes to implement game-specific logic
    TotalTime += InDeltaTime;

    // Update camera
    if (Camera)
    {
        Camera->UpdateMatrices();
    }
}

void KEngine::Render()
{
    if (!Renderer || !Camera)
    {
        return;
    }

    // Begin frame
    float ClearColor[4] = { 0.0f, 0.2f, 0.4f, 1.0f }; // Dark blue
    Renderer->BeginFrame(Camera.get(), ClearColor);

    // Basic rendering logic
    // Override in derived classes to implement actual rendering
    RenderFrame_Internal();

    // End frame
    Renderer->EndFrame(true); // Use V-Sync
}

void KEngine::OnResize(UINT32 NewWidth, UINT32 NewHeight)
{
    if (NewWidth == 0 || NewHeight == 0)
    {
        return;
    }
        

    WindowWidth = NewWidth;
    WindowHeight = NewHeight;

    // Resize graphics device
    if (GraphicsDevice)
    {
        HRESULT hr = GraphicsDevice->ResizeBuffers(NewWidth, NewHeight);
        if (FAILED(hr))
        {
            KLogger::HResultError(hr, "Window resize failed");
        }
    }

    // Update camera aspect ratio
    if (Camera)
    {
        float AspectRatio = static_cast<float>(NewWidth) / static_cast<float>(NewHeight);
        Camera->SetPerspective(
            Camera->GetFovY(),
            AspectRatio,
            Camera->GetNearZ(),
            Camera->GetFarZ()
        );
    }

    LOG_INFO("Window resized to " + std::to_string(NewWidth) + "x" + std::to_string(NewHeight));
}

HRESULT KEngine::InitializeWindow(HINSTANCE hInstance, const std::wstring& windowTitle)
{
    // Register window class
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = nullptr;
    wc.lpszClassName = L"KojeomEngineWindow";
    wc.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);

    if (!RegisterClassEx(&wc))
    {
        return E_FAIL;
    }

    // Calculate window size to fit client area
    RECT rc = { 0, 0, static_cast<LONG>(WindowWidth), static_cast<LONG>(WindowHeight) };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    // Create window
    WindowHandle = CreateWindow(
        L"KojeomEngineWindow",
        windowTitle.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!WindowHandle)
    {
        return E_FAIL;
    }

    // Show window
    ShowWindow(WindowHandle, SW_SHOW);
    UpdateWindow(WindowHandle);

    return S_OK;
}

HRESULT KEngine::InitializeGraphics()
{
    // Create graphics device
    GraphicsDevice = std::make_unique<KGraphicsDevice>();

    // Initialize graphics device
    HRESULT hr = GraphicsDevice->Initialize(WindowHandle, WindowWidth, WindowHeight, true);
    if (FAILED(hr))
    {
        return hr;
    }

    // Create and initialize renderer
    Renderer = std::make_unique<KRenderer>();
    hr = Renderer->Initialize(GraphicsDevice.get());
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Renderer initialization failed");
        return hr;
    }

    return S_OK;
}

void KEngine::ProcessMessages()
{
    MSG msg = {};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        if (msg.message == WM_QUIT)
        {
            bIsRunning = false;
            break;
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

void KEngine::UpdateTimer()
{
    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);

    // Calculate delta time (in seconds)
    DeltaTime = static_cast<float>(currentTime.QuadPart - LastTime.QuadPart) / 
                  static_cast<float>(Frequency.QuadPart);

    // Limit delta time (maximum 1/30 second)
    if (DeltaTime > 1.0f / 30.0f)
    {
        DeltaTime = 1.0f / 30.0f;
    }

    LastTime = currentTime;
}

void KEngine::CalculateFrameStats()
{
    FrameCount++;
    FrameTime += DeltaTime;

    // Calculate FPS every second
    if (FrameTime >= 1.0f)
    {
        FPS = static_cast<float>(FrameCount) / FrameTime;

        // Display FPS in window title
        std::wstring title = WindowTitle + L" - FPS: " + std::to_wstring(static_cast<int>(FPS));
        SetWindowText(WindowHandle, title.c_str());

        FrameCount = 0;
        FrameTime = 0.0f;
    }
}

// Window procedure implementation
LRESULT CALLBACK WindowProc(HWND WindowHandle, UINT Message, WPARAM WParam, LPARAM LParam)
{
    KEngine* engine = KEngine::Instance;
    
    switch (Message)
    {
    case WM_SIZE:
        if (engine && WParam != SIZE_MINIMIZED)
        {
            UINT width = LOWORD(LParam);
            UINT height = HIWORD(LParam);
            engine->OnResize(width, height);
        }
        break;

    case WM_KEYDOWN:
        if (WParam == VK_ESCAPE)
        {
            PostQuitMessage(0);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(WindowHandle, Message, WParam, LParam);
    }

    return 0;
}

void KEngine::SetupDebugEnvironment()
{
#ifdef _DEBUG
    // Setup memory leak detection
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    // Create debug console
    if (AllocConsole())
    {
        // Redirect standard I/O to console
        FILE* pCout;
        FILE* pCin;
        FILE* pCerr;
        
        freopen_s(&pCout, "CONOUT$", "w", stdout);
        freopen_s(&pCin, "CONIN$", "r", stdin);
        freopen_s(&pCerr, "CONOUT$", "w", stderr);

        // Set console title
        SetConsoleTitle(L"KojeomEngine Debug Console");
        
        // Synchronize C++ streams
        std::ios::sync_with_stdio(true);
        std::wcout.clear();
        std::cout.clear();
        std::wcerr.clear();
        std::cerr.clear();
        std::wcin.clear();
        std::cin.clear();
    }

    LOG_INFO("Debug environment setup completed");
#endif
}

void KEngine::CleanupDebugEnvironment()
{
#ifdef _DEBUG
    LOG_INFO("Cleaning up debug environment...");
    
    // Cleanup console
    FreeConsole();
#endif
} 