#include "GraphicsDevice.h"

KGraphicsDevice::~KGraphicsDevice()
{
    Cleanup();
}

HRESULT KGraphicsDevice::Initialize(HWND InWindowHandle, UINT32 InWidth, UINT32 InHeight, bool bEnableDebugLayer)
{
    WindowHandle = InWindowHandle;
    Width = InWidth;
    Height = InHeight;
    bDebugLayerEnabled = bEnableDebugLayer;

    LOG_INFO("Initializing Graphics Device...");

    // Create device and device context
    HRESULT hr = CreateDevice();
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Device creation failed");
        return hr;
    }

    // Create swap chain
    hr = CreateSwapChain(InWindowHandle);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Swap chain creation failed");
        return hr;
    }

    // Create render target view
    hr = CreateRenderTargetView();
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Render target view creation failed");
        return hr;
    }

    // Setup viewport
    SetupViewport();

    LOG_INFO("Graphics Device initialized successfully");
    return S_OK;
}

void KGraphicsDevice::Cleanup()
{
    if (Context)
    {
        Context->ClearState();
    }

    RenderTargetView.Reset();
    SwapChain.Reset();
    Context.Reset();
    Device.Reset();

#ifdef _DEBUG
    if (Debug)
    {
        Debug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL | D3D11_RLDO_IGNORE_INTERNAL);
        Debug.Reset();
    }
#endif
}

void KGraphicsDevice::BeginFrame(const float ClearColor[4])
{
    // Clear render target
    Context->ClearRenderTargetView(RenderTargetView.Get(), ClearColor);

    // Set render target
    Context->OMSetRenderTargets(1, RenderTargetView.GetAddressOf(), nullptr);
}

void KGraphicsDevice::EndFrame(bool bVSync)
{
    HRESULT hr = SwapChain->Present(bVSync ? 1 : 0, 0);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Present failed");
    }
}

HRESULT KGraphicsDevice::ResizeBuffers(UINT32 NewWidth, UINT32 NewHeight)
{
    if (NewWidth == 0 || NewHeight == 0)
    {
        return E_INVALIDARG;
    }

    Width = NewWidth;
    Height = NewHeight;

    // Release render target view
    Context->OMSetRenderTargets(0, nullptr, nullptr);
    RenderTargetView.Reset();

    // Resize swap chain buffers
    HRESULT hr = SwapChain->ResizeBuffers(0, NewWidth, NewHeight, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Swap chain buffer resize failed");
        return hr;
    }

    // Recreate render target view
    hr = CreateRenderTargetView();
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Render target view recreation failed");
        return hr;
    }

    // Update viewport
    SetupViewport();

    return S_OK;
}

HRESULT KGraphicsDevice::CreateDevice()
{
    UINT CreateDeviceFlags = 0;

#ifdef _DEBUG
    CreateDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };

    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware acceleration
        nullptr,                    // No software module
        CreateDeviceFlags,          // Creation flags
        FeatureLevels,              // Feature levels
        ARRAYSIZE(FeatureLevels),   // Number of feature levels
        D3D11_SDK_VERSION,          // SDK version
        &Device,                    // Output device
        &FeatureLevel,              // Output feature level
        &Context                    // Output device context
    );

    if (FAILED(hr))
    {
        LOG_ERROR("Failed to create D3D11 device");
        return hr;
    }

#ifdef _DEBUG
    // Get debug interface for more detailed error information
    hr = Device->QueryInterface(IID_PPV_ARGS(&Debug));
    if (FAILED(hr))
    {
        LOG_WARNING("Could not get debug interface");
    }
#endif

    LOG_INFO("D3D11 Device created successfully");
    return S_OK;
}

HRESULT KGraphicsDevice::CreateSwapChain(HWND InWindowHandle)
{
    // Get DXGI factory
    ComPtr<IDXGIDevice> DxgiDevice;
    HRESULT hr = Device->QueryInterface(IID_PPV_ARGS(&DxgiDevice));
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter> DxgiAdapter;
    hr = DxgiDevice->GetAdapter(&DxgiAdapter);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIFactory> DxgiFactory;
    hr = DxgiAdapter->GetParent(IID_PPV_ARGS(&DxgiFactory));
    if (FAILED(hr)) return hr;

    // Create swap chain description
    DXGI_SWAP_CHAIN_DESC SwapChainDesc = {};
    SwapChainDesc.BufferCount = 1;
    SwapChainDesc.BufferDesc.Width = Width;
    SwapChainDesc.BufferDesc.Height = Height;
    SwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    SwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    SwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    SwapChainDesc.OutputWindow = InWindowHandle;
    SwapChainDesc.SampleDesc.Count = 1;
    SwapChainDesc.SampleDesc.Quality = 0;
    SwapChainDesc.Windowed = TRUE;
    SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    // Create swap chain
    hr = DxgiFactory->CreateSwapChain(Device.Get(), &SwapChainDesc, &SwapChain);
    if (FAILED(hr))
    {
        LOG_ERROR("Failed to create swap chain");
        return hr;
    }

    LOG_INFO("Swap chain created successfully");
    return S_OK;
}

HRESULT KGraphicsDevice::CreateRenderTargetView()
{
    // Get back buffer from swap chain
    ComPtr<ID3D11Texture2D> BackBuffer;
    HRESULT hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
    if (FAILED(hr))
    {
        LOG_ERROR("Failed to get back buffer");
        return hr;
    }

    // Create render target view
    hr = Device->CreateRenderTargetView(BackBuffer.Get(), nullptr, &RenderTargetView);
    if (FAILED(hr))
    {
        LOG_ERROR("Failed to create render target view");
        return hr;
    }

    LOG_INFO("Render target view created successfully");
    return S_OK;
}

void KGraphicsDevice::SetupViewport()
{
    // Setup viewport
    D3D11_VIEWPORT Viewport = {};
    Viewport.TopLeftX = 0.0f;
    Viewport.TopLeftY = 0.0f;
    Viewport.Width = static_cast<float>(Width);
    Viewport.Height = static_cast<float>(Height);
    Viewport.MinDepth = 0.0f;
    Viewport.MaxDepth = 1.0f;

    Context->RSSetViewports(1, &Viewport);
} 