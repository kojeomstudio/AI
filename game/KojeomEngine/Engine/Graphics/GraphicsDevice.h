#pragma once

#include "../Utils/Common.h"
#include "../Utils/Logger.h"

/**
 * @brief DirectX 11 Graphics Device Manager
 * 
 * Manages DirectX 11 device, device context, swap chain and
 * other core graphics resources.
 */
class KGraphicsDevice
{
public:
    KGraphicsDevice() = default;
    ~KGraphicsDevice();

    // Copy and move prevention
    KGraphicsDevice(const KGraphicsDevice&) = delete;
    KGraphicsDevice& operator=(const KGraphicsDevice&) = delete;

    /**
     * @brief Initialize graphics device
     * @param WindowHandle Window handle
     * @param Width Back buffer width
     * @param Height Back buffer height
     * @param bEnableDebugLayer Whether to enable debug layer
     * @return Success: S_OK
     */
    HRESULT Initialize(HWND WindowHandle, UINT32 Width, UINT32 Height, bool bEnableDebugLayer = false);

    /**
     * @brief Cleanup resources
     */
    void Cleanup();

    /**
     * @brief Begin frame - clear back buffer
     * @param ClearColor Color to clear with
     */
    void BeginFrame(const float ClearColor[4] = Colors::CornflowerBlue);

    /**
     * @brief End frame - present back buffer
     * @param bVSync Whether to use V-Sync
     */
    void EndFrame(bool bVSync = true);

    /**
     * @brief Handle window resize
     * @param NewWidth New width
     * @param NewHeight New height
     */
    HRESULT ResizeBuffers(UINT32 NewWidth, UINT32 NewHeight);

    // Accessors
    ID3D11Device* GetDevice() const { return Device.Get(); }
    ID3D11DeviceContext* GetContext() const { return Context.Get(); }
    IDXGISwapChain* GetSwapChain() const { return SwapChain.Get(); }
    ID3D11RenderTargetView* GetRenderTargetView() const { return RenderTargetView.Get(); }
    
    UINT32 GetWidth() const { return Width; }
    UINT32 GetHeight() const { return Height; }
    float GetAspectRatio() const { return static_cast<float>(Width) / static_cast<float>(Height); }

private:
    /**
     * @brief Create DirectX 11 device
     */
    HRESULT CreateDevice();

    /**
     * @brief Create swap chain
     * @param WindowHandle Window handle
     */
    HRESULT CreateSwapChain(HWND WindowHandle);

    /**
     * @brief Create render target view
     */
    HRESULT CreateRenderTargetView();

    /**
     * @brief Setup viewport
     */
    void SetupViewport();

private:
    // DirectX 11 Core Objects (ComPtr handles automatic resource management)
    ComPtr<ID3D11Device> Device;
    ComPtr<ID3D11DeviceContext> Context;
    ComPtr<IDXGISwapChain> SwapChain;
    ComPtr<ID3D11RenderTargetView> RenderTargetView;

    // Device settings
    // Debug interface
#ifdef _DEBUG
    ComPtr<ID3D11Debug> Debug;
#endif

    // Window handle
    HWND WindowHandle = nullptr;

    // Device settings
    D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_11_0;
    UINT32 Width = 0;
    UINT32 Height = 0;
    bool bDebugLayerEnabled = false;
}; 