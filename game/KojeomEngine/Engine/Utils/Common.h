#pragma once

// UTF-8 character encoding support for MSVC
#ifdef _MSC_VER
#pragma execution_character_set("utf-8")
#endif

// Windows headers
#include <windows.h>

// DirectX 11 headers
#include <d3d11_1.h>
#include <directxcolors.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

// Standard libraries
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <iostream>
#include <crtdbg.h>

// DirectX namespace usage
using namespace DirectX;

// Unreal Engine style type definitions
using int8   = int8_t;
using int16  = int16_t;
using int32  = int32_t;
using int64  = int64_t;
using uint8  = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// Common macro definitions
#define SAFE_RELEASE(p) if(p) { p->Release(); p = nullptr; }
#define SAFE_DELETE(p) if(p) { delete p; p = nullptr; }
#define SAFE_DELETE_ARRAY(p) if(p) { delete[] p; p = nullptr; }

// Error handling macros
#define CHECK_HRESULT(hr) if(FAILED(hr)) return hr;
#define LOG_ERROR(msg) KLogger::Error(msg)
#define LOG_INFO(msg) KLogger::Info(msg)
#define LOG_WARNING(msg) KLogger::Warning(msg)

// Type aliases
using GraphicsDevicePtr = std::unique_ptr<class GraphicsDevice>;
using RendererPtr = std::unique_ptr<class Renderer>;
using CameraPtr = std::unique_ptr<class Camera>;

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

// Forward declarations
class KEngine;
class KGraphicsDevice;
class KRenderer;
class KCamera;
class KLogger;

// String conversion utilities
namespace StringUtils
{
    /**
     * @brief Convert wide string to multi-byte string safely
     * @param WideStr Wide string to convert
     * @return Converted multi-byte string
     */
    inline std::string WideToMultiByte(const std::wstring& WideStr)
    {
        if (WideStr.empty()) return std::string();
        
        int SizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &WideStr[0], (int)WideStr.size(), NULL, 0, NULL, NULL);
        std::string StrTo(SizeNeeded, 0);
        WideCharToMultiByte(CP_UTF8, 0, &WideStr[0], (int)WideStr.size(), &StrTo[0], SizeNeeded, NULL, NULL);
        return StrTo;
    }

    /**
     * @brief Convert multi-byte string to wide string safely
     * @param MultiStr Multi-byte string to convert
     * @return Converted wide string
     */
    inline std::wstring MultiByteToWide(const std::string& MultiStr)
    {
        if (MultiStr.empty()) return std::wstring();
        
        int SizeNeeded = MultiByteToWideChar(CP_UTF8, 0, &MultiStr[0], (int)MultiStr.size(), NULL, 0);
        std::wstring WStrTo(SizeNeeded, 0);
        MultiByteToWideChar(CP_UTF8, 0, &MultiStr[0], (int)MultiStr.size(), &WStrTo[0], SizeNeeded);
        return WStrTo;
    }
}

// Constants definition
namespace EngineConstants
{
    constexpr UINT32 DEFAULT_WINDOW_WIDTH = 1024;
    constexpr UINT32 DEFAULT_WINDOW_HEIGHT = 768;
    constexpr float DEFAULT_FOV = XM_PIDIV4;
    constexpr float DEFAULT_NEAR_PLANE = 0.1f;
    constexpr float DEFAULT_FAR_PLANE = 1000.0f;
} 