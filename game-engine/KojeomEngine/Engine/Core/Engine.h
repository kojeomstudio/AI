#pragma once

#include <windows.h>
#include <string>
#include <memory>
#include <functional>
#include "../Utils/Common.h"
#include "../Utils/Logger.h"
#include "../Graphics/GraphicsDevice.h"
#include "../Graphics/Camera.h"
#include "../Graphics/Renderer.h"

/**
 * @brief Main engine class
 * 
 * Responsible for engine initialization, main loop execution, and
 * window management, graphics device, camera, etc.
 */
class KEngine
{
public:
    KEngine();
    ~KEngine();

    // Prevent copy and move
    KEngine(const KEngine&) = delete;
    KEngine& operator=(const KEngine&) = delete;

    /**
     * @brief Initialize the engine
     * @param InstanceHandle Application instance
     * @param WindowTitle Window title
     * @param Width Window width
     * @param Height Window height
     * @return S_OK on success
     */
    HRESULT Initialize(HINSTANCE InstanceHandle, 
                      const std::wstring& WindowTitle = L"KojeomEngine",
                      UINT32 Width = EngineConstants::DEFAULT_WINDOW_WIDTH,
                      UINT32 Height = EngineConstants::DEFAULT_WINDOW_HEIGHT);

    /**
     * @brief Run the engine (main loop)
     * @return Application exit code
     */
    int32 Run();

    /**
     * @brief Shutdown the engine
     */
    void Shutdown();

    /**
     * @brief Update frame
     * @param DeltaTime Frame time in seconds
     */
    virtual void Update(float DeltaTime);

    /**
     * @brief Render frame
     */
    virtual void Render();

    /**
     * @brief Handle window resize
     * @param NewWidth New width
     * @param NewHeight New height
     */
    void OnResize(UINT32 NewWidth, UINT32 NewHeight);

    // Accessors
    KGraphicsDevice* GetGraphicsDevice() const { return GraphicsDevice.get(); }
    KCamera* GetCamera() const { return Camera.get(); }
    KRenderer* GetRenderer() const { return Renderer.get(); }
    HWND GetWindowHandle() const { return WindowHandle; }
    
    UINT32 GetWindowWidth() const { return WindowWidth; }
    UINT32 GetWindowHeight() const { return WindowHeight; }
    
    bool IsRunning() const { return bIsRunning; }

    // Static accessor (global engine instance)
    static KEngine* GetInstance() { return Instance; }

    /**
     * @brief Application execution helper function (template)
     * @tparam T Application class that inherits from KEngine
     * @param InstanceHandle Application instance
     * @param WindowTitle Window title
     * @param Width Window width
     * @param Height Window height
     * @param CustomInit Additional initialization function (optional)
     * @return Application exit code
     */
    template<typename T>
    static int32 RunApplication(HINSTANCE InstanceHandle, 
                             const std::wstring& WindowTitle,
                             UINT32 Width = EngineConstants::DEFAULT_WINDOW_WIDTH,
                             UINT32 Height = EngineConstants::DEFAULT_WINDOW_HEIGHT,
                             std::function<HRESULT(T*)> CustomInit = nullptr)
    {
        static_assert(std::is_base_of_v<KEngine, T>, "T must inherit from KEngine");

        // Setup debug environment
        SetupDebugEnvironment();

        LOG_INFO("=== " + StringUtils::WideToMultiByte(WindowTitle) + " Starting ===");

        // Create application
        std::unique_ptr<T> App = std::make_unique<T>();

        // Initialize engine
        HRESULT Result = App->Initialize(InstanceHandle, WindowTitle, Width, Height);
        if (FAILED(Result))
        {
            LOG_ERROR("Engine initialization failed");
            CleanupDebugEnvironment();
            return -1;
        }

        // Custom initialization (if provided)
        if (CustomInit)
        {
            Result = CustomInit(App.get());
            if (FAILED(Result))
            {
                LOG_ERROR("Custom initialization failed");
                App->Shutdown();
                CleanupDebugEnvironment();
                return -1;
            }
        }

        // Run main loop
        int32 ExitCode = App->Run();

        // Cleanup
        App->Shutdown();
        CleanupDebugEnvironment();

        LOG_INFO("=== " + StringUtils::WideToMultiByte(WindowTitle) + " Ending ===");
        return ExitCode;
    }

protected:
    /**
     * @brief Initialize window
     * @param InstanceHandle Application instance
     * @param WindowTitle Window title
     */
    HRESULT InitializeWindow(HINSTANCE InstanceHandle, const std::wstring& WindowTitle);

    /**
     * @brief Initialize graphics system
     */
    HRESULT InitializeGraphics();

    /// <summary>
	/// render frame implementation
    /// </summary>
    virtual void RenderFrame_Internal() {};

    /**
     * @brief Calculate frame timing
     */
    void CalculateFrameStats();

private:
    /**
     * @brief Process messages
     */
    void ProcessMessages();

    /**
     * @brief Update timer
     */
    void UpdateTimer();

private:
    // Window related
    HINSTANCE InstanceHandle;
    HWND WindowHandle;
    std::wstring WindowTitle;
    UINT32 WindowWidth;
    UINT32 WindowHeight;

    // System components
    std::unique_ptr<KGraphicsDevice> GraphicsDevice;
    std::unique_ptr<KCamera> Camera;
    std::unique_ptr<KRenderer> Renderer;

    // Engine state
    bool bIsRunning;
    bool bIsInitialized;

    // Timing
    LARGE_INTEGER Frequency;
    LARGE_INTEGER LastTime;
    float DeltaTime;
    float TotalTime;

    // Frame statistics
    UINT32 FrameCount;
    float FrameTime;
    float FPS;

    // Global instance
    static KEngine* Instance;

public:
    friend LRESULT CALLBACK WindowProc(HWND WindowHandle, UINT Message, WPARAM WParam, LPARAM LParam);

    /**
     * @brief Setup debug environment (console, memory leak detection)
     */
    static void SetupDebugEnvironment();

    /**
     * @brief Cleanup debug environment
     */
    static void CleanupDebugEnvironment();
}; 