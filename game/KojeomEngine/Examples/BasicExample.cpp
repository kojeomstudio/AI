#include "../Engine/Core/Engine.h"

/**
 * @brief Basic engine example application
 * 
 * Simple example using the new KojeomEngine structure.
 * Renders a blue background and displays FPS.
 */
class BasicExample : public KEngine
{
public:
    BasicExample() = default;
    ~BasicExample() = default;

    /**
     * @brief Update logic (called every frame)
     */
    void Update(float deltaTime) override
    {
        // Call parent class's default update
        KEngine::Update(deltaTime);

        // Add game-specific update logic here
        // Example: input processing, game object updates, etc.
    }

protected:
    virtual void RenderFrame_Internal() override
    {
        // to do.
	}
};

/**
 * @brief Application entry point
 */
int WINAPI WinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine,
    _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    UNREFERENCED_PARAMETER(nCmdShow);

    // Use improved engine helper function
    return KEngine::RunApplication<BasicExample>(
        hInstance,
        L"KojeomEngine - Basic Example",
        1024, 768
    );
} 