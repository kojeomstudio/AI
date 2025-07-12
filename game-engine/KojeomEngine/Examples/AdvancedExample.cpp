/**
 * @file AdvancedExample.cpp
 * @brief Advanced example using integrated rendering system
 * 
 * This example utilizes all components including Renderer, Mesh, Shader, Texture
 * in the new Engine structure to render various 3D objects.
 */

#include "../Engine/Core/Engine.h"
#include <cmath>

/**
 * @brief Advanced example application class
 */
class AdvancedExampleApp : public KEngine
{
public:
    AdvancedExampleApp() = default;
    ~AdvancedExampleApp() = default;

    /**
     * @brief Application-specific initialization
     */
    HRESULT InitializeApp()
    {
        // Create various meshes from renderer
        auto renderer = GetRenderer();
        if (!renderer)
        {
            LOG_ERROR("Renderer has not been initialized");
            return E_FAIL;
        }

        // Create various meshes
        m_triangleMesh = renderer->CreateTriangleMesh();
        m_cubeMesh = renderer->CreateCubeMesh();
        m_sphereMesh = renderer->CreateSphereMesh(32, 16);

        if (!m_triangleMesh || !m_cubeMesh || !m_sphereMesh)
        {
            LOG_ERROR("Mesh creation failed");
            return E_FAIL;
        }

        // Setup camera position
        auto camera = GetCamera();
        if (camera)
        {
            camera->SetPosition(XMFLOAT3(0.0f, 2.0f, -8.0f));
            camera->LookAt(XMFLOAT3(0.0f, 0.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f));
        }

        LOG_INFO("Advanced example application initialization completed");
        return S_OK;
    }

    /**
     * @brief Frame update
     */
    void Update(float deltaTime) override
    {
        // Call parent class update
        KEngine::Update(deltaTime);

        // Update rotation angle
        m_rotationAngle += deltaTime * 90.0f; // 90 degrees per second
        if (m_rotationAngle > 360.0f)
            m_rotationAngle -= 360.0f;

        // Camera rotation (orbital motion)
        float cameraAngle = deltaTime * 30.0f; // 30 degrees per second
        m_cameraAngle += cameraAngle;
        if (m_cameraAngle > 360.0f)
            m_cameraAngle -= 360.0f;

        float radius = 10.0f;
        float cameraX = radius * sinf(XMConvertToRadians(m_cameraAngle));
        float cameraZ = radius * cosf(XMConvertToRadians(m_cameraAngle));

        auto camera = GetCamera();
        if (camera)
        {
            camera->SetPosition(XMFLOAT3(cameraX, 3.0f, cameraZ));
            camera->LookAt(XMFLOAT3(0.0f, 0.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f));
        }
    }

protected:
    virtual void RenderFrame_Internal() override
    {
        auto renderer = GetRenderer();
        auto camera = GetCamera();

        // 1. Rotating cube at center
        XMMATRIX cubeWorld = XMMatrixRotationY(XMConvertToRadians(m_rotationAngle)) *
            XMMatrixTranslation(0.0f, 0.0f, 0.0f);
        renderer->RenderMeshBasic(m_cubeMesh, cubeWorld);

        // 2. Rotating triangle on left
        XMMATRIX triangleWorld = XMMatrixRotationZ(XMConvertToRadians(m_rotationAngle)) *
            XMMatrixScaling(2.0f, 2.0f, 1.0f) *
            XMMatrixTranslation(-4.0f, 0.0f, 0.0f);
        renderer->RenderMeshBasic(m_triangleMesh, triangleWorld);

        // 3. Rotating sphere on right
        XMMATRIX sphereWorld = XMMatrixRotationX(XMConvertToRadians(m_rotationAngle * 0.5f)) *
            XMMatrixScaling(1.5f, 1.5f, 1.5f) *
            XMMatrixTranslation(4.0f, 0.0f, 0.0f);
        renderer->RenderMeshBasic(m_sphereMesh, sphereWorld);

        // 4. Small cubes orbiting above
        for (int i = 0; i < 6; ++i)
        {
            float angle = (m_rotationAngle + i * 60.0f);
            float x = 2.0f * cosf(XMConvertToRadians(angle));
            float z = 2.0f * sinf(XMConvertToRadians(angle));

            XMMATRIX smallCubeWorld = XMMatrixScaling(0.3f, 0.3f, 0.3f) *
                XMMatrixRotationY(XMConvertToRadians(angle * 2.0f)) *
                XMMatrixTranslation(x, 3.0f, z);
            renderer->RenderMeshBasic(m_cubeMesh, smallCubeWorld);
        }

        // 5. Large rectangle on floor (grid-like)
        XMMATRIX floorWorld = XMMatrixScaling(10.0f, 0.1f, 10.0f) *
            XMMatrixTranslation(0.0f, -2.0f, 0.0f);

        // Use checkerboard texture
        auto textureManager = renderer->GetTextureManager();
        auto checkerTexture = textureManager->GetCheckerboardTexture();
        renderer->RenderMesh(m_cubeMesh, floorWorld, checkerTexture);
    }

private:
    // Meshes
    std::shared_ptr<KMesh> m_triangleMesh;
    std::shared_ptr<KMesh> m_cubeMesh;
    std::shared_ptr<KMesh> m_sphereMesh;

    // Animation variables
    float m_rotationAngle = 0.0f;
    float m_cameraAngle = 0.0f;
};

/**
 * @brief Main function
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

    // Use improved engine helper function (with custom initialization)
    return KEngine::RunApplication<AdvancedExampleApp>(
        hInstance,
        L"Advanced Rendering Example - KojeomEngine",
        1280, 720,
        [](AdvancedExampleApp* app) -> HRESULT {
            // Application-specific initialization
            return app->InitializeApp();
        }
    );
} 