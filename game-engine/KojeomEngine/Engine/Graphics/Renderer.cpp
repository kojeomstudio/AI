#include "Renderer.h"

HRESULT KRenderer::Initialize(KGraphicsDevice* InGraphicsDevice)
{
    if (!InGraphicsDevice)
    {
        LOG_ERROR("Invalid graphics device");
        return E_INVALIDARG;
    }

    GraphicsDevice = InGraphicsDevice;

    LOG_INFO("Initializing Renderer...");

    // Initialize default resources
    HRESULT hr = InitializeDefaultResources();
    if (FAILED(hr))
    {
        LOG_ERROR("Failed to initialize default resources");
        return hr;
    }

    LOG_INFO("Renderer initialized successfully");
    return S_OK;
}

void KRenderer::BeginFrame(KCamera* InCamera, const float ClearColor[4])
{
    if (!GraphicsDevice || !InCamera)
    {
        return;
    }

    CurrentCamera = InCamera;
    bInFrame = true;

    // Update camera matrices
    CurrentCamera->UpdateMatrices();

    // Begin frame on graphics device
    GraphicsDevice->BeginFrame(ClearColor);
}

void KRenderer::EndFrame(bool bVSync)
{
    if (!GraphicsDevice || !bInFrame)
    {
        return;
    }

    // End frame on graphics device
    GraphicsDevice->EndFrame(bVSync);

    bInFrame = false;
    CurrentCamera = nullptr;
}

void KRenderer::RenderObject(const FRenderObject& RenderObject)
{
    if (!GraphicsDevice || !CurrentCamera || !bInFrame)
    {
        return;
    }

    if (!RenderObject.Mesh || !RenderObject.Shader)
    {
        return;
    }

    ID3D11DeviceContext* Context = GraphicsDevice->GetContext();

    // Bind shader program
    RenderObject.Shader->Bind(Context);

    // Bind texture if available
    if (RenderObject.Texture)
    {
        RenderObject.Texture->Bind(Context, 0);
    }

    // Update mesh constant buffer with matrices
    RenderObject.Mesh->UpdateConstantBuffer(
        Context,
        RenderObject.WorldMatrix,
        CurrentCamera->GetViewMatrix(),
        CurrentCamera->GetProjectionMatrix()
    );

    // Render mesh
    RenderObject.Mesh->Render(Context);

    // Unbind texture
    if (RenderObject.Texture)
    {
        RenderObject.Texture->Unbind(Context, 0);
    }

    // Unbind shader program
    RenderObject.Shader->Unbind(Context);
}

void KRenderer::RenderMesh(std::shared_ptr<KMesh> InMesh, const XMMATRIX& WorldMatrix, 
                          std::shared_ptr<KTexture> InTexture)
{
    if (!BasicShader)
    {
        return;
    }

    FRenderObject RenderObject(InMesh, BasicShader, InTexture);
    RenderObject.WorldMatrix = WorldMatrix;
    KRenderer::RenderObject(RenderObject);
}

void KRenderer::RenderMeshBasic(std::shared_ptr<KMesh> InMesh, const XMMATRIX& WorldMatrix)
{
    RenderMesh(InMesh, WorldMatrix, nullptr);
}

void KRenderer::Cleanup()
{
    LOG_INFO("Cleaning up Renderer...");

    // Cleanup resources
    BasicShader.reset();
    TextureManager.Cleanup();

    GraphicsDevice = nullptr;
    CurrentCamera = nullptr;
    bInFrame = false;

    LOG_INFO("Renderer cleanup completed");
}

std::shared_ptr<KMesh> KRenderer::CreateTriangleMesh()
{
    if (!GraphicsDevice)
    {
        return nullptr;
    }

    return KMesh::CreateTriangle(GraphicsDevice->GetDevice());
}

std::shared_ptr<KMesh> KRenderer::CreateQuadMesh()
{
    if (!GraphicsDevice)
    {
        return nullptr;
    }

    return KMesh::CreateQuad(GraphicsDevice->GetDevice());
}

std::shared_ptr<KMesh> KRenderer::CreateCubeMesh()
{
    if (!GraphicsDevice)
    {
        return nullptr;
    }

    return KMesh::CreateCube(GraphicsDevice->GetDevice());
}

std::shared_ptr<KMesh> KRenderer::CreateSphereMesh(UINT32 Slices, UINT32 Stacks)
{
    if (!GraphicsDevice)
    {
        return nullptr;
    }

    return KMesh::CreateSphere(GraphicsDevice->GetDevice(), Slices, Stacks);
}

HRESULT KRenderer::InitializeDefaultResources()
{
    // Create basic shader program
    BasicShader = std::make_shared<KShaderProgram>();
    HRESULT hr = BasicShader->CreateBasicColorShader(GraphicsDevice->GetDevice());
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Basic shader program creation failed");
        return hr;
    }

    // Initialize texture manager
    hr = TextureManager.CreateDefaultTextures(GraphicsDevice->GetDevice());
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Default texture creation failed");
        return hr;
    }

    LOG_INFO("Default resources initialized successfully");
    return S_OK;
} 