#pragma once

#include "../Utils/Common.h"
#include "../Utils/Logger.h"
#include "GraphicsDevice.h"
#include "Camera.h"
#include "Shader.h"
#include "Mesh.h"
#include "Texture.h"

/**
 * @brief Render object containing all rendering components
 */
struct FRenderObject
{
    std::shared_ptr<KMesh> Mesh;
    std::shared_ptr<KShaderProgram> Shader;
    std::shared_ptr<KTexture> Texture;
    XMMATRIX WorldMatrix = XMMatrixIdentity();
    
    FRenderObject(std::shared_ptr<KMesh> InMesh, std::shared_ptr<KShaderProgram> InShader, std::shared_ptr<KTexture> InTexture = nullptr)
        : Mesh(InMesh), Shader(InShader), Texture(InTexture) {}
};

/**
 * @brief Main renderer class
 * 
 * Manages all graphics components in an integrated manner
 * and provides a unified rendering pipeline.
 */
class KRenderer
{
public:
    KRenderer() = default;
    ~KRenderer() = default;

    // Prevent copying and moving
    KRenderer(const KRenderer&) = delete;
    KRenderer& operator=(const KRenderer&) = delete;

    /**
     * @brief Initialize renderer
     * @param InGraphicsDevice Graphics device
     * @return S_OK on success
     */
    HRESULT Initialize(KGraphicsDevice* InGraphicsDevice);

    /**
     * @brief Begin rendering frame
     * @param InCamera Camera
     * @param ClearColor Clear color
     */
    void BeginFrame(KCamera* InCamera, const float ClearColor[4] = Colors::CornflowerBlue);

    /**
     * @brief End rendering frame
     * @param bVSync Use V-Sync
     */
    void EndFrame(bool bVSync = true);

    /**
     * @brief Render object
     * @param RenderObject Object to render
     */
    void RenderObject(const FRenderObject& RenderObject);

    /**
     * @brief Render mesh (simple version)
     * @param InMesh Mesh
     * @param WorldMatrix World matrix
     * @param InTexture Texture (optional)
     */
    void RenderMesh(std::shared_ptr<KMesh> InMesh, const XMMATRIX& WorldMatrix, 
                   std::shared_ptr<KTexture> InTexture = nullptr);

    /**
     * @brief Render mesh with basic shader
     * @param InMesh Mesh
     * @param WorldMatrix World matrix
     */
    void RenderMeshBasic(std::shared_ptr<KMesh> InMesh, const XMMATRIX& WorldMatrix);

    /**
     * @brief Cleanup resources
     */
    void Cleanup();

    // Getters
    KShaderProgram* GetBasicShader() const { return BasicShader.get(); }
    KTextureManager* GetTextureManager() { return &TextureManager; }

    // Basic mesh factory methods
    std::shared_ptr<KMesh> CreateTriangleMesh();
    std::shared_ptr<KMesh> CreateQuadMesh();
    std::shared_ptr<KMesh> CreateCubeMesh();
    std::shared_ptr<KMesh> CreateSphereMesh(UINT32 Slices = 16, UINT32 Stacks = 16);

private:
    /**
     * @brief Initialize default resources
     */
    HRESULT InitializeDefaultResources();

private:
    // Core components
    KGraphicsDevice* GraphicsDevice = nullptr;
    KCamera* CurrentCamera = nullptr;

    // Rendering resources
    std::shared_ptr<KShaderProgram> BasicShader;
    KTextureManager TextureManager;

    // Current frame state
    bool bInFrame = false;
}; 