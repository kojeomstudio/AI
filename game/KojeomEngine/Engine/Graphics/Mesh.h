#pragma once

#include "../Utils/Common.h"
#include "../Utils/Logger.h"

/**
 * @brief 3D Vertex structure
 */
struct FVertex
{
    XMFLOAT3 Position;  // Position in 3D space
    XMFLOAT4 Color;     // Vertex color
    XMFLOAT3 Normal;    // Normal vector
    XMFLOAT2 TexCoord;  // Texture coordinates

    FVertex() : Position(0.0f, 0.0f, 0.0f), Color(1.0f, 1.0f, 1.0f, 1.0f),
               Normal(0.0f, 1.0f, 0.0f), TexCoord(0.0f, 0.0f) {}

    FVertex(const XMFLOAT3& InPosition, const XMFLOAT4& InColor = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f))
        : Position(InPosition), Color(InColor), Normal(0.0f, 1.0f, 0.0f), TexCoord(0.0f, 0.0f) {}

    FVertex(const XMFLOAT3& InPosition, const XMFLOAT4& InColor,
            const XMFLOAT3& InNormal, const XMFLOAT2& InTexCoord)
        : Position(InPosition), Color(InColor), Normal(InNormal), TexCoord(InTexCoord) {}
};

/**
 * @brief Constant buffer structure
 */
struct FConstantBuffer
{
    XMMATRIX WorldMatrix;
    XMMATRIX ViewMatrix;
    XMMATRIX ProjectionMatrix;
};

/**
 * @brief 3D Mesh class
 * 
 * Manages vertex and index data for rendering 3D geometry.
 */
class KMesh
{
public:
    KMesh() = default;
    ~KMesh() = default;

    // Copy prevention, move allowed
    KMesh(const KMesh&) = delete;
    KMesh& operator=(const KMesh&) = delete;
    KMesh(KMesh&&) = default;
    KMesh& operator=(KMesh&&) = default;

    /**
     * @brief Initialize the mesh
     * @param Device DirectX 11 device
     * @param Vertices Vertex array
     * @param VertexCount Number of vertices
     * @param Indices Index array
     * @param IndexCount Number of indices
     * @return Success: S_OK
     */
    HRESULT Initialize(ID3D11Device* Device, 
                      const FVertex* Vertices, UINT32 VertexCount,
                      const UINT32* Indices = nullptr, UINT32 IndexCount = 0);

    /**
     * @brief Render the mesh
     * @param Context DirectX 11 device context
     */
    void Render(ID3D11DeviceContext* Context);

    /**
     * @brief Update constant buffer
     * @param Context DirectX 11 device context
     * @param WorldMatrix World transformation matrix
     * @param ViewMatrix View transformation matrix
     * @param ProjectionMatrix Projection transformation matrix
     */
    void UpdateConstantBuffer(ID3D11DeviceContext* Context,
                             const XMMATRIX& WorldMatrix,
                             const XMMATRIX& ViewMatrix,
                             const XMMATRIX& ProjectionMatrix);

    /**
     * @brief Cleanup resources
     */
    void Cleanup();

    // Accessors
    ID3D11Buffer* GetVertexBuffer() const { return VertexBuffer.Get(); }
    ID3D11Buffer* GetIndexBuffer() const { return IndexBuffer.Get(); }
    ID3D11Buffer* GetConstantBuffer() const { return ConstantBuffer.Get(); }
    
    UINT32 GetVertexCount() const { return VertexCount; }
    UINT32 GetIndexCount() const { return IndexCount; }
    bool HasIndices() const { return IndexCount > 0; }

    /**
     * @brief Static factory methods
     */
    static std::unique_ptr<KMesh> CreateTriangle(ID3D11Device* Device);
    static std::unique_ptr<KMesh> CreateQuad(ID3D11Device* Device);
    static std::unique_ptr<KMesh> CreateCube(ID3D11Device* Device);
    static std::unique_ptr<KMesh> CreateSphere(ID3D11Device* Device, UINT32 Slices = 16, UINT32 Stacks = 16);

private:
    /**
     * @brief Create vertex buffer
     */
    HRESULT CreateVertexBuffer(ID3D11Device* Device, const FVertex* Vertices, UINT32 VertexCount);

    /**
     * @brief Create index buffer
     */
    HRESULT CreateIndexBuffer(ID3D11Device* Device, const UINT32* Indices, UINT32 IndexCount);

    /**
     * @brief Create constant buffer
     */
    HRESULT CreateConstantBuffer(ID3D11Device* Device);

private:
    // DirectX resources
    ComPtr<ID3D11Buffer> VertexBuffer;
    ComPtr<ID3D11Buffer> IndexBuffer;
    ComPtr<ID3D11Buffer> ConstantBuffer;

    // Mesh information
    UINT32 VertexCount = 0;
    UINT32 IndexCount = 0;
}; 