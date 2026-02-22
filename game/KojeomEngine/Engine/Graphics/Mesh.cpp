#include "Mesh.h"
#include <cmath>

HRESULT KMesh::Initialize(ID3D11Device* Device, 
                        const FVertex* Vertices, UINT32 VertexCount,
                        const UINT32* Indices, UINT32 IndexCount)
{
    this->VertexCount = VertexCount;
    this->IndexCount = IndexCount;

    // Create vertex buffer
    HRESULT hr = CreateVertexBuffer(Device, Vertices, VertexCount);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Vertex buffer creation failed");
        return hr;
    }

    // Create index buffer (if indices exist)
    if (Indices && IndexCount > 0)
    {
        hr = CreateIndexBuffer(Device, Indices, IndexCount);
        if (FAILED(hr))
        {
            KLogger::HResultError(hr, "Index buffer creation failed");
            return hr;
        }
    }

    // Create constant buffer
    hr = CreateConstantBuffer(Device);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Constant buffer creation failed");
        return hr;
    }

    LOG_INFO("Mesh initialization completed, vertices: " + std::to_string(VertexCount) + 
             ", indices: " + std::to_string(IndexCount));
    return S_OK;
}

void KMesh::Render(ID3D11DeviceContext* Context)
{
    // Set vertex buffer
    UINT32 Stride = sizeof(FVertex);
    UINT32 Offset = 0;
    Context->IASetVertexBuffers(0, 1, VertexBuffer.GetAddressOf(), &Stride, &Offset);

    // Set index buffer (if exists)
    if (HasIndices())
    {
        Context->IASetIndexBuffer(IndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    }

    // Set primitive topology
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Bind constant buffer
    Context->VSSetConstantBuffers(0, 1, ConstantBuffer.GetAddressOf());

    // Draw
    if (HasIndices())
    {
        Context->DrawIndexed(IndexCount, 0, 0);
    }
    else
    {
        Context->Draw(VertexCount, 0);
    }
}

void KMesh::UpdateConstantBuffer(ID3D11DeviceContext* Context,
                               const XMMATRIX& WorldMatrix,
                               const XMMATRIX& ViewMatrix,
                               const XMMATRIX& ProjMatrix)
{
    FConstantBuffer CB;
    CB.WorldMatrix = XMMatrixTranspose(WorldMatrix);
    CB.ViewMatrix = XMMatrixTranspose(ViewMatrix);
    CB.ProjectionMatrix = XMMatrixTranspose(ProjMatrix);

    Context->UpdateSubresource(ConstantBuffer.Get(), 0, nullptr, &CB, 0, 0);
}

void KMesh::Cleanup()
{
    ConstantBuffer.Reset();
    IndexBuffer.Reset();
    VertexBuffer.Reset();
    
    VertexCount = 0;
    IndexCount = 0;
}

HRESULT KMesh::CreateVertexBuffer(ID3D11Device* Device, const FVertex* Vertices, UINT32 VertexCount)
{
    D3D11_BUFFER_DESC BufferDesc = {};
    BufferDesc.Usage = D3D11_USAGE_DEFAULT;
    BufferDesc.ByteWidth = sizeof(FVertex) * VertexCount;
    BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    BufferDesc.CPUAccessFlags = 0;

    D3D11_SUBRESOURCE_DATA InitData = {};
    InitData.pSysMem = Vertices;

    return Device->CreateBuffer(&BufferDesc, &InitData, &VertexBuffer);
}

HRESULT KMesh::CreateIndexBuffer(ID3D11Device* Device, const UINT32* Indices, UINT32 IndexCount)
{
    D3D11_BUFFER_DESC BufferDesc = {};
    BufferDesc.Usage = D3D11_USAGE_DEFAULT;
    BufferDesc.ByteWidth = sizeof(UINT32) * IndexCount;
    BufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    BufferDesc.CPUAccessFlags = 0;

    D3D11_SUBRESOURCE_DATA InitData = {};
    InitData.pSysMem = Indices;

    return Device->CreateBuffer(&BufferDesc, &InitData, &IndexBuffer);
}

HRESULT KMesh::CreateConstantBuffer(ID3D11Device* Device)
{
    D3D11_BUFFER_DESC BufferDesc = {};
    BufferDesc.Usage = D3D11_USAGE_DEFAULT;
    BufferDesc.ByteWidth = sizeof(FConstantBuffer);
    BufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    BufferDesc.CPUAccessFlags = 0;

    return Device->CreateBuffer(&BufferDesc, nullptr, &ConstantBuffer);
}

// Static factory methods implementation

std::unique_ptr<KMesh> KMesh::CreateTriangle(ID3D11Device* Device)
{
    // Triangle vertex data
    FVertex Vertices[] = {
        FVertex(XMFLOAT3(0.0f, 0.5f, 0.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f)),   // Red top
        FVertex(XMFLOAT3(0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f)),  // Green bottom right
        FVertex(XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f))  // Blue bottom left
    };

    auto Mesh = std::make_unique<KMesh>();
    HRESULT hr = Mesh->Initialize(Device, Vertices, 3);
    
    if (FAILED(hr))
    {
        LOG_ERROR("Triangle mesh creation failed");
        return nullptr;
    }

    return Mesh;
}

std::unique_ptr<KMesh> KMesh::CreateQuad(ID3D11Device* Device)
{
    // Quad vertex data
    FVertex Vertices[] = {
        FVertex(XMFLOAT3(-0.5f, 0.5f, 0.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f)),   // Top left
        FVertex(XMFLOAT3(0.5f, 0.5f, 0.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f)),    // Top right
        FVertex(XMFLOAT3(0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f)),   // Bottom right
        FVertex(XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f))   // Bottom left
    };

    // Index data (two triangles forming a quad)
    UINT32 Indices[] = {
        0, 1, 2,  // First triangle
        0, 2, 3   // Second triangle
    };

    auto Mesh = std::make_unique<KMesh>();
    HRESULT hr = Mesh->Initialize(Device, Vertices, 4, Indices, 6);
    
    if (FAILED(hr))
    {
        LOG_ERROR("Quad mesh creation failed");
        return nullptr;
    }

    return Mesh;
}

std::unique_ptr<KMesh> KMesh::CreateCube(ID3D11Device* Device)
{
    // Cube vertex data (same as legacy PrimitiveModel but using new Vertex structure)
    FVertex Vertices[] = {
        // Top face
        FVertex(XMFLOAT3(-1.0f, 1.0f, -1.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f)),
        FVertex(XMFLOAT3(1.0f, 1.0f, -1.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f)),
        FVertex(XMFLOAT3(1.0f, 1.0f, 1.0f), XMFLOAT4(0.0f, 1.0f, 1.0f, 1.0f)),
        FVertex(XMFLOAT3(-1.0f, 1.0f, 1.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f)),
        
        // Bottom face
        FVertex(XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT4(1.0f, 0.0f, 1.0f, 1.0f)),
        FVertex(XMFLOAT3(1.0f, -1.0f, -1.0f), XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f)),
        FVertex(XMFLOAT3(1.0f, -1.0f, 1.0f), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f)),
        FVertex(XMFLOAT3(-1.0f, -1.0f, 1.0f), XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f))
    };

    // Cube index data (same as legacy)
    UINT32 Indices[] = {
        3,1,0,  2,1,3,  // Top
        0,5,4,  1,5,0,  // Front
        3,4,7,  0,4,3,  // Left
        1,6,5,  2,6,1,  // Right
        2,7,6,  3,7,2,  // Back
        6,4,5,  7,4,6,  // Bottom
    };

    auto Mesh = std::make_unique<KMesh>();
    HRESULT hr = Mesh->Initialize(Device, Vertices, 8, Indices, 36);
    
    if (FAILED(hr))
    {
        LOG_ERROR("Cube mesh creation failed");
        return nullptr;
    }

    return Mesh;
}

std::unique_ptr<KMesh> KMesh::CreateSphere(ID3D11Device* Device, UINT32 Slices, UINT32 Stacks)
{
    std::vector<FVertex> Vertices;
    std::vector<UINT32> Indices;

    // Sphere generation algorithm
    const float Radius = 1.0f;
    const float Pi = XM_PI;

    // Generate vertices
    for (UINT32 i = 0; i <= Stacks; ++i)
    {
        float StackAngle = Pi * i / Stacks - Pi / 2.0f; // -π/2 to π/2
        float XY = Radius * cosf(StackAngle);
        float Z = Radius * sinf(StackAngle);

        for (UINT32 j = 0; j <= Slices; ++j)
        {
            float SectorAngle = 2 * Pi * j / Slices; // 0 to 2π

            FVertex Vertex;
            Vertex.Position.x = XY * cosf(SectorAngle);
            Vertex.Position.y = Z;
            Vertex.Position.z = XY * sinf(SectorAngle);

            // Normal vector calculation
            Vertex.Normal = Vertex.Position;

            // Color based on position
            Vertex.Color.x = (Vertex.Position.x + 1.0f) * 0.5f;
            Vertex.Color.y = (Vertex.Position.y + 1.0f) * 0.5f;
            Vertex.Color.z = (Vertex.Position.z + 1.0f) * 0.5f;
            Vertex.Color.w = 1.0f;

            // Texture coordinates
            Vertex.TexCoord.x = (float)j / Slices;
            Vertex.TexCoord.y = (float)i / Stacks;

            Vertices.push_back(Vertex);
        }
    }

    // Generate indices
    for (UINT32 i = 0; i < Stacks; ++i)
    {
        UINT32 K1 = i * (Slices + 1);
        UINT32 K2 = K1 + Slices + 1;

        for (UINT32 j = 0; j < Slices; ++j, ++K1, ++K2)
        {
            if (i != 0)
            {
                Indices.push_back(K1);
                Indices.push_back(K2);
                Indices.push_back(K1 + 1);
            }

            if (i != (Stacks - 1))
            {
                Indices.push_back(K1 + 1);
                Indices.push_back(K2);
                Indices.push_back(K2 + 1);
            }
        }
    }

    auto Mesh = std::make_unique<KMesh>();
    HRESULT hr = Mesh->Initialize(Device, Vertices.data(), static_cast<UINT32>(Vertices.size()),
                                 Indices.data(), static_cast<UINT32>(Indices.size()));
    
    if (FAILED(hr))
    {
        LOG_ERROR("Sphere mesh creation failed");
        return nullptr;
    }

    return Mesh;
} 