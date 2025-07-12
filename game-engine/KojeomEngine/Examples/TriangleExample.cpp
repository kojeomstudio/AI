#include "../Engine/Core/Engine.h"

// Simple vertex structure
struct SimpleVertex
{
    XMFLOAT3 Position;  // Position
    XMFLOAT4 Color;     // Color
};

// Constant buffer structure
struct ConstantBuffer
{
    XMMATRIX WorldMatrix;
    XMMATRIX ViewMatrix;
    XMMATRIX ProjectionMatrix;
};

/**
 * @brief Triangle rendering example
 * 
 * Example that renders a simple colored triangle on screen.
 * Demonstrates basic usage of vertex buffers, index buffers, and shaders.
 */
class TriangleExample : public KEngine
{
public:
    TriangleExample() = default;
    ~TriangleExample()
    {
        CleanupResources();
    }

    /**
     * @brief Resource initialization
     */
    HRESULT InitializeResources()
    {
        HRESULT hr = CreateShaders();
        if (FAILED(hr)) return hr;

        hr = CreateGeometry();
        if (FAILED(hr)) return hr;

        hr = CreateConstantBuffer();
        if (FAILED(hr)) return hr;

        return S_OK;
    }

    void Update(float deltaTime) override
    {
        KEngine::Update(deltaTime);

        // Rotate world matrix
        static float rotation = 0.0f;
        rotation += deltaTime;
        m_worldMatrix = XMMatrixRotationZ(rotation);
    }

    void Render() override
    {
        auto device = GetGraphicsDevice();
        if (!device) return;

        auto context = device->GetContext();

        // Begin frame
        float clearColor[4] = { 0.0f, 0.1f, 0.2f, 1.0f };
        device->BeginFrame(clearColor);

        // Render triangle
        RenderTriangle(context);

        // End frame
        device->EndFrame(true);
    }

private:
    HRESULT CreateShaders()
    {
        auto device = GetGraphicsDevice()->GetDevice();

        // Simple vertex shader HLSL code
        const char* vertexShaderSource = R"(
            cbuffer ConstantBuffer : register(b0)
            {
                matrix World;
                matrix View; 
                matrix Projection;
            }

            struct VS_INPUT
            {
                float4 Pos : POSITION;
                float4 Color : COLOR;
            };

            struct PS_INPUT
            {
                float4 Pos : SV_POSITION;
                float4 Color : COLOR;
            };

            PS_INPUT main(VS_INPUT input)
            {
                PS_INPUT output = (PS_INPUT)0;
                output.Pos = mul(input.Pos, World);
                output.Pos = mul(output.Pos, View);
                output.Pos = mul(output.Pos, Projection);
                output.Color = input.Color;
                return output;
            }
        )";

        // Simple pixel shader HLSL code
        const char* pixelShaderSource = R"(
            struct PS_INPUT
            {
                float4 Pos : SV_POSITION;
                float4 Color : COLOR;
            };

            float4 main(PS_INPUT input) : SV_Target
            {
                return input.Color;
            }
        )";

        // Compile vertex shader
        ComPtr<ID3DBlob> vsBlob, psBlob, errorBlob;
        HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), 
                               nullptr, nullptr, nullptr, "main", "vs_4_0", 
                               0, 0, &vsBlob, &errorBlob);
        
        if (FAILED(hr))
        {
            if (errorBlob)
            {
                LOG_ERROR("Vertex Shader Compile Error: " + 
                         std::string((char*)errorBlob->GetBufferPointer()));
            }
            return hr;
        }

        // Compile pixel shader
        hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), 
                       nullptr, nullptr, nullptr, "main", "ps_4_0", 
                       0, 0, &psBlob, &errorBlob);
        
        if (FAILED(hr))
        {
            if (errorBlob)
            {
                LOG_ERROR("Pixel Shader Compile Error: " + 
                         std::string((char*)errorBlob->GetBufferPointer()));
            }
            return hr;
        }

        // Create shader objects
        hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), 
                                       vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
        if (FAILED(hr)) return hr;

        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), 
                                      psBlob->GetBufferSize(), nullptr, &m_pixelShader);
        if (FAILED(hr)) return hr;

        // Create input layout
        D3D11_INPUT_ELEMENT_DESC layout[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        hr = device->CreateInputLayout(layout, ARRAYSIZE(layout), 
                                      vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), 
                                      &m_inputLayout);
        
        return hr;
    }

    HRESULT CreateGeometry()
    {
        auto device = GetGraphicsDevice()->GetDevice();

        // Triangle vertex data
        SimpleVertex vertices[] =
        {
            { XMFLOAT3(0.0f, 0.5f, 0.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) },  // Red top
            { XMFLOAT3(0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) }, // Green bottom-right
            { XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) } // Blue bottom-left
        };

        // Create vertex buffer
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.ByteWidth = sizeof(SimpleVertex) * 3;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.CPUAccessFlags = 0;

        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = vertices;

        HRESULT hr = device->CreateBuffer(&bd, &initData, &m_vertexBuffer);
        if (FAILED(hr)) return hr;

        // Index data
        WORD indices[] = { 0, 1, 2 };

        // Create index buffer
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.ByteWidth = sizeof(WORD) * 3;
        bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        bd.CPUAccessFlags = 0;
        initData.pSysMem = indices;

        hr = device->CreateBuffer(&bd, &initData, &m_indexBuffer);
        
        return hr;
    }

    HRESULT CreateConstantBuffer()
    {
        auto device = GetGraphicsDevice()->GetDevice();

        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.ByteWidth = sizeof(ConstantBuffer);
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = 0;

        return device->CreateBuffer(&bd, nullptr, &m_constantBuffer);
    }

    void RenderTriangle(ID3D11DeviceContext* context)
    {
        // Update constant buffer
        ConstantBuffer cb;
        cb.WorldMatrix = XMMatrixTranspose(m_worldMatrix);
        cb.ViewMatrix = XMMatrixTranspose(GetCamera()->GetViewMatrix());
        cb.ProjectionMatrix = XMMatrixTranspose(GetCamera()->GetProjectionMatrix());
        
        context->UpdateSubresource(m_constantBuffer.Get(), 0, nullptr, &cb, 0, 0);

        // Set vertex buffer
        UINT stride = sizeof(SimpleVertex);
        UINT offset = 0;
        context->IASetVertexBuffers(0, 1, m_vertexBuffer.GetAddressOf(), &stride, &offset);
        context->IASetIndexBuffer(m_indexBuffer.Get(), DXGI_FORMAT_R16_UINT, 0);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->IASetInputLayout(m_inputLayout.Get());

        // Set shaders
        context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
        context->VSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());
        context->PSSetShader(m_pixelShader.Get(), nullptr, 0);

        // Draw
        context->DrawIndexed(3, 0, 0);
    }

    void CleanupResources()
    {
        m_constantBuffer.Reset();
        m_indexBuffer.Reset();
        m_vertexBuffer.Reset();
        m_inputLayout.Reset();
        m_pixelShader.Reset();
        m_vertexShader.Reset();
    }

private:
    // Shader resources
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11InputLayout> m_inputLayout;

    // Geometry resources
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11Buffer> m_indexBuffer;
    ComPtr<ID3D11Buffer> m_constantBuffer;

    // Transformation matrix
    XMMATRIX m_worldMatrix = XMMatrixIdentity();
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

    // Use improved engine helper function (with custom initialization)
    return KEngine::RunApplication<TriangleExample>(
        hInstance,
        L"KojeomEngine - Triangle Example",
        1024, 768,
        [](TriangleExample* app) -> HRESULT {
            // Initialize resources
            HRESULT hr = app->InitializeResources();
            if (FAILED(hr)) return hr;
            
            // Setup camera
            app->GetCamera()->SetPosition(0.0f, 0.0f, -2.0f);
            
            return S_OK;
        }
    );
} 