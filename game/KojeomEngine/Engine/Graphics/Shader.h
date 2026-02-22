#pragma once

#include "../Utils/Common.h"
#include "../Utils/Logger.h"

/**
 * @brief Shader type enumeration
 */
enum class EShaderType
{
    Vertex,
    Pixel,
    Geometry,
    Hull,
    Domain,
    Compute
};

/**
 * @brief Individual shader class
 * 
 * Manages individual shaders such as VS, PS, etc.
 */
class KShader
{
public:
    KShader() = default;
    ~KShader() = default;

    /**
     * @brief Load shader from file
     * @param device DirectX 11 device
     * @param filename Shader file path
     * @param entryPoint Entry point function name
     * @param type Shader type
     * @return Success: S_OK
     */
    HRESULT LoadFromFile(ID3D11Device* Device, const std::wstring& Filename, 
                        const std::string& EntryPoint, EShaderType Type);

    /**
     * @brief Compile shader from string
     * @param Device DirectX 11 device
     * @param Source Shader source code
     * @param EntryPoint Entry point function name
     * @param Type Shader type
     * @return Success: S_OK
     */
    HRESULT CompileFromString(ID3D11Device* Device, const std::string& Source,
                             const std::string& EntryPoint, EShaderType Type);

    /**
     * @brief Bind shader
     * @param Context DirectX 11 device context
     */
    void Bind(ID3D11DeviceContext* Context) const;

    /**
     * @brief Unbind shader
     * @param Context DirectX 11 device context
     */
    void Unbind(ID3D11DeviceContext* Context) const;

    // Getters
    ID3D11VertexShader* GetVertexShader() const { return VertexShader.Get(); }
    ID3D11PixelShader* GetPixelShader() const { return PixelShader.Get(); }
    ID3D11GeometryShader* GetGeometryShader() const { return GeometryShader.Get(); }
    ID3DBlob* GetBlob() const { return Blob.Get(); }
    EShaderType GetType() const { return Type; }

private:
    /**
     * @brief Get profile string corresponding to shader type
     */
    std::string GetProfileString(EShaderType InType) const;

    /**
     * @brief Create shader object from compiled bytecode blob
     */
    HRESULT CreateShaderFromBlob(ID3D11Device* Device);

private:
    EShaderType Type = EShaderType::Vertex;
    ComPtr<ID3DBlob> Blob;

    // Shader objects (only one is used depending on type)
    ComPtr<ID3D11VertexShader> VertexShader;
    ComPtr<ID3D11PixelShader> PixelShader;
    ComPtr<ID3D11GeometryShader> GeometryShader;
    ComPtr<ID3D11HullShader> HullShader;
    ComPtr<ID3D11DomainShader> DomainShader;
    ComPtr<ID3D11ComputeShader> ComputeShader;
};

/**
 * @brief Shader program class
 * 
 * Manages shader programs that combine multiple shaders (VS+PS, etc.).
 */
class KShaderProgram
{
public:
    KShaderProgram() = default;
    ~KShaderProgram() = default;

    /**
     * @brief Create basic colored shader program
     * @param Device DirectX 11 device
     * @return Success: S_OK
     */
    HRESULT CreateBasicColorShader(ID3D11Device* Device);

    /**
     * @brief Create Phong lighting shader program
     *
     * Uses cbuffer b0 (World/View/Projection) and b1 (LightBuffer).
     * Vertex layout: POSITION(float3), COLOR(float4), NORMAL(float3)
     * @param Device DirectX 11 device
     * @return Success: S_OK
     */
    HRESULT CreatePhongShader(ID3D11Device* Device);

    /**
     * @brief Add shader
     * @param Shader Shader to add
     */
    void AddShader(std::shared_ptr<KShader> Shader);

    /**
     * @brief Create input layout
     * @param Device DirectX 11 device
     * @param InputElements Input element array
     * @param NumElements Number of input elements
     * @return Success: S_OK
     */
    HRESULT CreateInputLayout(ID3D11Device* Device, 
                             const D3D11_INPUT_ELEMENT_DESC* InputElements, 
                             UINT32 NumElements);

    /**
     * @brief Bind shader program
     * @param Context DirectX 11 device context
     */
    void Bind(ID3D11DeviceContext* Context) const;

    /**
     * @brief Unbind shader program
     * @param Context DirectX 11 device context
     */
    void Unbind(ID3D11DeviceContext* Context) const;

    // Getters
    ID3D11InputLayout* GetInputLayout() const { return InputLayout.Get(); }
    std::shared_ptr<KShader> GetShader(EShaderType Type) const;

private:
    std::vector<std::shared_ptr<KShader>> Shaders;
    ComPtr<ID3D11InputLayout> InputLayout;
};
