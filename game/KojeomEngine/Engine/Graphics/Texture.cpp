#include "Texture.h"

// UTexture class implementation

HRESULT KTexture::LoadFromFile(ID3D11Device* Device, const std::wstring& Filename)
{
    // For now, return success (texture loading implementation can be added later)
    // This would typically use DirectXTex library or similar
    LOG_WARNING("Texture loading from file not yet implemented: " + StringUtils::WideToMultiByte(Filename));
    return S_OK;
}

HRESULT KTexture::CreateSolidColor(ID3D11Device* Device, UINT32 InWidth, UINT32 InHeight, const XMFLOAT4& Color)
{
    Width = InWidth;
    Height = InHeight;
    Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    // Create texture data
    std::vector<UINT32> TextureData(InWidth * InHeight);
    UINT32 ColorValue = 
        (static_cast<UINT32>(Color.w * 255) << 24) |  // Alpha
        (static_cast<UINT32>(Color.z * 255) << 16) |  // Blue
        (static_cast<UINT32>(Color.y * 255) << 8) |   // Green
        (static_cast<UINT32>(Color.x * 255));         // Red

    std::fill(TextureData.begin(), TextureData.end(), ColorValue);

    // Create texture description
    D3D11_TEXTURE2D_DESC TextureDesc = {};
    TextureDesc.Width = InWidth;
    TextureDesc.Height = InHeight;
    TextureDesc.MipLevels = 1;
    TextureDesc.ArraySize = 1;
    TextureDesc.Format = Format;
    TextureDesc.SampleDesc.Count = 1;
    TextureDesc.SampleDesc.Quality = 0;
    TextureDesc.Usage = D3D11_USAGE_DEFAULT;
    TextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    TextureDesc.CPUAccessFlags = 0;

    // Create subresource data
    D3D11_SUBRESOURCE_DATA InitData = {};
    InitData.pSysMem = TextureData.data();
    InitData.SysMemPitch = InWidth * sizeof(UINT32);
    InitData.SysMemSlicePitch = 0;

    // Create texture
    HRESULT hr = Device->CreateTexture2D(&TextureDesc, &InitData, &Texture);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Solid color texture creation failed");
        return hr;
    }

    // Create shader resource view
    hr = CreateShaderResourceView(Device);
    if (FAILED(hr))
    {
        return hr;
    }

    // Create sampler state
    hr = CreateSamplerState(Device);
    if (FAILED(hr))
    {
        return hr;
    }

    return S_OK;
}

HRESULT KTexture::CreateCheckerboard(ID3D11Device* Device, UINT32 InWidth, UINT32 InHeight,
                                     const XMFLOAT4& Color1, const XMFLOAT4& Color2, UINT32 CheckSize)
{
    Width = InWidth;
    Height = InHeight;
    Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    // Create texture data
    std::vector<UINT32> TextureData(InWidth * InHeight);

    UINT32 ColorValue1 = 
        (static_cast<UINT32>(Color1.w * 255) << 24) |
        (static_cast<UINT32>(Color1.z * 255) << 16) |
        (static_cast<UINT32>(Color1.y * 255) << 8) |
        (static_cast<UINT32>(Color1.x * 255));

    UINT32 ColorValue2 = 
        (static_cast<UINT32>(Color2.w * 255) << 24) |
        (static_cast<UINT32>(Color2.z * 255) << 16) |
        (static_cast<UINT32>(Color2.y * 255) << 8) |
        (static_cast<UINT32>(Color2.x * 255));

    // Generate checkerboard pattern
    for (UINT32 y = 0; y < InHeight; ++y)
    {
        for (UINT32 x = 0; x < InWidth; ++x)
        {
            bool Check = ((x / CheckSize) + (y / CheckSize)) % 2 == 0;
            TextureData[y * InWidth + x] = Check ? ColorValue1 : ColorValue2;
        }
    }

    // Create texture description
    D3D11_TEXTURE2D_DESC TextureDesc = {};
    TextureDesc.Width = InWidth;
    TextureDesc.Height = InHeight;
    TextureDesc.MipLevels = 1;
    TextureDesc.ArraySize = 1;
    TextureDesc.Format = Format;
    TextureDesc.SampleDesc.Count = 1;
    TextureDesc.SampleDesc.Quality = 0;
    TextureDesc.Usage = D3D11_USAGE_DEFAULT;
    TextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    TextureDesc.CPUAccessFlags = 0;

    // Create subresource data
    D3D11_SUBRESOURCE_DATA InitData = {};
    InitData.pSysMem = TextureData.data();
    InitData.SysMemPitch = InWidth * sizeof(UINT32);
    InitData.SysMemSlicePitch = 0;

    // Create texture
    HRESULT hr = Device->CreateTexture2D(&TextureDesc, &InitData, &Texture);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Checkerboard texture creation failed");
        return hr;
    }

    // Create shader resource view
    hr = CreateShaderResourceView(Device);
    if (FAILED(hr))
    {
        return hr;
    }

    // Create sampler state
    hr = CreateSamplerState(Device);
    if (FAILED(hr))
    {
        return hr;
    }

    return S_OK;
}

void KTexture::Bind(ID3D11DeviceContext* Context, UINT32 Slot) const
{
    if (ShaderResourceView)
    {
        Context->PSSetShaderResources(Slot, 1, ShaderResourceView.GetAddressOf());
    }

    if (SamplerState)
    {
        Context->PSSetSamplers(Slot, 1, SamplerState.GetAddressOf());
    }
}

void KTexture::Unbind(ID3D11DeviceContext* Context, UINT32 Slot) const
{
    ID3D11ShaderResourceView* NullSRV = nullptr;
    ID3D11SamplerState* NullSampler = nullptr;
    
    Context->PSSetShaderResources(Slot, 1, &NullSRV);
    Context->PSSetSamplers(Slot, 1, &NullSampler);
}

void KTexture::Cleanup()
{
    SamplerState.Reset();
    ShaderResourceView.Reset();
    Texture.Reset();
    Width = 0;
    Height = 0;
    Format = DXGI_FORMAT_R8G8B8A8_UNORM;
}

HRESULT KTexture::CreateShaderResourceView(ID3D11Device* Device)
{
    if (!Texture)
    {
        return E_FAIL;
    }

    HRESULT hr = Device->CreateShaderResourceView(Texture.Get(), nullptr, &ShaderResourceView);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Shader resource view creation failed");
        return hr;
    }

    return S_OK;
}

HRESULT KTexture::CreateSamplerState(ID3D11Device* Device)
{
    D3D11_SAMPLER_DESC SamplerDesc = {};
    SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    SamplerDesc.MipLODBias = 0.0f;
    SamplerDesc.MaxAnisotropy = 1;
    SamplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    SamplerDesc.BorderColor[0] = 0.0f;
    SamplerDesc.BorderColor[1] = 0.0f;
    SamplerDesc.BorderColor[2] = 0.0f;
    SamplerDesc.BorderColor[3] = 0.0f;
    SamplerDesc.MinLOD = 0.0f;
    SamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    HRESULT hr = Device->CreateSamplerState(&SamplerDesc, &SamplerState);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Sampler state creation failed");
        return hr;
    }

    return S_OK;
}

// UTextureManager class implementation

HRESULT KTextureManager::CreateDefaultTextures(ID3D11Device* Device)
{
    // Create white texture
    WhiteTexture = std::make_shared<KTexture>();
    HRESULT hr = WhiteTexture->CreateSolidColor(Device, 64, 64, XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f));
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "White texture creation failed");
        return hr;
    }

    // Create black texture
    BlackTexture = std::make_shared<KTexture>();
    hr = BlackTexture->CreateSolidColor(Device, 64, 64, XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f));
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Black texture creation failed");
        return hr;
    }

    // Create checkerboard texture
    CheckerboardTexture = std::make_shared<KTexture>();
    hr = CheckerboardTexture->CreateCheckerboard(Device, 128, 128, 
                                                 XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f), 
                                                 XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f), 
                                                 16);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Checkerboard texture creation failed");
        return hr;
    }

    LOG_INFO("Default textures created successfully");
    return S_OK;
}

void KTextureManager::Cleanup()
{
    // Cleanup default textures
    if (WhiteTexture)
    {
        WhiteTexture->Cleanup();
        WhiteTexture.reset();
    }

    if (BlackTexture)
    {
        BlackTexture->Cleanup();
        BlackTexture.reset();
    }

    if (CheckerboardTexture)
    {
        CheckerboardTexture->Cleanup();
        CheckerboardTexture.reset();
    }

    // Cleanup cached textures
    for (auto& Pair : TextureCache)
    {
        if (Pair.second)
        {
            Pair.second->Cleanup();
        }
    }
    TextureCache.clear();

    LOG_INFO("Texture manager cleanup completed");
}

std::shared_ptr<KTexture> KTextureManager::LoadTexture(ID3D11Device* Device, const std::wstring& Filename)
{
    // Check if texture is already loaded
    auto It = TextureCache.find(Filename);
    if (It != TextureCache.end())
    {
        return It->second;
    }

    // Load new texture
    auto NewTexture = std::make_shared<KTexture>();
    HRESULT hr = NewTexture->LoadFromFile(Device, Filename);
    if (FAILED(hr))
    {
        KLogger::HResultError(hr, "Texture loading failed: " + StringUtils::WideToMultiByte(Filename));
        return nullptr;
    }

    // Cache the texture
    TextureCache[Filename] = NewTexture;
    return NewTexture;
}