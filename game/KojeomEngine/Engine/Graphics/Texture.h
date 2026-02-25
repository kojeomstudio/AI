#pragma once

#include "../Utils/Common.h"
#include "../Utils/Logger.h"

/**
 * @brief Texture class
 * 
 * Class for loading and managing 2D textures.
 */
class KTexture
{
public:
    KTexture() = default;
    ~KTexture() = default;

    // Prevent copying, allow moving
    KTexture(const KTexture&) = delete;
    KTexture& operator=(const KTexture&) = delete;
    KTexture(KTexture&&) = default;
    KTexture& operator=(KTexture&&) = default;

    /**
     * @brief Load texture from file
     * @param Device DirectX 11 device
     * @param Filename Texture file path
     * @return S_OK on success
     */
    HRESULT LoadFromFile(ID3D11Device* Device, const std::wstring& Filename);

    /**
     * @brief Create texture from memory (solid color texture, etc.)
     * @param Device DirectX 11 device
     * @param Width Texture width
     * @param Height Texture height
     * @param Color Color (RGBA)
     * @return S_OK on success
     */
    HRESULT CreateSolidColor(ID3D11Device* Device, UINT32 Width, UINT32 Height, const XMFLOAT4& Color);

    /**
     * @brief Create checkerboard pattern texture
     * @param Device DirectX 11 device
     * @param Width Texture width
     * @param Height Texture height
     * @param Color1 First color
     * @param Color2 Second color
     * @param CheckSize Check size
     * @return S_OK on success
     */
    HRESULT CreateCheckerboard(ID3D11Device* Device, UINT32 Width, UINT32 Height,
                              const XMFLOAT4& Color1, const XMFLOAT4& Color2, UINT32 CheckSize = 32);

    /**
     * @brief Bind texture (to make it usable in shaders)
     * @param Context DirectX 11 device context
     * @param Slot Texture slot number
     */
    void Bind(ID3D11DeviceContext* Context, UINT32 Slot = 0) const;

    /**
     * @brief Unbind texture
     * @param Context DirectX 11 device context
     * @param Slot Texture slot number
     */
    void Unbind(ID3D11DeviceContext* Context, UINT32 Slot = 0) const;

    /**
     * @brief Cleanup resources
     */
    void Cleanup();

    // Accessors
    ID3D11Texture2D* GetTexture() const { return Texture.Get(); }
    ID3D11ShaderResourceView* GetShaderResourceView() const { return ShaderResourceView.Get(); }
    ID3D11SamplerState* GetSamplerState() const { return SamplerState.Get(); }
    
    UINT32 GetWidth() const { return Width; }
    UINT32 GetHeight() const { return Height; }
    DXGI_FORMAT GetFormat() const { return Format; }

private:
    /**
     * @brief Create shader resource view
     */
    HRESULT CreateShaderResourceView(ID3D11Device* Device);

    /**
     * @brief Create sampler state
     */
    HRESULT CreateSamplerState(ID3D11Device* Device);

private:
    ComPtr<ID3D11Texture2D> Texture;
    ComPtr<ID3D11ShaderResourceView> ShaderResourceView;
    ComPtr<ID3D11SamplerState> SamplerState;

    UINT32 Width = 0;
    UINT32 Height = 0;
    DXGI_FORMAT Format = DXGI_FORMAT_R8G8B8A8_UNORM;
};

/**
 * @brief Texture manager class
 * 
 * Manager for loading and caching multiple textures.
 */
class KTextureManager
{
public:
    KTextureManager() = default;
    ~KTextureManager() = default;

    // Prevent copying and moving (singleton pattern)
    KTextureManager(const KTextureManager&) = delete;
    KTextureManager& operator=(const KTextureManager&) = delete;

    /**
     * @brief Load texture (with caching support)
     * @param Device DirectX 11 device
     * @param Filename Texture file path
     * @return Texture pointer (nullptr on failure)
     */
    std::shared_ptr<KTexture> LoadTexture(ID3D11Device* Device, const std::wstring& Filename);

    /**
     * @brief Create default textures
     * @param Device DirectX 11 device
     * @return S_OK on success
     */
    HRESULT CreateDefaultTextures(ID3D11Device* Device);

    /**
     * @brief Get default white texture
     */
    std::shared_ptr<KTexture> GetWhiteTexture() const { return WhiteTexture; }

    /**
     * @brief Get default black texture
     */
    std::shared_ptr<KTexture> GetBlackTexture() const { return BlackTexture; }

    /**
     * @brief Get default checkerboard texture
     */
    std::shared_ptr<KTexture> GetCheckerboardTexture() const { return CheckerboardTexture; }

    /**
     * @brief Cleanup all textures
     */
    void Cleanup();

private:
    std::unordered_map<std::wstring, std::shared_ptr<KTexture>> TextureCache;
    
    // Default textures
    std::shared_ptr<KTexture> WhiteTexture;
    std::shared_ptr<KTexture> BlackTexture;
    std::shared_ptr<KTexture> CheckerboardTexture;
}; 