#pragma once

#include "../Utils/Common.h"

/**
 * @brief Directional light structure
 *
 * Represents a directional light with color and ambient terms.
 * Direction is the direction the light travels (surface to light = -Direction).
 */
struct FDirectionalLight
{
    XMFLOAT3 Direction   = {  0.5f, -1.0f,  0.5f };
    float    Padding0    = 0.0f;
    XMFLOAT4 Color       = {  1.0f,  0.95f, 0.9f, 1.0f };
    XMFLOAT4 AmbientColor= {  0.15f, 0.15f, 0.2f, 1.0f };
};

/**
 * @brief GPU-side light constant buffer (register b1)
 *
 * Must be 16-byte aligned. Total size: 64 bytes.
 */
struct FLightBuffer
{
    XMFLOAT3 LightDirection;   // 12 bytes
    float    Padding0;         //  4 bytes
    XMFLOAT4 LightColor;       // 16 bytes
    XMFLOAT4 AmbientColor;     // 16 bytes
    XMFLOAT3 CameraPosition;   // 12 bytes
    float    Padding1;         //  4 bytes
    // Total: 64 bytes
};
static_assert(sizeof(FLightBuffer) % 16 == 0, "FLightBuffer must be 16-byte aligned");
