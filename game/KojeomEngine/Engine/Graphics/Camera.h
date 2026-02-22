#pragma once

#include "../Utils/Common.h"

/**
 * @brief 3D Camera class
 * 
 * Manages view and projection matrices,
 * handles camera position, rotation, and projection settings.
 */
enum class ECameraProjectionType
{
    Perspective,
    Orthographic
};

class KCamera
{
public:
    KCamera();
    ~KCamera() = default;

    /**
     * @brief Set camera position
     * @param InPosition Camera position
     */
    void SetPosition(const XMFLOAT3& InPosition);
    void SetPosition(float X, float Y, float Z);

    /**
     * @brief Set camera rotation (Euler angles)
     * @param InRotation Rotation angles (radians)
     */
    void SetRotation(const XMFLOAT3& InRotation);
    void SetRotation(float Pitch, float Yaw, float Roll);

    /**
     * @brief Set camera to look at a specific point
     * @param Target Point to look at
     * @param Up Up direction vector (default: (0, 1, 0))
     */
    void LookAt(const XMFLOAT3& Target, const XMFLOAT3& Up = XMFLOAT3(0.0f, 1.0f, 0.0f));

    /**
     * @brief Set perspective projection
     * @param FovY Vertical field of view (radians)
     * @param AspectRatio Aspect ratio
     * @param NearZ Near clipping plane
     * @param FarZ Far clipping plane
     */
    void SetPerspective(float FovY, float AspectRatio, float NearZ, float FarZ);

    /**
     * @brief Set orthographic projection
     * @param Width Projection width
     * @param Height Projection height
     * @param NearZ Near clipping plane
     * @param FarZ Far clipping plane
     */
    void SetOrthographic(float Width, float Height, float NearZ, float FarZ);

    /**
     * @brief Update camera matrices
     * Must be called when position or rotation is changed
     */
    void UpdateMatrices();

    /**
     * @brief Move camera (relative)
     * @param Offset Movement vector
     */
    void Move(const XMFLOAT3& Offset);

    /**
     * @brief Rotate camera (relative)
     * @param DeltaRotation Rotation change (radians)
     */
    void Rotate(const XMFLOAT3& DeltaRotation);

    // Accessors
    const XMMATRIX& GetViewMatrix() const { return ViewMatrix; }
    const XMMATRIX& GetProjectionMatrix() const { return ProjectionMatrix; }
    
    const XMFLOAT3& GetPosition() const { return Position; }
    const XMFLOAT3& GetRotation() const { return Rotation; }
    
    XMFLOAT3 GetForward() const { return Forward; }
    XMFLOAT3 GetRight() const { return Right; }
    XMFLOAT3 GetUp() const { return Up; }

    float GetFovY() const { return FovY; }
    float GetAspectRatio() const { return AspectRatio; }
    float GetNearZ() const { return NearZ; }
    float GetFarZ() const { return FarZ; }
    float GetOrthoWidth() const { return OrthoWidth; }
    float GetOrthoHeight() const { return OrthoHeight; }
    ECameraProjectionType GetProjectionType() const { return ProjectionType; }

private:
    /**
     * @brief Update direction vectors
     */
    void UpdateVectors();

private:
    // Camera transform
    XMFLOAT3 Position;       // Position
    XMFLOAT3 Rotation;       // Rotation (pitch, yaw, roll)
    
    // Direction vectors
    XMFLOAT3 Forward;        // Forward vector
    XMFLOAT3 Right;          // Right vector
    XMFLOAT3 Up;             // Up vector

    // Matrices
    XMMATRIX ViewMatrix;
    XMMATRIX ProjectionMatrix;

    // Projection settings
    float FovY;              // Vertical field of view
    float AspectRatio;       // Aspect ratio
    float NearZ;             // Near clipping plane
    float FarZ;              // Far clipping plane
    float OrthoWidth;        // Orthographic width
    float OrthoHeight;       // Orthographic height
    ECameraProjectionType ProjectionType; // Current projection type

    // Flags
    bool bViewMatrixDirty;   // Whether view matrix needs update
    bool bProjectionMatrixDirty; // Whether projection matrix needs update
}; 