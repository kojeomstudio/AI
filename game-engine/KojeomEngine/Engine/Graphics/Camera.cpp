#include "Camera.h"

KCamera::KCamera()
    : Position(0.0f, 0.0f, 0.0f)
    , Rotation(0.0f, 0.0f, 0.0f)
    , Forward(0.0f, 0.0f, 1.0f)
    , Right(1.0f, 0.0f, 0.0f)
    , Up(0.0f, 1.0f, 0.0f)
    , FovY(XM_PIDIV4)
    , AspectRatio(16.0f / 9.0f)
    , NearZ(0.1f)
    , FarZ(1000.0f)
    , OrthoWidth(1.0f)
    , OrthoHeight(1.0f)
    , ProjectionType(ECameraProjectionType::Perspective)
    , bViewMatrixDirty(true)
    , bProjectionMatrixDirty(true)
{
    ViewMatrix = XMMatrixIdentity();
    ProjectionMatrix = XMMatrixIdentity();
}

void KCamera::SetPosition(const XMFLOAT3& InPosition)
{
    Position = InPosition;
    bViewMatrixDirty = true;
}

void KCamera::SetPosition(float X, float Y, float Z)
{
    SetPosition(XMFLOAT3(X, Y, Z));
}

void KCamera::SetRotation(const XMFLOAT3& InRotation)
{
    Rotation = InRotation;
    UpdateVectors();
    bViewMatrixDirty = true;
}

void KCamera::SetRotation(float Pitch, float Yaw, float Roll)
{
    SetRotation(XMFLOAT3(Pitch, Yaw, Roll));
}

void KCamera::LookAt(const XMFLOAT3& Target, const XMFLOAT3& InUp)
{
    XMVECTOR PositionVec = XMLoadFloat3(&Position);
    XMVECTOR TargetVec = XMLoadFloat3(&Target);
    XMVECTOR UpVec = XMLoadFloat3(&InUp);

    // Calculate direction vectors
    XMVECTOR ForwardVec = XMVector3Normalize(XMVectorSubtract(TargetVec, PositionVec));
    XMVECTOR RightVec = XMVector3Normalize(XMVector3Cross(UpVec, ForwardVec));
    UpVec = XMVector3Cross(ForwardVec, RightVec);

    // Store direction vectors
    XMStoreFloat3(&Forward, ForwardVec);
    XMStoreFloat3(&Right, RightVec);
    XMStoreFloat3(&Up, UpVec);

    // Calculate rotation from direction vectors
    float Pitch = asinf(-Forward.y);
    float Yaw = atan2f(Forward.x, Forward.z);
    Rotation = XMFLOAT3(Pitch, Yaw, 0.0f);

    bViewMatrixDirty = true;
}

void KCamera::SetPerspective(float InFovY, float InAspectRatio, float InNearZ, float InFarZ)
{
    FovY = InFovY;
    AspectRatio = InAspectRatio;
    NearZ = InNearZ;
    FarZ = InFarZ;
    ProjectionType = ECameraProjectionType::Perspective;
    bProjectionMatrixDirty = true;
}

void KCamera::SetOrthographic(float Width, float Height, float InNearZ, float InFarZ)
{
    OrthoWidth = Width;
    OrthoHeight = Height;
    NearZ = InNearZ;
    FarZ = InFarZ;
    ProjectionType = ECameraProjectionType::Orthographic;
    bProjectionMatrixDirty = true;
}

void KCamera::UpdateMatrices()
{
    if (bViewMatrixDirty)
    {
        XMVECTOR PositionVec = XMLoadFloat3(&Position);
        XMVECTOR ForwardVec = XMLoadFloat3(&Forward);
        XMVECTOR UpVec = XMLoadFloat3(&Up);

        XMVECTOR TargetVec = XMVectorAdd(PositionVec, ForwardVec);
        ViewMatrix = XMMatrixLookAtLH(PositionVec, TargetVec, UpVec);

        bViewMatrixDirty = false;
    }

    if (bProjectionMatrixDirty)
    {
        if (ProjectionType == ECameraProjectionType::Perspective)
        {
            ProjectionMatrix = XMMatrixPerspectiveFovLH(FovY, AspectRatio, NearZ, FarZ);
        }
        else
        {
            ProjectionMatrix = XMMatrixOrthographicLH(OrthoWidth, OrthoHeight, NearZ, FarZ);
        }
        bProjectionMatrixDirty = false;
    }
}

void KCamera::Move(const XMFLOAT3& Offset)
{
    XMVECTOR PositionVec = XMLoadFloat3(&Position);
    XMVECTOR OffsetVec = XMLoadFloat3(&Offset);
    PositionVec = XMVectorAdd(PositionVec, OffsetVec);
    XMStoreFloat3(&Position, PositionVec);
    bViewMatrixDirty = true;
}

void KCamera::Rotate(const XMFLOAT3& DeltaRotation)
{
    Rotation.x += DeltaRotation.x;
    Rotation.y += DeltaRotation.y;
    Rotation.z += DeltaRotation.z;

    // Clamp pitch to prevent gimbal lock
    if (Rotation.x > XM_PIDIV2 - 0.01f)
        Rotation.x = XM_PIDIV2 - 0.01f;
    if (Rotation.x < -XM_PIDIV2 + 0.01f)
        Rotation.x = -XM_PIDIV2 + 0.01f;

    UpdateVectors();
    bViewMatrixDirty = true;
}

void KCamera::UpdateVectors()
{
    // Calculate forward vector from rotation
    XMMATRIX RotationMatrix = XMMatrixRotationRollPitchYaw(Rotation.x, Rotation.y, Rotation.z);
    
    XMVECTOR ForwardVec = XMVector3TransformNormal(XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f), RotationMatrix);
    XMVECTOR RightVec = XMVector3TransformNormal(XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f), RotationMatrix);
    XMVECTOR UpVec = XMVector3TransformNormal(XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), RotationMatrix);

    XMStoreFloat3(&Forward, ForwardVec);
    XMStoreFloat3(&Right, RightVec);
    XMStoreFloat3(&Up, UpVec);
} 