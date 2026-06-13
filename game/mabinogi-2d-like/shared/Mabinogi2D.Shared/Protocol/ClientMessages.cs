using MessagePack;

namespace Mabinogi2D.Shared.Protocol;

/// <summary>WebSocket 연결 후 첫 메시지. JWT로 인증된 캐릭터로 월드에 입장 요청.</summary>
[MessagePackObject]
public sealed class JoinRequest
{
    [Key(0)] public string Token { get; set; } = "";   // Auth 서버가 발급한 JWT
    [Key(1)] public long CharacterId { get; set; }
}

/// <summary>
/// 이동 "의도"를 보낸다(목적지 좌표가 아니라 입력 방향).
/// 권위 서버가 검증·시뮬레이션하므로 클라이언트는 위치를 직접 지정하지 않는다.
/// </summary>
[MessagePackObject]
public sealed class MoveInput
{
    [Key(0)] public float DirX { get; set; }   // -1..1 정규화 방향
    [Key(1)] public float DirY { get; set; }
    [Key(2)] public uint ClientTick { get; set; }   // 클라 예측 보정용 시퀀스
}

[MessagePackObject]
public sealed class AttackInput
{
    [Key(0)] public long TargetEntityId { get; set; }
}

[MessagePackObject]
public sealed class ChatSend
{
    [Key(0)] public string Text { get; set; } = "";
}

[MessagePackObject]
public sealed class Ping
{
    [Key(0)] public long ClientTimeMs { get; set; }
}
