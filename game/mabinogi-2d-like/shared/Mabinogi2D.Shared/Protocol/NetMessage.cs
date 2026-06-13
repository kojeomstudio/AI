using MessagePack;

namespace Mabinogi2D.Shared.Protocol;

/// <summary>
/// 클라이언트와 서버가 주고받는 모든 실시간 메시지의 종류.
/// 와이어에는 byte 하나로 직렬화되어 디스패치 헤더로 쓰인다.
/// </summary>
public enum MessageType : byte
{
    // client -> server
    JoinRequest = 1,
    MoveInput = 2,
    AttackInput = 3,
    ChatSend = 4,
    Ping = 5,

    // server -> client
    JoinAccepted = 64,
    WorldSnapshot = 65,
    EntitySpawned = 66,
    EntityDespawned = 67,
    CombatEvent = 68,
    ChatBroadcast = 69,
    Pong = 70,
    Error = 127,
}

/// <summary>
/// 모든 메시지의 봉투. 첫 바이트로 종류를 식별하고 Payload는 해당 타입의 MessagePack 바이트.
/// 이렇게 하면 디스패처가 전체를 역직렬화하지 않고도 라우팅할 수 있다.
/// </summary>
[MessagePackObject]
public sealed class Envelope
{
    [Key(0)] public MessageType Type { get; set; }
    [Key(1)] public byte[] Payload { get; set; } = System.Array.Empty<byte>();
}
