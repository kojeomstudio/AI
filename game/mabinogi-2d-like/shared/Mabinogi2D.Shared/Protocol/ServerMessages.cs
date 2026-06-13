using MessagePack;

namespace Mabinogi2D.Shared.Protocol;

/// <summary>입장 수락. 내 엔티티 id와 현재 맵을 알려준다.</summary>
[MessagePackObject]
public sealed class JoinAccepted
{
    [Key(0)] public long SelfEntityId { get; set; }
    [Key(1)] public int MapId { get; set; }
    [Key(2)] public int ServerTickRate { get; set; }
}

/// <summary>한 엔티티의 권위 상태. 스냅샷에 묶여 브로드캐스트된다.</summary>
[MessagePackObject]
public sealed class EntityState
{
    [Key(0)] public long EntityId { get; set; }
    [Key(1)] public float X { get; set; }
    [Key(2)] public float Y { get; set; }
    [Key(3)] public int Hp { get; set; }
    [Key(4)] public int MaxHp { get; set; }
    [Key(5)] public EntityKind Kind { get; set; }
    [Key(6)] public string Name { get; set; } = "";
}

public enum EntityKind : byte { Player = 0, Monster = 1, Npc = 2 }

/// <summary>관심영역(AOI) 내 엔티티들의 주기적 스냅샷. 틱마다 발행.</summary>
[MessagePackObject]
public sealed class WorldSnapshot
{
    [Key(0)] public uint ServerTick { get; set; }
    [Key(1)] public EntityState[] Entities { get; set; } = System.Array.Empty<EntityState>();
}

[MessagePackObject]
public sealed class EntitySpawned
{
    [Key(0)] public EntityState Entity { get; set; } = new();
}

[MessagePackObject]
public sealed class EntityDespawned
{
    [Key(0)] public long EntityId { get; set; }
}

[MessagePackObject]
public sealed class CombatEvent
{
    [Key(0)] public long AttackerId { get; set; }
    [Key(1)] public long TargetId { get; set; }
    [Key(2)] public int Damage { get; set; }
    [Key(3)] public bool TargetDied { get; set; }
}

[MessagePackObject]
public sealed class ChatBroadcast
{
    [Key(0)] public string SenderName { get; set; } = "";
    [Key(1)] public string Text { get; set; } = "";
}

[MessagePackObject]
public sealed class Pong
{
    [Key(0)] public long ClientTimeMs { get; set; }
    [Key(1)] public long ServerTimeMs { get; set; }
}

[MessagePackObject]
public sealed class ErrorMessage
{
    [Key(0)] public string Code { get; set; } = "";
    [Key(1)] public string Detail { get; set; } = "";
}
