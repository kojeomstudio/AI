namespace Mabinogi2D.Server.Game;

/// <summary>
/// MVP 몬스터: 고정 위치에서 대기하다 공격받으면 데미지를 입고, 죽으면 일정 시간 후 같은 자리에 부활.
/// 상태는 틱 스레드만 변경한다(단일 소유자).
/// </summary>
public sealed class Monster
{
    public required long EntityId { get; init; }
    public required string Name { get; init; }

    public float X;
    public float Y;
    public float SpawnX;
    public float SpawnY;

    public int Hp;
    public int MaxHp;

    public bool Alive => Hp > 0;

    /// <summary>죽었을 때 부활 예정 게임시간(초). Alive가 false일 때만 의미.</summary>
    public double RespawnAt;
}
