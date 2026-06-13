using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Threading.Channels;
using Mabinogi2D.Shared.Protocol;

namespace Mabinogi2D.Server.Game;

/// <summary>접속한 플레이어 하나의 연결·상태. 동시성 경계가 명확하도록 소유 스레드를 주석으로 표기.</summary>
public sealed class PlayerSession
{
    public required long EntityId { get; init; }
    public required long CharacterId { get; init; }
    public required long AccountId { get; init; }
    public required string Name { get; init; }
    public required WebSocket Socket { get; init; }

    // 위치/HP/Exp는 틱 스레드만 쓴다 (단일 소유자 → 락 불필요)
    public float X;
    public float Y;
    public int Hp = 100;
    public int MaxHp = 100;
    public long Exp;

    // 수신 스레드가 최신 입력으로 갈아끼우고 틱 스레드가 읽는다 (참조 대입은 원자적)
    public volatile MoveInput? PendingMove;

    // 공격은 이산 이벤트라 큐로 받는다(수신 스레드 enqueue, 틱 스레드 dequeue). 값=대상 EntityId.
    public readonly ConcurrentQueue<long> PendingAttacks = new();

    // 다음 공격 가능 게임시간(초). 틱 스레드 단독 소유(쿨다운).
    public double NextAttackTime;

    // 틱 스레드가 스냅샷을 넣고 송신 펌프가 빼서 소켓에 쓴다.
    // 가득 차면 오래된 것부터 버려 느린 클라이언트가 서버를 막지 않게 한다.
    public readonly Channel<byte[]> Outgoing = Channel.CreateBounded<byte[]>(
        new BoundedChannelOptions(64) { FullMode = BoundedChannelFullMode.DropOldest });

    public void Enqueue(byte[] data) => Outgoing.Writer.TryWrite(data);
}
