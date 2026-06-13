using System.Collections.Concurrent;
using System.Net.WebSockets;
using Mabinogi2D.Server.Auth;
using Mabinogi2D.Server.Data;
using Mabinogi2D.Shared;
using Mabinogi2D.Shared.Protocol;
using Microsoft.EntityFrameworkCore;
using EntityState = Mabinogi2D.Shared.Protocol.EntityState;

namespace Mabinogi2D.Server.Game;

/// <summary>
/// 권위(authoritative) 월드. 단일 공유 맵, 플레이어 이동만 시뮬레이션(MVP).
/// 입력은 수신 스레드가 큐잉하고, 위치 갱신/스냅샷 발행은 <see cref="Tick"/>(틱 스레드)이 단독으로 한다.
/// </summary>
public sealed class GameWorld
{
    private readonly JwtService _jwt;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<GameWorld> _logger;

    private readonly ConcurrentDictionary<long, PlayerSession> _sessions = new();
    private long _nextEntityId;
    private uint _serverTick;

    public GameWorld(JwtService jwt, IServiceScopeFactory scopeFactory, ILogger<GameWorld> logger)
    {
        _jwt = jwt;
        _scopeFactory = scopeFactory;
        _logger = logger;
    }

    // ── 틱 루프 (GameLoop 호출) ───────────────────────────────────────────────
    public void Tick()
    {
        _serverTick++;
        const float dt = 1f / GameConstants.TickRate;

        // 1) 입력 적용 — 위치는 틱 스레드 단독 소유
        foreach (var s in _sessions.Values)
        {
            var mv = s.PendingMove;
            if (mv is null) continue;
            float len = MathF.Sqrt(mv.DirX * mv.DirX + mv.DirY * mv.DirY);
            if (len > 0.01f)
            {
                s.X += mv.DirX / len * GameConstants.PlayerMoveSpeed * dt;
                s.Y += mv.DirY / len * GameConstants.PlayerMoveSpeed * dt;
            }
        }

        // 2) 세션별 관심영역(AOI) 스냅샷 발행
        var sessions = _sessions.Values;
        const float aoiSq = GameConstants.AoiRadius * GameConstants.AoiRadius;
        foreach (var self in sessions)
        {
            var visible = new List<EntityState>();
            foreach (var other in sessions)
            {
                float dx = other.X - self.X, dy = other.Y - self.Y;
                if (dx * dx + dy * dy <= aoiSq)
                {
                    visible.Add(new EntityState
                    {
                        EntityId = other.EntityId,
                        X = other.X,
                        Y = other.Y,
                        Hp = other.Hp,
                        MaxHp = other.MaxHp,
                        Kind = EntityKind.Player,
                        Name = other.Name,
                    });
                }
            }
            self.Enqueue(ProtocolCodec.Encode(MessageType.WorldSnapshot, new WorldSnapshot
            {
                ServerTick = _serverTick,
                Entities = visible.ToArray(),
            }));
        }
    }

    // ── 연결 처리 (WebSocket 엔드포인트 호출) ─────────────────────────────────
    public async Task HandleConnectionAsync(WebSocket socket, CancellationToken ct)
    {
        // 1) 첫 메시지는 반드시 JoinRequest
        var first = await ReceiveAsync(socket, ct);
        if (first is null) return;

        var env = ProtocolCodec.DecodeEnvelope(first);
        if (env.Type != MessageType.JoinRequest)
        {
            await SendError(socket, "PROTOCOL", "첫 메시지는 JoinRequest여야 합니다.", ct);
            return;
        }

        var join = ProtocolCodec.DecodePayload<JoinRequest>(env);
        var accountId = _jwt.ValidateAndGetAccountId(join.Token);
        if (accountId is null)
        {
            await SendError(socket, "AUTH", "토큰이 유효하지 않습니다.", ct);
            return;
        }

        // 2) 캐릭터 로드 + 소유권 확인
        Character? ch;
        using (var scope = _scopeFactory.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<GameDbContext>();
            ch = await db.Characters.FirstOrDefaultAsync(c => c.Id == join.CharacterId, ct);
        }
        if (ch is null || ch.AccountId != accountId)
        {
            await SendError(socket, "CHAR", "캐릭터가 없거나 소유자가 아닙니다.", ct);
            return;
        }

        // 3) 세션 생성 및 등록
        var session = new PlayerSession
        {
            EntityId = Interlocked.Increment(ref _nextEntityId),
            CharacterId = ch.Id,
            AccountId = accountId.Value,
            Name = ch.Name,
            Socket = socket,
            X = ch.PosX,
            Y = ch.PosY,
            Hp = ch.Hp,
            MaxHp = ch.MaxHp,
        };
        _sessions[session.EntityId] = session;
        _logger.LogInformation("입장: {Name} (entity {Id}), 접속 {Count}명", session.Name, session.EntityId, _sessions.Count);

        session.Enqueue(ProtocolCodec.Encode(MessageType.JoinAccepted, new JoinAccepted
        {
            SelfEntityId = session.EntityId,
            MapId = ch.MapId,
            ServerTickRate = GameConstants.TickRate,
        }));

        // 4) 송신 펌프 + 수신 루프 동시 구동
        var sendPump = SendPumpAsync(session, ct);
        try
        {
            await ReceiveLoopAsync(session, ct);
        }
        finally
        {
            _sessions.TryRemove(session.EntityId, out _);
            session.Outgoing.Writer.TryComplete();
            try { await sendPump; } catch { /* ignore */ }
            await PersistAsync(session);
            _logger.LogInformation("퇴장: {Name}, 접속 {Count}명", session.Name, _sessions.Count);
        }
    }

    private async Task ReceiveLoopAsync(PlayerSession s, CancellationToken ct)
    {
        while (s.Socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
        {
            var data = await ReceiveAsync(s.Socket, ct);
            if (data is null) break;

            var env = ProtocolCodec.DecodeEnvelope(data);
            switch (env.Type)
            {
                case MessageType.MoveInput:
                    s.PendingMove = ProtocolCodec.DecodePayload<MoveInput>(env);
                    break;

                case MessageType.ChatSend:
                    var chat = ProtocolCodec.DecodePayload<ChatSend>(env);
                    Broadcast(ProtocolCodec.Encode(MessageType.ChatBroadcast, new ChatBroadcast
                    {
                        SenderName = s.Name,
                        Text = chat.Text,
                    }));
                    break;

                case MessageType.Ping:
                    var ping = ProtocolCodec.DecodePayload<Ping>(env);
                    s.Enqueue(ProtocolCodec.Encode(MessageType.Pong, new Pong
                    {
                        ClientTimeMs = ping.ClientTimeMs,
                        ServerTimeMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    }));
                    break;

                // AttackInput(전투)은 다음 증분에서 처리
            }
        }
    }

    private static async Task SendPumpAsync(PlayerSession s, CancellationToken ct)
    {
        try
        {
            await foreach (var data in s.Outgoing.Reader.ReadAllAsync(ct))
                await s.Socket.SendAsync(data, WebSocketMessageType.Binary, endOfMessage: true, ct);
        }
        catch
        {
            // 연결 종료/취소 — 정리는 HandleConnectionAsync의 finally가 담당
        }
    }

    private void Broadcast(byte[] data)
    {
        foreach (var s in _sessions.Values) s.Enqueue(data);
    }

    private async Task PersistAsync(PlayerSession s)
    {
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<GameDbContext>();
            var ch = await db.Characters.FirstOrDefaultAsync(c => c.Id == s.CharacterId);
            if (ch is null) return;
            ch.PosX = s.X;
            ch.PosY = s.Y;
            ch.Hp = s.Hp;
            await db.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "{Name} 상태 저장 실패", s.Name);
        }
    }

    // ── 저수준 WebSocket 헬퍼 ────────────────────────────────────────────────
    private static async Task<byte[]?> ReceiveAsync(WebSocket socket, CancellationToken ct)
    {
        var buffer = new byte[4096];
        using var ms = new MemoryStream();
        WebSocketReceiveResult result;
        do
        {
            try { result = await socket.ReceiveAsync(buffer, ct); }
            catch { return null; }
            if (result.MessageType == WebSocketMessageType.Close) return null;
            ms.Write(buffer, 0, result.Count);
        } while (!result.EndOfMessage);
        return ms.ToArray();
    }

    private static async Task SendError(WebSocket socket, string code, string detail, CancellationToken ct)
    {
        try
        {
            var bytes = ProtocolCodec.Encode(MessageType.Error, new ErrorMessage { Code = code, Detail = detail });
            await socket.SendAsync(bytes, WebSocketMessageType.Binary, true, ct);
        }
        catch { /* ignore */ }
    }
}
