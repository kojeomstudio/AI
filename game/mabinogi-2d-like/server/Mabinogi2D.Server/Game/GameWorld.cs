using System.Collections.Concurrent;
using System.Linq;
using System.Net.WebSockets;
using Mabinogi2D.Server.Auth;
using Mabinogi2D.Server.Data;
using Mabinogi2D.Shared;
using Mabinogi2D.Shared.Protocol;
using Microsoft.EntityFrameworkCore;
using EntityState = Mabinogi2D.Shared.Protocol.EntityState;

namespace Mabinogi2D.Server.Game;

/// <summary>
/// 권위(authoritative) 월드. 단일 공유 맵, 플레이어 이동 + 근접 전투(고정 몬스터)를 시뮬레이션(MVP).
/// 입력은 수신 스레드가 큐잉하고, 위치/전투/스냅샷은 <see cref="Tick"/>(틱 스레드)이 단독으로 처리한다.
/// </summary>
public sealed class GameWorld
{
    private readonly JwtService _jwt;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<GameWorld> _logger;

    private readonly ConcurrentDictionary<long, PlayerSession> _sessions = new();
    private readonly List<Monster> _monsters = new();   // 틱 스레드만 변경
    private long _nextEntityId;
    private uint _serverTick;
    private double _now;   // 누적 게임시간(초), 틱 스레드 단독 소유

    public GameWorld(JwtService jwt, IServiceScopeFactory scopeFactory, ILogger<GameWorld> logger)
    {
        _jwt = jwt;
        _scopeFactory = scopeFactory;
        _logger = logger;
        SpawnMonsters();
    }

    private void SpawnMonsters()
    {
        (float x, float y, string name)[] spawns =
        {
            (120f, 0f, "Slime"),
            (-120f, 60f, "Slime"),
            (0f, 140f, "Goblin"),
        };
        foreach (var (x, y, name) in spawns)
        {
            _monsters.Add(new Monster
            {
                EntityId = Interlocked.Increment(ref _nextEntityId),
                Name = name,
                X = x, Y = y, SpawnX = x, SpawnY = y,
                Hp = GameConstants.MonsterMaxHp,
                MaxHp = GameConstants.MonsterMaxHp,
            });
        }
        _logger.LogInformation("몬스터 {Count}마리 스폰", _monsters.Count);
    }

    // ── 틱 루프 (GameLoop 호출) ───────────────────────────────────────────────
    public void Tick()
    {
        _serverTick++;
        const float dt = 1f / GameConstants.TickRate;
        _now += dt;

        ApplyMovement(dt);
        ProcessAttacks();
        RespawnMonsters();
        BroadcastSnapshots();
    }

    private void ApplyMovement(float dt)
    {
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
    }

    private void ProcessAttacks()
    {
        const float rangeSq = GameConstants.AttackRange * GameConstants.AttackRange;
        foreach (var s in _sessions.Values)
        {
            while (s.PendingAttacks.TryDequeue(out var targetId))
            {
                if (_now < s.NextAttackTime) continue;   // 쿨다운 중
                var m = _monsters.FirstOrDefault(x => x.EntityId == targetId && x.Alive);
                if (m is null) continue;                 // 없거나 이미 죽은 대상

                float dx = m.X - s.X, dy = m.Y - s.Y;
                if (dx * dx + dy * dy > rangeSq) continue;   // 사거리 밖

                s.NextAttackTime = _now + GameConstants.AttackCooldownSeconds;
                m.Hp -= GameConstants.AttackDamage;
                bool died = m.Hp <= 0;
                if (died)
                {
                    m.Hp = 0;
                    m.RespawnAt = _now + GameConstants.MonsterRespawnSeconds;
                    s.Exp += GameConstants.ExpPerKill;
                    _logger.LogInformation("{Player} 처치: {Monster} (exp +{Exp} → 누적 {Total})",
                        s.Name, m.Name, GameConstants.ExpPerKill, s.Exp);
                }

                Broadcast(ProtocolCodec.Encode(MessageType.CombatEvent, new CombatEvent
                {
                    AttackerId = s.EntityId,
                    TargetId = m.EntityId,
                    Damage = GameConstants.AttackDamage,
                    TargetDied = died,
                }));
            }
        }
    }

    private void RespawnMonsters()
    {
        foreach (var m in _monsters)
        {
            if (!m.Alive && _now >= m.RespawnAt)
            {
                m.Hp = m.MaxHp;
                m.X = m.SpawnX;
                m.Y = m.SpawnY;
            }
        }
    }

    private void BroadcastSnapshots()
    {
        const float aoiSq = GameConstants.AoiRadius * GameConstants.AoiRadius;
        var players = _sessions.Values;
        foreach (var self in players)
        {
            var visible = new List<EntityState>();
            foreach (var other in players)
                if (WithinAoi(self, other.X, other.Y, aoiSq))
                    visible.Add(PlayerState(other));
            foreach (var m in _monsters)
                if (m.Alive && WithinAoi(self, m.X, m.Y, aoiSq))
                    visible.Add(MonsterState(m));

            self.Enqueue(ProtocolCodec.Encode(MessageType.WorldSnapshot, new WorldSnapshot
            {
                ServerTick = _serverTick,
                Entities = visible.ToArray(),
            }));
        }
    }

    private static bool WithinAoi(PlayerSession self, float x, float y, float aoiSq)
    {
        float dx = x - self.X, dy = y - self.Y;
        return dx * dx + dy * dy <= aoiSq;
    }

    private static EntityState PlayerState(PlayerSession s) => new()
    {
        EntityId = s.EntityId, X = s.X, Y = s.Y, Hp = s.Hp, MaxHp = s.MaxHp,
        Kind = EntityKind.Player, Name = s.Name,
    };

    private static EntityState MonsterState(Monster m) => new()
    {
        EntityId = m.EntityId, X = m.X, Y = m.Y, Hp = m.Hp, MaxHp = m.MaxHp,
        Kind = EntityKind.Monster, Name = m.Name,
    };

    // ── 연결 처리 (WebSocket 엔드포인트 호출) ─────────────────────────────────
    public async Task HandleConnectionAsync(WebSocket socket, CancellationToken ct)
    {
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
            Exp = ch.Exp,
        };
        _sessions[session.EntityId] = session;
        _logger.LogInformation("입장: {Name} (entity {Id}), 접속 {Count}명", session.Name, session.EntityId, _sessions.Count);

        session.Enqueue(ProtocolCodec.Encode(MessageType.JoinAccepted, new JoinAccepted
        {
            SelfEntityId = session.EntityId,
            MapId = ch.MapId,
            ServerTickRate = GameConstants.TickRate,
        }));

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

                case MessageType.AttackInput:
                    var atk = ProtocolCodec.DecodePayload<AttackInput>(env);
                    s.PendingAttacks.Enqueue(atk.TargetEntityId);
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
            ch.Exp = s.Exp;
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
