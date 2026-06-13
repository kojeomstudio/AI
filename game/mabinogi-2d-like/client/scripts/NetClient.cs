using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;
using Mabinogi2D.Shared.Protocol;
using HttpClient = System.Net.Http.HttpClient;

namespace Mabinogi2D.Client;

/// <summary>
/// 서버 연결을 담당. REST로 로그인/캐릭터 확보 → WebSocket(/ws)으로 실시간 채널 연결.
/// 프로토콜 직렬화는 서버와 동일한 Mabinogi2D.Shared.ProtocolCodec을 그대로 쓴다.
/// </summary>
public partial class NetClient : Node
{
    [Export] public string HttpBase = "http://localhost:5080";
    [Export] public string WsUrl = "ws://localhost:5080/ws";
    [Export] public string Username = "hero";
    [Export] public string Password = "secret";
    [Export] public string CharacterName = "Tarlach";

    private readonly WebSocketPeer _ws = new();
    private WebSocketPeer.State _lastState = WebSocketPeer.State.Closed;

    public long SelfEntityId { get; private set; } = -1;
    public event Action<WorldSnapshot>? OnSnapshot;
    public event Action<string>? OnStatus;

    private string? _token;
    private long _characterId = -1;
    private volatile bool _authReady;
    private volatile bool _authFailed;
    private bool _wsStarted;
    private bool _joinSent;

    public override void _Ready()
    {
        ParseCmdlineArgs();
        _ = AuthAsync();
    }

    /// <summary>`-- --user X --char Y --server http://host:port` 형태의 사용자 인자로 기본값을 덮어쓴다(다중 인스턴스 테스트용).</summary>
    private void ParseCmdlineArgs()
    {
        var args = OS.GetCmdlineUserArgs();
        for (int i = 0; i + 1 < args.Length; i++)
        {
            switch (args[i])
            {
                case "--user": Username = args[i + 1]; break;
                case "--char": CharacterName = args[i + 1]; break;
                case "--server":
                    var b = args[i + 1].TrimEnd('/');
                    HttpBase = b;
                    WsUrl = b.Replace("https://", "wss://").Replace("http://", "ws://") + "/ws";
                    break;
            }
        }
    }

    /// <summary>회원가입(있으면 무시) → 로그인 → 캐릭터 확보. 백그라운드 Task, 결과는 플래그로 전달.</summary>
    private async Task AuthAsync()
    {
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri(HttpBase) };

            await http.PostAsJsonAsync("/auth/register", new { username = Username, password = Password });

            var loginResp = await http.PostAsJsonAsync("/auth/login", new { username = Username, password = Password });
            loginResp.EnsureSuccessStatusCode();
            var login = await loginResp.Content.ReadFromJsonAsync<JsonElement>();
            _token = login.GetProperty("token").GetString();

            http.DefaultRequestHeaders.Authorization = new("Bearer", _token);
            var list = await http.GetFromJsonAsync<JsonElement>("/characters");
            if (list.GetArrayLength() > 0)
            {
                _characterId = list[0].GetProperty("id").GetInt64();
            }
            else
            {
                var create = await http.PostAsJsonAsync("/characters", new { name = CharacterName });
                var ch = await create.Content.ReadFromJsonAsync<JsonElement>();
                _characterId = ch.GetProperty("id").GetInt64();
            }
            _authReady = true;
        }
        catch (Exception e)
        {
            GD.PrintErr($"[auth] 실패: {e.Message}");
            _authFailed = true;
        }
    }

    public override void _Process(double delta)
    {
        if (_authFailed) { OnStatus?.Invoke("인증 실패 — 서버(5080)가 실행 중인지 확인"); return; }
        if (!_authReady) { OnStatus?.Invoke("로그인 중..."); return; }

        if (!_wsStarted)
        {
            var err = _ws.ConnectToUrl(WsUrl);
            if (err != Error.Ok) { OnStatus?.Invoke($"WS 연결 실패: {err}"); return; }
            _wsStarted = true;
        }

        _ws.Poll();
        var state = _ws.GetReadyState();

        if (state == WebSocketPeer.State.Open)
        {
            if (!_joinSent)
            {
                Send(MessageType.JoinRequest, new JoinRequest { Token = _token!, CharacterId = _characterId });
                _joinSent = true;
                OnStatus?.Invoke("입장 요청 전송...");
            }
            while (_ws.GetAvailablePacketCount() > 0)
                HandlePacket(_ws.GetPacket());
        }
        else if (state == WebSocketPeer.State.Closed && _lastState == WebSocketPeer.State.Open)
        {
            OnStatus?.Invoke("서버 연결이 종료되었습니다.");
        }
        _lastState = state;
    }

    private void HandlePacket(byte[] data)
    {
        var env = ProtocolCodec.DecodeEnvelope(data);
        switch (env.Type)
        {
            case MessageType.JoinAccepted:
                var ja = ProtocolCodec.DecodePayload<JoinAccepted>(env);
                SelfEntityId = ja.SelfEntityId;
                OnStatus?.Invoke($"입장 완료 — entity={ja.SelfEntityId}, map={ja.MapId}, {ja.ServerTickRate}Hz");
                break;
            case MessageType.WorldSnapshot:
                OnSnapshot?.Invoke(ProtocolCodec.DecodePayload<WorldSnapshot>(env));
                break;
            case MessageType.Error:
                var er = ProtocolCodec.DecodePayload<ErrorMessage>(env);
                OnStatus?.Invoke($"서버 오류 {er.Code}: {er.Detail}");
                break;
        }
    }

    /// <summary>이동 의도(방향) 전송. 서버가 검증·시뮬레이션한다.</summary>
    public void SendMove(float dirX, float dirY)
    {
        if (_ws.GetReadyState() != WebSocketPeer.State.Open || !_joinSent) return;
        Send(MessageType.MoveInput, new MoveInput { DirX = dirX, DirY = dirY });
    }

    private void Send<T>(MessageType type, T payload) => _ws.PutPacket(ProtocolCodec.Encode(type, payload));
}
