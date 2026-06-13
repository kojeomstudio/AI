using System.Collections.Generic;
using System.Linq;
using Godot;
using Mabinogi2D.Shared.Protocol;

namespace Mabinogi2D.Client;

/// <summary>
/// MVP 클라이언트: 화살표 입력을 서버로 보내고, 월드 스냅샷의 엔티티를 사각형으로 그린다.
/// 스프라이트는 다음 단계(그리드 시트 임포터). 지금은 이동 동기화 검증이 목표.
/// </summary>
public partial class Main : Node2D
{
    private NetClient _net = null!;
    private Label _status = null!;

    private WorldSnapshot? _latest;
    private readonly Dictionary<long, Vector2> _display = new();  // 부드러운 표시 위치
    private Vector2 _lastSentDir = new(float.NaN, float.NaN);

    public override void _Ready()
    {
        _net = GetNode<NetClient>("NetClient");
        _status = GetNode<Label>("UI/Status");
        _net.OnSnapshot += s => _latest = s;
        _net.OnStatus += msg => _status.Text = msg;
    }

    public override void _Process(double delta)
    {
        // 1) 입력 → 서버 (방향이 바뀔 때만 전송)
        var dir = new Vector2(
            Input.GetAxis("ui_left", "ui_right"),
            Input.GetAxis("ui_up", "ui_down"));
        if (!dir.IsEqualApprox(_lastSentDir))
        {
            _net.SendMove(dir.X, dir.Y);
            _lastSentDir = dir;
        }

        // 2) 표시 위치를 서버 권위 위치로 보간 (원격 플레이어 부드럽게)
        if (_latest is { } snap)
        {
            var seen = new HashSet<long>();
            foreach (var e in snap.Entities)
            {
                seen.Add(e.EntityId);
                var target = new Vector2(e.X, e.Y);
                _display[e.EntityId] = _display.TryGetValue(e.EntityId, out var cur)
                    ? cur.Lerp(target, Mathf.Min(1f, (float)delta * 12f))
                    : target;
            }
            _display.Keys.Where(id => !seen.Contains(id)).ToList()
                .ForEach(id => _display.Remove(id));
        }
        QueueRedraw();
    }

    public override void _Draw()
    {
        if (_latest is not { } snap) return;
        var offset = GetViewportRect().Size * 0.5f;   // 월드 원점을 화면 중앙으로
        foreach (var e in snap.Entities)
        {
            if (!_display.TryGetValue(e.EntityId, out var p)) continue;
            var pos = p + offset;
            var color = e.EntityId == _net.SelfEntityId ? Colors.SkyBlue : Colors.OrangeRed;
            DrawRect(new Rect2(pos - new Vector2(12, 12), new Vector2(24, 24)), color);
            DrawString(ThemeDB.FallbackFont, pos + new Vector2(-14, -18), e.Name,
                HorizontalAlignment.Left, -1, 12);
        }
    }
}
