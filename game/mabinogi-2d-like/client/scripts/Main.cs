using System.Collections.Generic;
using System.Linq;
using Godot;
using Mabinogi2D.Shared.Protocol;

namespace Mabinogi2D.Client;

/// <summary>
/// MVP 클라이언트: 화살표로 이동, Space로 가장 가까운 몬스터 공격. 스냅샷 엔티티를 사각형 + HP바로 그린다.
/// 스프라이트 교체는 다음 단계(그리드 시트 임포터).
/// </summary>
public partial class Main : Node2D
{
    private NetClient _net = null!;
    private Label _status = null!;

    private WorldSnapshot? _latest;
    private readonly Dictionary<long, Vector2> _display = new();  // 부드러운 표시 위치
    private Vector2 _lastSentDir = new(float.NaN, float.NaN);
    private string _combatLine = "";

    public override void _Ready()
    {
        _net = GetNode<NetClient>("NetClient");
        _status = GetNode<Label>("UI/Status");
        _net.OnSnapshot += s => _latest = s;
        _net.OnStatus += msg => _status.Text = msg;
        _net.OnCombat += OnCombat;
    }

    private void OnCombat(CombatEvent e)
    {
        var name = EntityName(e.TargetId);
        _combatLine = e.TargetDied
            ? $"{name} 처치!"
            : $"{name}에게 {e.Damage} 데미지";
    }

    public override void _Process(double delta)
    {
        // 1) 이동 입력 → 서버 (방향이 바뀔 때만 전송)
        var dir = new Vector2(
            Input.GetAxis("ui_left", "ui_right"),
            Input.GetAxis("ui_up", "ui_down"));
        if (!dir.IsEqualApprox(_lastSentDir))
        {
            _net.SendMove(dir.X, dir.Y);
            _lastSentDir = dir;
        }

        // 2) 공격 입력 (Space/Enter) → 가장 가까운 몬스터 대상
        if (Input.IsActionJustPressed("ui_accept"))
        {
            var targetId = NearestMonsterToSelf();
            if (targetId >= 0) _net.SendAttack(targetId);
        }

        // 3) 표시 위치를 서버 권위 위치로 보간
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

            var color = e.Kind switch
            {
                EntityKind.Monster => Colors.LimeGreen,
                _ => e.EntityId == _net.SelfEntityId ? Colors.SkyBlue : Colors.OrangeRed,
            };
            DrawRect(new Rect2(pos - new Vector2(12, 12), new Vector2(24, 24)), color);

            // HP 바
            if (e.MaxHp > 0)
            {
                var barPos = pos + new Vector2(-14, -22);
                var barSize = new Vector2(28, 4);
                float ratio = Mathf.Clamp((float)e.Hp / e.MaxHp, 0f, 1f);
                DrawRect(new Rect2(barPos, barSize), Colors.DarkRed);
                DrawRect(new Rect2(barPos, new Vector2(barSize.X * ratio, barSize.Y)), Colors.Green);
            }

            DrawString(ThemeDB.FallbackFont, pos + new Vector2(-14, -26), e.Name,
                HorizontalAlignment.Left, -1, 11);
        }

        if (_combatLine.Length > 0)
            DrawString(ThemeDB.FallbackFont, new Vector2(8, 56), _combatLine, HorizontalAlignment.Left, -1, 14);
    }

    private long NearestMonsterToSelf()
    {
        if (_latest is not { } snap || !_display.TryGetValue(_net.SelfEntityId, out var self))
            return -1;
        long best = -1; float bestDist = float.MaxValue;
        foreach (var e in snap.Entities)
        {
            if (e.Kind != EntityKind.Monster) continue;
            float d = self.DistanceSquaredTo(new Vector2(e.X, e.Y));
            if (d < bestDist) { bestDist = d; best = e.EntityId; }
        }
        return best;
    }

    private string EntityName(long entityId)
    {
        if (_latest is { } snap)
            foreach (var e in snap.Entities)
                if (e.EntityId == entityId) return e.Name;
        return $"#{entityId}";
    }
}
