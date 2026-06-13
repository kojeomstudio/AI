namespace Mabinogi2D.Shared;

/// <summary>서버와 클라이언트가 합의해야 하는 게임 규칙 상수.</summary>
public static class GameConstants
{
    /// <summary>서버 시뮬레이션 틱 레이트(Hz). 클라 보간/예측 기준.</summary>
    public const int TickRate = 20;

    /// <summary>플레이어 이동 속도(월드 유닛/초).</summary>
    public const float PlayerMoveSpeed = 96f;

    /// <summary>관심영역 반경(월드 유닛). 이 범위 밖 엔티티는 스냅샷에서 제외.</summary>
    public const float AoiRadius = 480f;

    /// <summary>MVP 단일 공유 맵 id.</summary>
    public const int StartingMapId = 1;
}
