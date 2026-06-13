using Mabinogi2D.Shared;

namespace Mabinogi2D.Server.Game;

/// <summary>고정 주기(20Hz) 틱 루프. 월드 시뮬레이션을 한 스레드에서 순차 구동한다.</summary>
public sealed class GameLoop : BackgroundService
{
    private readonly GameWorld _world;
    private readonly ILogger<GameLoop> _logger;

    public GameLoop(GameWorld world, ILogger<GameLoop> logger)
    {
        _world = world;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var period = TimeSpan.FromSeconds(1.0 / GameConstants.TickRate);
        using var timer = new PeriodicTimer(period);
        _logger.LogInformation("게임 루프 시작 @ {Hz}Hz", GameConstants.TickRate);

        while (await timer.WaitForNextTickAsync(stoppingToken))
        {
            try { _world.Tick(); }
            catch (Exception ex) { _logger.LogError(ex, "틱 처리 중 예외"); }
        }
    }
}
