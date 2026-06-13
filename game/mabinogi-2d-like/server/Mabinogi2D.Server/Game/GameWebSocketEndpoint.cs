namespace Mabinogi2D.Server.Game;

public static class GameWebSocketEndpoint
{
    /// <summary>게임 실시간 채널. 클라이언트는 /ws로 WebSocket 업그레이드 후 JoinRequest를 보낸다.</summary>
    public static void MapGameWebSocket(this WebApplication app)
    {
        app.Map("/ws", async (HttpContext ctx, GameWorld world) =>
        {
            if (!ctx.WebSockets.IsWebSocketRequest)
            {
                ctx.Response.StatusCode = StatusCodes.Status400BadRequest;
                return;
            }
            using var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            await world.HandleConnectionAsync(socket, ctx.RequestAborted);
        });
    }
}
