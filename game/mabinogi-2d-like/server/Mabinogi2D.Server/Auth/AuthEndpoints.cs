using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using Mabinogi2D.Server.Data;
using Microsoft.EntityFrameworkCore;

namespace Mabinogi2D.Server.Auth;

public static class AuthEndpoints
{
    public static void MapAuthEndpoints(this WebApplication app)
    {
        var auth = app.MapGroup("/auth");

        auth.MapPost("/register", async (RegisterRequest req, GameDbContext db) =>
        {
            if (string.IsNullOrWhiteSpace(req.Username) || (req.Password?.Length ?? 0) < 4)
                return Results.BadRequest("username 필수, password 4자 이상");
            if (await db.Accounts.AnyAsync(a => a.Username == req.Username))
                return Results.Conflict("이미 사용 중인 username");

            var acc = new Account
            {
                Username = req.Username,
                PasswordHash = BCrypt.Net.BCrypt.HashPassword(req.Password),
            };
            db.Accounts.Add(acc);
            await db.SaveChangesAsync();
            return Results.Ok(new { acc.Id, acc.Username });
        });

        auth.MapPost("/login", async (LoginRequest req, GameDbContext db, JwtService jwt) =>
        {
            var acc = await db.Accounts.FirstOrDefaultAsync(a => a.Username == req.Username);
            if (acc is null || !BCrypt.Net.BCrypt.Verify(req.Password, acc.PasswordHash))
                return Results.Unauthorized();
            return Results.Ok(new LoginResponse(jwt.CreateToken(acc.Id, acc.Username), acc.Id));
        });

        var chars = app.MapGroup("/characters").RequireAuthorization();

        chars.MapGet("/", async (ClaimsPrincipal user, GameDbContext db) =>
        {
            var accId = GetAccountId(user);
            var list = await db.Characters
                .Where(c => c.AccountId == accId)
                .Select(c => new CharacterDto(c.Id, c.Name, c.Level, c.MapId))
                .ToListAsync();
            return Results.Ok(list);
        });

        chars.MapPost("/", async (CreateCharacterRequest req, ClaimsPrincipal user, GameDbContext db) =>
        {
            var accId = GetAccountId(user);
            if (string.IsNullOrWhiteSpace(req.Name))
                return Results.BadRequest("name 필수");
            if (await db.Characters.AnyAsync(c => c.Name == req.Name))
                return Results.Conflict("이미 사용 중인 캐릭터 이름");

            var ch = new Character { AccountId = accId, Name = req.Name };
            db.Characters.Add(ch);
            await db.SaveChangesAsync();
            return Results.Ok(new CharacterDto(ch.Id, ch.Name, ch.Level, ch.MapId));
        });
    }

    private static long GetAccountId(ClaimsPrincipal user)
    {
        var sub = user.FindFirst(JwtRegisteredClaimNames.Sub)?.Value
                  ?? user.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        return long.Parse(sub!);
    }
}
