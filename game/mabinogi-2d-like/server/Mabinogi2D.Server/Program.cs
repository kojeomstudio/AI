using System.Text;
using Mabinogi2D.Server.Auth;
using Mabinogi2D.Server.Data;
using Mabinogi2D.Server.Game;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// 영속 DB. 개발 기본값은 SQLite(Docker 불필요), 설정으로 Postgres 전환.
//   Database:Provider = "Sqlite" (기본) | "Postgres"
var dbProvider = builder.Configuration["Database:Provider"] ?? "Sqlite";
builder.Services.AddDbContext<GameDbContext>(opt =>
{
    if (dbProvider.Equals("Postgres", StringComparison.OrdinalIgnoreCase))
        opt.UseNpgsql(builder.Configuration.GetConnectionString("Postgres"));
    else
        opt.UseSqlite(builder.Configuration.GetConnectionString("Sqlite") ?? "Data Source=mabinogi-dev.db");
});

// 게임 월드
builder.Services.AddSingleton<GameWorld>();
builder.Services.AddHostedService<GameLoop>();

var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? "mabinogi-2d-like";

// JWT 서명 키는 절대 소스에 커밋하지 않는다. 환경변수 Jwt__Key 또는 user-secrets로 주입.
//   - 비-Development: 키가 없으면 기동 실패(fail-closed). 토큰 위조 = 인증 우회이므로 안전한 기본값을 두지 않는다.
//   - Development: 편의를 위해 매 실행 임의 키 생성(서버 재시작 시 기존 토큰은 무효화됨).
var jwtKey = builder.Configuration["Jwt:Key"];
if (string.IsNullOrWhiteSpace(jwtKey))
{
    if (builder.Environment.IsDevelopment())
    {
        jwtKey = Convert.ToBase64String(System.Security.Cryptography.RandomNumberGenerator.GetBytes(48));
        Console.WriteLine("[warn] Jwt:Key 미설정 — 개발용 임시 키 생성(재시작 시 토큰 무효). 운영에서는 Jwt__Key를 설정하세요.");
    }
    else
    {
        throw new InvalidOperationException(
            "Jwt:Key가 설정되지 않았습니다. 환경변수 Jwt__Key 또는 user-secrets로 32바이트 이상의 키를 제공하세요.");
    }
}
else if (Encoding.UTF8.GetByteCount(jwtKey) < 32)
{
    throw new InvalidOperationException("Jwt:Key는 HMAC-SHA256용으로 최소 32바이트여야 합니다.");
}

builder.Services.AddSingleton(new JwtService(jwtKey, jwtIssuer));
builder.Services
    .AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(o => o.TokenValidationParameters = JwtService.BuildValidationParameters(jwtKey, jwtIssuer));
builder.Services.AddAuthorization();

var app = builder.Build();

// 개발용 스키마 생성 (마이그레이션은 이후 단계에서 도입)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<GameDbContext>();
    db.Database.EnsureCreated();
}

app.UseWebSockets();
app.UseAuthentication();
app.UseAuthorization();

app.MapGet("/", () => "mabinogi-2d-like server running");
app.MapAuthEndpoints();
app.MapGameWebSocket();

app.Run();
