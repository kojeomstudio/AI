using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;

namespace Mabinogi2D.Server.Auth;

/// <summary>
/// REST 로그인 시 JWT를 발급하고, 게임 WebSocket 입장 시 같은 토큰을 검증한다.
/// REST(JwtBearer 미들웨어)와 WebSocket(수동 검증)이 동일 파라미터를 공유한다.
/// </summary>
public class JwtService
{
    private readonly string _key;
    private readonly string _issuer;
    private readonly TokenValidationParameters _validationParams;

    // 키/이슈어는 Program.cs에서 해석된 단일 값을 주입받는다(설정·dev 임시키 경로가 한 곳으로 수렴).
    public JwtService(string key, string issuer)
    {
        _key = key;
        _issuer = issuer;
        _validationParams = BuildValidationParameters(key, issuer);
    }

    public static TokenValidationParameters BuildValidationParameters(string key, string issuer) => new()
    {
        ValidateIssuer = true,
        ValidIssuer = issuer,
        ValidateAudience = true,
        ValidAudience = issuer,
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(key)),
        ValidateLifetime = true,
        ClockSkew = TimeSpan.FromSeconds(30),
    };

    public string CreateToken(long accountId, string username)
    {
        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, accountId.ToString()),
            new Claim("username", username),
        };
        var creds = new SigningCredentials(
            new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_key)),
            SecurityAlgorithms.HmacSha256);
        var token = new JwtSecurityToken(
            issuer: _issuer,
            audience: _issuer,
            claims: claims,
            expires: DateTime.UtcNow.AddHours(12),
            signingCredentials: creds);
        return new JwtSecurityTokenHandler().WriteToken(token);
    }

    /// <summary>WebSocket JoinRequest 검증용. 유효하면 accountId, 아니면 null.</summary>
    public long? ValidateAndGetAccountId(string token)
    {
        try
        {
            var handler = new JwtSecurityTokenHandler { MapInboundClaims = false };
            var principal = handler.ValidateToken(token, _validationParams, out _);
            var sub = principal.FindFirst(JwtRegisteredClaimNames.Sub)?.Value
                      ?? principal.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            return long.TryParse(sub, out var id) ? id : null;
        }
        catch
        {
            return null;
        }
    }
}
