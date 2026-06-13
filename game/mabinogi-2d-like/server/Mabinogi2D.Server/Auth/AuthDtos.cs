namespace Mabinogi2D.Server.Auth;

public record RegisterRequest(string Username, string Password);
public record LoginRequest(string Username, string Password);
public record LoginResponse(string Token, long AccountId);

public record CreateCharacterRequest(string Name);
public record CharacterDto(long Id, string Name, int Level, int MapId);
