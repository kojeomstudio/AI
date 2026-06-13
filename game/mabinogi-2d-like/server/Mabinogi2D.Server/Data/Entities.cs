using Mabinogi2D.Shared;

namespace Mabinogi2D.Server.Data;

public class Account
{
    public long Id { get; set; }
    public string Username { get; set; } = "";
    public string PasswordHash { get; set; } = "";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public List<Character> Characters { get; set; } = new();
}

public class Character
{
    public long Id { get; set; }
    public long AccountId { get; set; }
    public Account? Account { get; set; }

    public string Name { get; set; } = "";
    public int Level { get; set; } = 1;
    public long Exp { get; set; }

    public int MapId { get; set; } = GameConstants.StartingMapId;
    public float PosX { get; set; }
    public float PosY { get; set; }

    public int Hp { get; set; } = 100;
    public int MaxHp { get; set; } = 100;
}
