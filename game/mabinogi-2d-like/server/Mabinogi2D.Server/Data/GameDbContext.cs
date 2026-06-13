using Microsoft.EntityFrameworkCore;

namespace Mabinogi2D.Server.Data;

public class GameDbContext : DbContext
{
    public GameDbContext(DbContextOptions<GameDbContext> options) : base(options) { }

    public DbSet<Account> Accounts => Set<Account>();
    public DbSet<Character> Characters => Set<Character>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        b.Entity<Account>().HasIndex(a => a.Username).IsUnique();
        b.Entity<Character>().HasIndex(c => c.Name).IsUnique();
        b.Entity<Account>()
            .HasMany(a => a.Characters)
            .WithOne(c => c.Account!)
            .HasForeignKey(c => c.AccountId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
