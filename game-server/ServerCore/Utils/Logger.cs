using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public enum LogLevel
{
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal
}

public interface ILogger
{
    Task LogAsync(LogLevel level, string message);
}

public class ConsoleLogger : ILogger
{
    public Task LogAsync(LogLevel level, string message)
    {
        var log = $"[{DateTime.UtcNow:O}] [{level}] {message}";
        Console.WriteLine(log);
        return Task.CompletedTask;
    }
}
public class FileLogger : ILogger
{
    private readonly string _filePath;
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    public FileLogger(string filePath)
    {
        _filePath = filePath;
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
    }

    public async Task LogAsync(LogLevel level, string message)
    {
        var log = $"[{DateTime.UtcNow:O}] [{level}] {message}{Environment.NewLine}";

        await _semaphore.WaitAsync();
        try
        {
            await File.AppendAllTextAsync(_filePath, log);
        }
        finally
        {
            _semaphore.Release();
        }
    }
}

public class ServerLogger : ILogger
{
    private static readonly Lazy<ServerLogger> _instance =
        new(() => new ServerLogger());

    public static ServerLogger Instance => _instance.Value;

    private readonly List<ILogger> _loggers;

    private ServerLogger()
    {
        _loggers = new List<ILogger>
        {
            new ConsoleLogger(),
            new FileLogger("logs/server.log")
        };
    }

    public async Task LogAsync(LogLevel level, string message)
    {
        foreach (var logger in _loggers)
            await logger.LogAsync(level, message);
    }

    public void Log(LogLevel level, string msg)
    {
        LogAsync(level, msg).GetAwaiter().GetResult();
    }

    public Task LogInfoAsync(string msg) => LogAsync(LogLevel.Info, msg);
    public Task LogErrorAsync(string msg) => LogAsync(LogLevel.Error, msg);
}

