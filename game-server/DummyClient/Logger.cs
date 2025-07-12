using System;

public enum LogLevel
{
    Info,
    Debug,
    Error
}

public class ClientLogger
{
    private static readonly Lazy<ClientLogger> _instance = new(() => new ClientLogger());

    public static ClientLogger Instance => _instance.Value;

    private ClientLogger()
    {
        // to do...
    }

    private void WriteLog(LogLevel level, string message)
    {
        var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        var log = $"[{timestamp}] [{level}] {message}";

        switch (level)
        {
            case LogLevel.Info:
                Console.ForegroundColor = ConsoleColor.Green;
                break;
            case LogLevel.Debug:
                Console.ForegroundColor = ConsoleColor.Cyan;
                break;
            case LogLevel.Error:
                Console.ForegroundColor = ConsoleColor.Red;
                break;
        }

        Console.WriteLine(log);
        Console.ResetColor();
    }

    public void Info(string message) => WriteLog(LogLevel.Info, message);
    public void Debug(string message) => WriteLog(LogLevel.Debug, message);
    public void Error(string message) => WriteLog(LogLevel.Error, message);
}
