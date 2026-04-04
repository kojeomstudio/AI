using System;
using System.IO;

namespace QuadTerminal.Services;

public static class Logger
{
    private static readonly string LogDir = Path.Combine(AppContext.BaseDirectory, "logs");
    private static readonly object _lock = new();
    private static string? _logFile;

    public static void Init()
    {
        try
        {
            Directory.CreateDirectory(LogDir);
        }
        catch { }
    }

    private static string LogFile
    {
        get
        {
            if (_logFile != null) return _logFile;
            var date = DateTime.Now.ToString("yyyy-MM-dd");
            _logFile = Path.Combine(LogDir, $"quad-terminal_{date}.log");
            return _logFile;
        }
    }

    public static void Info(string message)
    {
        Write("INFO", message);
    }

    public static void Warn(string message)
    {
        Write("WARN", message);
    }

    public static void Error(string message, Exception? ex = null)
    {
        Write("ERROR", ex != null ? $"{message}\n{ex}" : message);
    }

    private static void Write(string level, string message)
    {
        var line = $"[{DateTime.Now:HH:mm:ss.fff}] [{level}] {message}";
        try
        {
            lock (_lock)
            {
                Directory.CreateDirectory(LogDir);
                File.AppendAllText(LogFile, line + Environment.NewLine);
            }
        }
        catch { }

        System.Diagnostics.Debug.WriteLine(line);
    }
}
