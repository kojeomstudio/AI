using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;

namespace CascViewerWPF.Services
{
    public class LogService
    {
        private static LogService? _instance;
        public static LogService Instance => _instance ??= new LogService();

        private readonly string _logFilePath;
        public ObservableCollection<LogEntry> Logs { get; } = new ObservableCollection<LogEntry>();

        private LogService()
        {
            _logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "app.log");
            Log("Logger initialized.");
        }

        public void Log(string message, LogLevel level = LogLevel.Info)
        {
            var entry = new LogEntry
            {
                Timestamp = DateTime.Now,
                Message = message,
                Level = level
            };

            // Update UI safely
            if (System.Windows.Application.Current != null)
            {
                System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                {
                    Logs.Add(entry);
                    // Keep only last 1000 logs in UI
                    if (Logs.Count > 1000) Logs.RemoveAt(0);
                }));
            }

            // Write to file
            try
            {
                File.AppendAllText(_logFilePath, $"[{entry.Timestamp:yyyy-MM-dd HH:mm:ss}] [{level}] {message}{Environment.NewLine}");
            }
            catch
            {
                // Ignore logging errors to prevent app crash
            }
        }
    }

    public enum LogLevel { Info, Warning, Error }

    public class LogEntry
    {
        public DateTime Timestamp { get; set; }
        public string? Message { get; set; }
        public LogLevel Level { get; set; }
        public string DisplayText => $"[{Timestamp:HH:mm:ss}] {Message}";
    }
}
