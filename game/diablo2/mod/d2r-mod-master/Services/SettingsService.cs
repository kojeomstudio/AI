using System;
using System.IO;
using System.Text.Json;

namespace D2RModMaster.Services
{
    public class UserSettings
    {
        public string? LastD2RPath { get; set; }
        public string? LastSearchMask { get; set; }
    }

    public class SettingsService
    {
        private static SettingsService? _instance;
        public static SettingsService Instance => _instance ??= new SettingsService();

        private readonly string _settingsFilePath;
        private UserSettings _settings;

        private SettingsService()
        {
            _settingsFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "settings.json");
            _settings = LoadSettings();
        }

        public UserSettings Settings => _settings;

        public void Save(string? d2rPath, string? searchMask)
        {
            _settings.LastD2RPath = d2rPath;
            _settings.LastSearchMask = searchMask;
            
            try
            {
                string json = JsonSerializer.Serialize(_settings, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_settingsFilePath, json);
            }
            catch (Exception ex)
            {
                LogService.Instance.Log($"Failed to save settings: {ex.Message}", LogLevel.Warning);
            }
        }

        private UserSettings LoadSettings()
        {
            try
            {
                if (File.Exists(_settingsFilePath))
                {
                    string json = File.ReadAllText(_settingsFilePath);
                    var loaded = JsonSerializer.Deserialize<UserSettings>(json);
                    return loaded ?? new UserSettings { LastSearchMask = "*" };
                }
            }
            catch (Exception ex)
            {
                LogService.Instance.Log($"Failed to load settings: {ex.Message}", LogLevel.Warning);
            }

            return new UserSettings { LastSearchMask = "*" };
        }
    }
}
