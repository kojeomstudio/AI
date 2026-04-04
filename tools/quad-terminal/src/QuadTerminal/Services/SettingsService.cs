using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using QuadTerminal.Models;

namespace QuadTerminal.Services;

public class SettingsService
{
    private readonly string _settingsPath;

    public SettingsService()
    {
        _settingsPath = Path.Combine(AppContext.BaseDirectory, "settings.json");
    }

    public AppSettings Load()
    {
        if (!File.Exists(_settingsPath)) return new AppSettings();
        try
        {
            var json = File.ReadAllText(_settingsPath);
            return JsonSerializer.Deserialize<AppSettings>(json, JsonSerializerOptions) ?? new AppSettings();
        }
        catch
        {
            return new AppSettings();
        }
    }

    public void Save(AppSettings settings)
    {
        try
        {
            var json = JsonSerializer.Serialize(settings, JsonSerializerOptions);
            File.WriteAllText(_settingsPath, json);
        }
        catch
        {
        }
    }

    private static readonly JsonSerializerOptions JsonSerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };
}
