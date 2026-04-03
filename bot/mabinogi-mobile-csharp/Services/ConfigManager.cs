using System.IO;
using System.Text.Json;
using MabinogiMacro.Models;

namespace MabinogiMacro.Services;

public class ConfigManager
{
    private readonly string _configDir;

    public AppConfig AppConfig { get; private set; } = new();
    public ActionConfig ActionConfig { get; private set; } = new();
    public ElementMapping ElementMapping { get; private set; } = new();

    public ConfigManager()
    {
        _configDir = Path.Combine(AppContext.BaseDirectory, "Config");
        LoadAll();
    }

    private void LoadAll()
    {
        AppConfig = LoadJson<AppConfig>(Path.Combine(_configDir, "config.json")) ?? new();
        ActionConfig = LoadJson<ActionConfig>(Path.Combine(_configDir, "action_config.json")) ?? new();
        ElementMapping = LoadJson<ElementMapping>(Path.Combine(_configDir, "elements.json")) ?? new();
    }

    private static T? LoadJson<T>(string path) where T : class
    {
        if (!File.Exists(path)) return null;
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
    }

    public void Reload()
    {
        LoadAll();
    }
}
