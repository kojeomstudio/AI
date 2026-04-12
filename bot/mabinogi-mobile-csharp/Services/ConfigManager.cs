using System.IO;
using System.Text.Json;
using MabinogiMacro.Models;

namespace MabinogiMacro.Services;

public class ConfigManager
{
    private readonly string _baseDir;
    private readonly string _configDir;

    public AppConfig AppConfig { get; private set; } = new();
    public ActionConfig ActionConfig { get; private set; } = new();
    public ElementMapping ElementMapping { get; private set; } = new();

    public ConfigManager()
    {
        _baseDir = AppContext.BaseDirectory;
        _configDir = Path.Combine(_baseDir, "Config");
        LoadAll();
    }

    public string ResolvePath(string relativePath)
    {
        if (string.IsNullOrEmpty(relativePath)) return relativePath;
        if (Path.IsPathRooted(relativePath)) return relativePath;
        return Path.Combine(_baseDir, relativePath);
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
