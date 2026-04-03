using System.Text.Json.Serialization;

namespace MabinogiMacro.Models;

public class AppConfig
{
    [JsonPropertyName("window_title")]
    public string WindowTitle { get; set; } = "Mabinogi Mobile";

    [JsonPropertyName("tick_interval")]
    public double TickInterval { get; set; } = 0.5;

    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = string.Empty;

    [JsonPropertyName("confidence_threshold")]
    public double ConfidenceThreshold { get; set; } = 0.5;
}

public class ActionConfig
{
    [JsonPropertyName("input_method")]
    public string InputMethod { get; set; } = "postmessage";

    [JsonPropertyName("default_delay")]
    public double DefaultDelay { get; set; } = 0.5;

    [JsonPropertyName("actions")]
    public Dictionary<string, ActionDef> Actions { get; set; } = new();

    [JsonPropertyName("priority_rules")]
    public List<PriorityRule> PriorityRules { get; set; } = new();
}

public class ActionDef
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "click";

    [JsonPropertyName("key")]
    public string? Key { get; set; }

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("delay")]
    public double Delay { get; set; } = 0.5;

    [JsonPropertyName("conditions")]
    public List<string> Conditions { get; set; } = new();
}

public class PriorityRule
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("conditions")]
    public List<string> Conditions { get; set; } = new();

    [JsonPropertyName("action")]
    public string Action { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;
}

public class ElementMapping
{
    [JsonPropertyName("elements")]
    public List<ElementDef> Elements { get; set; } = new();
}

public class ElementDef
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("class_id")]
    public int ClassId { get; set; }

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;
}
