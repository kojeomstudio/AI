using MabinogiMacro.Models;
using Serilog;

namespace MabinogiMacro.Services;

public class ActionProcessor
{
    private readonly InputManager _inputManager;
    private readonly ActionConfig _config;
    private readonly Dictionary<string, DateTime> _lastActionTime = new();
    private readonly Dictionary<string, int> _actionCounts = new();

    public ActionProcessor(ActionConfig config, InputManager inputManager)
    {
        _config = config;
        _inputManager = inputManager;
    }

    private InputMethod ParseInputMethod()
    {
        return _config.InputMethod?.ToLowerInvariant() switch
        {
            "sendinput" => InputMethod.SendInput,
            "send_message" or "sendmessage" => InputMethod.SendMessage,
            _ => InputMethod.PostMessage,
        };
    }

    public bool ProcessDetectedElements(Dictionary<ElementType, DetectedElement> matched)
    {
        if (matched.Count == 0) return false;

        var detectedTypes = matched.Keys.ToList();
        Log.Debug("Detected elements: {Types}", string.Join(", ", detectedTypes.Select(t => t.ToString())));

        if (HandlePriorityRules(detectedTypes, matched))
            return true;

        return HandleIndividualActions(matched);
    }

    private bool HandlePriorityRules(List<ElementType> detectedTypes, Dictionary<ElementType, DetectedElement> matched)
    {
        var inputMethod = ParseInputMethod();

        foreach (var rule in _config.PriorityRules)
        {
            if (rule.Conditions.Count == 0) continue;

            var detectedNames = new HashSet<string>(detectedTypes.Select(t => t.ToString()));
            if (!rule.Conditions.All(c => detectedNames.Contains(c))) continue;

            Log.Debug("Matched priority rule: {Rule}", rule.Name);

            if (rule.Action == "wait")
            {
                Log.Information("State: Wait - {Desc}", rule.Description);
                return true;
            }

            if (!_config.Actions.ContainsKey(rule.Action)) continue;

            var targetType = detectedTypes.FirstOrDefault(t => t.ToString() == rule.Action);
            if (targetType == default && matched.ContainsKey(targetType))
                continue;

            if (matched.TryGetValue(targetType, out var element))
            {
                if (CheckCooldown(rule.Action))
                {
                    return ExecuteAction(rule.Action, element, inputMethod);
                }
            }
        }

        return false;
    }

    private bool HandleIndividualActions(Dictionary<ElementType, DetectedElement> matched)
    {
        var inputMethod = ParseInputMethod();

        foreach (var (type, element) in matched)
        {
            var name = type.ToString();
            if (!name.StartsWith("UI_")) continue;
            if (name == "UI_WORKING" || name == "UI_COMPASS") continue;

            if (_config.Actions.ContainsKey(name) && CheckCooldown(name))
            {
                return ExecuteAction(name, element, inputMethod);
            }
        }

        return false;
    }

    private bool CheckCooldown(string actionName)
    {
        if (!_lastActionTime.ContainsKey(actionName)) return true;

        double delay = _config.DefaultDelay;
        if (_config.Actions.TryGetValue(actionName, out var actionDef))
            delay = actionDef.Delay;

        return (DateTime.UtcNow - _lastActionTime[actionName]).TotalSeconds >= delay;
    }

    private void UpdateActionTime(string actionName)
    {
        _lastActionTime[actionName] = DateTime.UtcNow;
        _actionCounts.TryGetValue(actionName, out var count);
        _actionCounts[actionName] = count + 1;
    }

    private bool ExecuteAction(string actionName, DetectedElement element, InputMethod inputMethod)
    {
        if (!_config.Actions.TryGetValue(actionName, out var actionDef)) return false;

        bool success = actionDef.Type switch
        {
            "click" => _inputManager.Click(element.CenterX, element.CenterY, inputMethod),
            "key" when actionDef.Key != null => _inputManager.SendKey(actionDef.Key, inputMethod),
            _ => false,
        };

        if (success)
        {
            UpdateActionTime(actionName);
            Log.Information("Action executed: {Action} at ({X},{Y})", actionName, element.CenterX, element.CenterY);
        }

        return success;
    }

    public ActionStats GetStats()
    {
        return new ActionStats
        {
            TotalActions = _actionCounts.Values.Sum(),
            ActionCounts = new(_actionCounts),
        };
    }

    public void TestInputMethods(int x, int y)
    {
        Log.Information("=== Input Method Test at ({X},{Y}) ===", x, y);

        foreach (InputMethod method in Enum.GetValues<InputMethod>())
        {
            var success = _inputManager.Click(x, y, method);
            Log.Information("  {Method}: {Result}", method, success ? "OK" : "FAIL");
            Thread.Sleep(1000);
        }

        foreach (var key in new[] { "space", "enter" })
        {
            foreach (InputMethod method in Enum.GetValues<InputMethod>())
            {
                var success = _inputManager.SendKey(key, method);
                Log.Information("  Key={Key} {Method}: {Result}", key, method, success ? "OK" : "FAIL");
                Thread.Sleep(500);
            }
        }

        Log.Information("=== Test Complete ===");
    }
}

public record ActionStats
{
    public int TotalActions { get; init; }
    public Dictionary<string, int> ActionCounts { get; init; } = new();
}
