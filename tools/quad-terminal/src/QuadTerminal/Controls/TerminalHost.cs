using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using EasyWindowsTerminalControl;
using QuadTerminal.Services;

namespace QuadTerminal.Controls;

public class TerminalHost : UserControl
{
    private EasyTerminalControl? _terminal;
    private int _paneIndex;
    private string _shell = "powershell.exe";
    private string _workingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private string? _cwdFile;

    public event EventHandler? TerminalReady;

    public static readonly DependencyProperty ShellProperty =
        DependencyProperty.Register(nameof(Shell), typeof(string), typeof(TerminalHost),
            new PropertyMetadata("powershell.exe"));

    public static readonly DependencyProperty WorkingDirectoryProperty =
        DependencyProperty.Register(nameof(WorkingDirectory), typeof(string), typeof(TerminalHost),
            new PropertyMetadata(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)));

    public static readonly DependencyProperty PaneIndexProperty =
        DependencyProperty.Register(nameof(PaneIndex), typeof(int), typeof(TerminalHost),
            new PropertyMetadata(0));

    public string Shell
    {
        get => (string)GetValue(ShellProperty);
        set => SetValue(ShellProperty, value);
    }

    public string WorkingDirectory
    {
        get => (string)GetValue(WorkingDirectoryProperty);
        set => SetValue(WorkingDirectoryProperty, value);
    }

    public int PaneIndex
    {
        get => (int)GetValue(PaneIndexProperty);
        set => SetValue(PaneIndexProperty, value);
    }

    protected override void OnInitialized(EventArgs e)
    {
        base.OnInitialized(e);
        Unloaded += OnUnloaded;
    }

    public void StartTerminal()
    {
        Logger.Info($"[Pane {PaneIndex}] Starting terminal: {_shell}");

        _cwdFile = Path.Combine(Path.GetTempPath(), $"QuadTerminal_pane{_paneIndex}.cwd");
        try { File.Delete(_cwdFile); } catch { }

        var workDir = _workingDirectory;
        if (!string.IsNullOrEmpty(workDir) && !Directory.Exists(workDir))
            workDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrEmpty(workDir))
            workDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        try
        {
            string cwdHook = BuildCwdHookScript();
            string arguments = $"-NoExit -ExecutionPolicy Bypass -Command \"{cwdHook}\"";

            _terminal = new EasyTerminalControl
            {
                StartupCommandLine = $"{_shell} {arguments}"
            };

            Content = _terminal;

            _terminal.Loaded += (_, _) =>
            {
                Logger.Info($"[Pane {PaneIndex}] Terminal control loaded, ready.");
                TerminalReady?.Invoke(this, EventArgs.Empty);
            };

            Logger.Info($"[Pane {PaneIndex}] EasyTerminalControl created.");
        }
        catch (Exception ex)
        {
            Logger.Error($"[Pane {PaneIndex}] Exception: {ex.Message}", ex);
            TerminalReady?.Invoke(this, EventArgs.Empty);
        }
    }

    private string BuildCwdHookScript()
    {
        _cwdFile ??= Path.Combine(Path.GetTempPath(), $"QuadTerminal_pane{_paneIndex}.cwd");
        return $"$_qt_cwd='{_cwdFile}'; $_qt_op=$function:prompt; function prompt{{ try{{(Get-Location).Path|Out-File -LiteralPath $_qt_cwd -Encoding utf8 -EA SilentlyContinue}}catch{{}}; if($_qt_op){{&$_qt_op}}else{{'PS '+(Get-Location).Path+'> '}} }}; Clear-Host";
    }

    public string GetCurrentDirectory()
    {
        if (_cwdFile == null) return WorkingDirectory;
        try
        {
            if (File.Exists(_cwdFile))
            {
                var dir = File.ReadAllText(_cwdFile).Trim().Trim('"');
                if (Directory.Exists(dir)) return dir;
            }
        }
        catch { }
        return WorkingDirectory;
    }

    private void OnUnloaded(object sender, RoutedEventArgs e)
    {
        _terminal = null;
        Content = null;
        try { if (_cwdFile != null) File.Delete(_cwdFile); } catch { }
    }
}
