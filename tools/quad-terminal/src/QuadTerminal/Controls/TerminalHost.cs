using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;

namespace QuadTerminal.Controls;

public class TerminalHost : HwndHost
{
    private Process? _process;
    private IntPtr _childHwnd;
    private int _paneIndex;
    private string? _cwdFile;
    private string? _initScriptFile;

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
        catch
        {
        }
        return WorkingDirectory;
    }

    protected override HandleRef BuildWindowCore(HandleRef hwndParent)
    {
        _cwdFile = Path.Combine(Path.GetTempPath(), $"QuadTerminal_pane{_paneIndex}.cwd");
        try { File.Delete(_cwdFile); } catch { }

        var workDir = WorkingDirectory;
        if (!string.IsNullOrEmpty(workDir) && !Directory.Exists(workDir))
            workDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrEmpty(workDir))
            workDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        string arguments = "";
        if (Shell.Contains("powershell", StringComparison.OrdinalIgnoreCase))
        {
            arguments = BuildPowerShellArguments();
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = Shell,
            Arguments = arguments,
            UseShellExecute = false,
            WorkingDirectory = workDir,
            CreateNoWindow = false,
            StandardErrorEncoding = System.Text.Encoding.UTF8,
            StandardOutputEncoding = System.Text.Encoding.UTF8
        };

        try
        {
            _process = Process.Start(startInfo);
            if (_process == null)
                throw new Exception("Failed to start process");

            _process.WaitForInputIdle(5000);

            IntPtr hwnd = IntPtr.Zero;
            for (int i = 0; i < 50; i++)
            {
                _process.Refresh();
                hwnd = _process.MainWindowHandle;
                if (hwnd != IntPtr.Zero) break;
                Thread.Sleep(100);
            }

            if (hwnd == IntPtr.Zero)
                throw new Exception("Could not find console window handle");

            _childHwnd = hwnd;

            int style = GetWindowLong(hwnd, GWL_STYLE);
            style &= ~(WS_CAPTION | WS_THICKFRAME | WS_BORDER | WS_DLGFRAME |
                       WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_POPUP);
            SetWindowLong(hwnd, GWL_STYLE, style);

            int exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
            exStyle &= ~(WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE | WS_EX_DLGMODALFRAME);
            SetWindowLong(hwnd, GWL_EXSTYLE, exStyle);

            SetParent(hwnd, hwndParent.Handle);
            ShowWindow(hwnd, SW_SHOW);

            return new HandleRef(this, hwnd);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to start terminal ({Shell}):\n{ex.Message}", "QuadTerminal Error",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return new HandleRef(this, IntPtr.Zero);
        }
    }

    private string BuildPowerShellArguments()
    {
        string cwdPlaceholder = "__QT_CWD__";
        string script = """
            $_qt_cwd = '__QT_CWD__'
            $_qt_op = if ($function:prompt) { $function:prompt.Clone() } else { $null }
            function prompt {
                try { (Get-Location).Path | Out-File -LiteralPath $_qt_cwd -Encoding utf8 -ErrorAction SilentlyContinue } catch {}
                if ($_qt_op) { & $_qt_op } else { 'PS ' + (Get-Location).Path + '> ' }
            }
            Clear-Host
            """.Replace(cwdPlaceholder, _cwdFile);
        _initScriptFile = Path.Combine(Path.GetTempPath(), $"QuadTerminal_init{_paneIndex}.ps1");
        File.WriteAllText(_initScriptFile, script);
        return $"-NoExit -ExecutionPolicy Bypass -File \"{_initScriptFile}\"";
    }

    protected override void DestroyWindowCore(HandleRef hwnd)
    {
        if (_process != null && !_process.HasExited)
        {
            try { _process.Kill(entireProcessTree: true); } catch { }
        }
        _childHwnd = IntPtr.Zero;

        try { if (_cwdFile != null) File.Delete(_cwdFile); } catch { }
        try { if (_initScriptFile != null) File.Delete(_initScriptFile); } catch { }
    }

    protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
    {
        base.OnRenderSizeChanged(sizeInfo);
        ResizeChild();
    }

    private void ResizeChild()
    {
        if (_childHwnd == IntPtr.Zero) return;
        var w = (int)Math.Max(1, ActualWidth);
        var h = (int)Math.Max(1, ActualHeight);
        MoveWindow(_childHwnd, 0, 0, w, h, true);
    }

    #region Win32

    private const int GWL_STYLE = -16;
    private const int GWL_EXSTYLE = -20;
    private const int WS_CAPTION = 0x00C00000;
    private const int WS_THICKFRAME = 0x00040000;
    private const int WS_BORDER = 0x00800000;
    private const int WS_DLGFRAME = 0x00400000;
    private const int WS_SYSMENU = 0x00080000;
    private const int WS_MAXIMIZEBOX = 0x00010000;
    private const int WS_MINIMIZEBOX = 0x00020000;
    private const int WS_POPUP = unchecked((int)0x80000000);
    private const int WS_EX_WINDOWEDGE = 0x00000100;
    private const int WS_EX_CLIENTEDGE = 0x00000200;
    private const int WS_EX_STATICEDGE = 0x00020000;
    private const int WS_EX_DLGMODALFRAME = 0x00000001;
    private const int SW_SHOW = 5;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int GetWindowLong(IntPtr hWnd, int nIndex);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    private static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);

    [DllImport("user32.dll")]
    private static extern IntPtr SetFocus(IntPtr hWnd);

    #endregion
}
