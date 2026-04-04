using System.ComponentModel;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using QuadTerminal.Controls;
using QuadTerminal.Models;
using QuadTerminal.Services;

namespace QuadTerminal;

public partial class MainWindow : Window
{
    private readonly SettingsService _settingsService = new();
    private readonly AppSettings _settings;
    private readonly TerminalHost[] _terminals = new TerminalHost[4];
    private readonly TextBlock[] _headers = new TextBlock[4];

    public MainWindow()
    {
        InitializeComponent();
        _settings = _settingsService.Load();

        _terminals[0] = Terminal0;
        _terminals[1] = Terminal1;
        _terminals[2] = Terminal2;
        _terminals[3] = Terminal3;

        _headers[0] = Pane0Header;
        _headers[1] = Pane1Header;
        _headers[2] = Pane2Header;
        _headers[3] = Pane3Header;

        for (int i = 0; i < 4; i++)
        {
            _terminals[i].PaneIndex = i;
            _terminals[i].Shell = _settings.Panes[i].Shell;
            _terminals[i].WorkingDirectory = _settings.Panes[i].WorkingDirectory;
            UpdatePaneHeader(i, _settings.Panes[i].WorkingDirectory);
        }
    }

    private void Window_Loaded(object sender, RoutedEventArgs e)
    {
        ApplyLayout(_settings.Layout);
        RestoreWindowBounds();
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        Thread.Sleep(300);

        for (int i = 0; i < 4; i++)
        {
            _settings.Panes[i].WorkingDirectory = _terminals[i].GetCurrentDirectory();
        }

        if (WindowState != WindowState.Minimized)
        {
            _settings.WindowBounds = new WindowBoundsSettings
            {
                X = RestoreBounds.X,
                Y = RestoreBounds.Y,
                Width = RestoreBounds.Width,
                Height = RestoreBounds.Height,
                WindowState = WindowState == WindowState.Maximized ? "Maximized" : "Normal"
            };
        }

        _settingsService.Save(_settings);
    }

    private void UpdatePaneHeader(int index, string path)
    {
        _headers[index].Text = string.IsNullOrEmpty(path) ? "" : $" {index + 1}: {path}";
    }

    private void ApplyLayout(string layout)
    {
        bool showRight = layout is "Quad" or "DualH";
        bool showBottom = layout is "Quad" or "DualV";

        VSplitter.Visibility = showRight ? Visibility.Visible : Visibility.Collapsed;
        HSplitter.Visibility = showBottom ? Visibility.Visible : Visibility.Collapsed;

        RightCol.Width = showRight ? new GridLength(1, GridUnitType.Star) : new GridLength(0);
        Pane1Container.Visibility = showRight ? Visibility.Visible : Visibility.Collapsed;
        Pane3Container.Visibility = (showRight && showBottom) ? Visibility.Visible : Visibility.Collapsed;
        Pane2Container.Visibility = showBottom ? Visibility.Visible : Visibility.Collapsed;

        _settings.Layout = layout;
    }

    private void RestoreWindowBounds()
    {
        if (_settings.WindowBounds == null) return;
        var b = _settings.WindowBounds;
        Left = b.X;
        Top = b.Y;
        Width = b.Width;
        Height = b.Height;
        if (b.WindowState == "Maximized")
            WindowState = WindowState.Maximized;
    }

    private void LayoutSingle_Click(object sender, RoutedEventArgs e) => ApplyLayout("Single");
    private void LayoutDualH_Click(object sender, RoutedEventArgs e) => ApplyLayout("DualH");
    private void LayoutDualV_Click(object sender, RoutedEventArgs e) => ApplyLayout("DualV");
    private void LayoutQuad_Click(object sender, RoutedEventArgs e) => ApplyLayout("Quad");
}
