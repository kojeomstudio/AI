using System.Collections.ObjectModel;
using System.Text;
using System.Windows;
using MabinogiMacro.Models;
using MabinogiMacro.Services;
using MabinogiMacro.Native;
using Serilog;

namespace MabinogiMacro;

public partial class MainWindow : Window
{
    private CancellationTokenSource? _cts;
    private InputManager? _inputManager;
    private ActionProcessor? _actionProcessor;
    private ConfigManager _configManager;

    public MainWindow()
    {
        InitializeComponent();
        _configManager = new ConfigManager();

        TxtWindowTitle.Text = _configManager.AppConfig.WindowTitle;
        TxtTickInterval.Text = _configManager.AppConfig.TickInterval.ToString();
        TxtConfidence.Text = _configManager.AppConfig.ConfidenceThreshold.ToString();

        var method = _configManager.ActionConfig.InputMethod?.ToLowerInvariant();
        if (method == "sendinput") CmbInputMethod.SelectedIndex = 0;
        else if (method == "send_message" || method == "sendmessage") CmbInputMethod.SelectedIndex = 2;
        else CmbInputMethod.SelectedIndex = 1;

        UpdateStatus("Ready");
        LogToUi("Application initialized. Select settings and click Start.");
    }

    private void BtnStart_Click(object sender, RoutedEventArgs e)
    {
        if (_cts != null && !_cts.IsCancellationRequested)
        {
            LogToUi("Macro is already running.");
            return;
        }

        var windowTitle = TxtWindowTitle.Text.Trim();
        if (string.IsNullOrEmpty(windowTitle))
        {
            MessageBox.Show("Window title cannot be empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!double.TryParse(TxtTickInterval.Text, out var tickInterval) || tickInterval < 0.1)
        {
            MessageBox.Show("Tick interval must be a positive number (>= 0.1).", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!double.TryParse(TxtConfidence.Text, out var confidence) || confidence is < 0 or > 1)
        {
            MessageBox.Show("Confidence must be between 0 and 1.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        _configManager.AppConfig.WindowTitle = windowTitle;
        _configManager.AppConfig.TickInterval = tickInterval;
        _configManager.AppConfig.ConfidenceThreshold = confidence;
        _configManager.ActionConfig.InputMethod = CmbInputMethod.SelectedIndex switch
        {
            0 => "sendinput",
            1 => "postmessage",
            _ => "send_message",
        };

        _inputManager = new InputManager(windowTitle);
        _actionProcessor = new ActionProcessor(_configManager.ActionConfig, _inputManager);

        BtnStart.IsEnabled = false;
        BtnStop.IsEnabled = true;
        BtnTest.IsEnabled = false;
        TxtWindowTitle.IsEnabled = false;

        _cts = new CancellationTokenSource();
        var token = _cts.Token;
        _ = RunMacroAsync(token);

        LogToUi($"Macro started. Window: {windowTitle}, Tick: {tickInterval}s, Confidence: {confidence}");
    }

    private async Task RunMacroAsync(CancellationToken token)
    {
        try
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    if (!_inputManager!.MonitorProcess())
                    {
                        UpdateStatus("Waiting for window...");
                        LogToUi("Window not found. Retrying in 5s...");
                        await Task.Delay(5000, token);
                        continue;
                    }

                    UpdateStatus("Running...");

                    var capture = new CaptureService();
                    var bitmap = capture.CaptureWindow(_inputManager.Hwnd);
                    if (bitmap == null)
                    {
                        LogToUi("Failed to capture window. Retrying in 5s...");
                        await Task.Delay(5000, token);
                        continue;
                    }

                    bitmap.Dispose();
                    _actionProcessor!.ProcessDetectedElements(new Dictionary<ElementType, DetectedElement>());
                }
                catch (Exception ex)
                {
                    LogToUi($"Error: {ex.Message}");
                    Log.Error(ex, "Macro loop error");
                }

                var tickMs = (int)(_configManager.AppConfig.TickInterval * 1000);
                await Task.Delay(tickMs, token);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected on stop
        }
        finally
        {
            Dispatcher.Invoke(() =>
            {
                BtnStart.IsEnabled = true;
                BtnStop.IsEnabled = false;
                BtnTest.IsEnabled = true;
                TxtWindowTitle.IsEnabled = true;
                UpdateStatus("Stopped");

                if (_actionProcessor != null)
                {
                    var stats = _actionProcessor.GetStats();
                    TxtStats.Text = $"Total actions: {stats.TotalActions}\n" +
                        string.Join("\n", stats.ActionCounts.Select(kv => $"  {kv.Key}: {kv.Value}"));
                }
            });

            LogToUi("Macro stopped.");
        }
    }

    private void BtnStop_Click(object sender, RoutedEventArgs e) => StopMacro();

    private void StopCommand_Executed(object sender, System.Windows.Input.ExecutedRoutedEventArgs e) => StopMacro();

    private void StopMacro()
    {
        if (_cts != null)
        {
            _cts.Cancel();
            LogToUi("Stop requested...");
        }
    }

    private void BtnTest_Click(object sender, RoutedEventArgs e)
    {
        var windowTitle = TxtWindowTitle.Text.Trim();
        if (string.IsNullOrEmpty(windowTitle)) return;

        var inputManager = new InputManager(windowTitle);
        if (inputManager.Hwnd == IntPtr.Zero)
        {
            MessageBox.Show($"Window '{windowTitle}' not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var info = inputManager.GetWindowInfo();
        if (info != null)
        {
            var centerX = info.Left + info.Width / 2;
            var centerY = info.Top + info.Height / 2;

            _actionProcessor = new ActionProcessor(_configManager.ActionConfig, inputManager);
            _ = Task.Run(() => _actionProcessor.TestInputMethods(centerX, centerY));
        }
    }

    private void BtnReload_Click(object sender, RoutedEventArgs e)
    {
        _configManager.Reload();
        TxtWindowTitle.Text = _configManager.AppConfig.WindowTitle;
        TxtTickInterval.Text = _configManager.AppConfig.TickInterval.ToString();
        TxtConfidence.Text = _configManager.AppConfig.ConfidenceThreshold.ToString();
        LogToUi("Configuration reloaded.");
    }

    private void UpdateStatus(string status)
    {
        Dispatcher.Invoke(() => TxtStatus.Text = status);
    }

    private void LogToUi(string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        Dispatcher.Invoke(() =>
        {
            TxtLog.AppendText($"[{timestamp}] {message}\n");
            TxtLog.ScrollToEnd();
        });
    }

    private void Window_Closed(object sender, EventArgs e)
    {
        _cts?.Cancel();
        Log.CloseAndFlush();
    }
}
