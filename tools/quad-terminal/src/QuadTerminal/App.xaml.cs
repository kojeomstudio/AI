using System.Windows;
using QuadTerminal.Services;

namespace QuadTerminal;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        Logger.Init();
        Logger.Info("QuadTerminal starting...");
        base.OnStartup(e);
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Logger.Info($"QuadTerminal exiting (code={e.ApplicationExitCode})");
        base.OnExit(e);
    }
}
