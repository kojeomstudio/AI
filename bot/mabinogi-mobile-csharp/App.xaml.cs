using System.Windows;
using MabinogiMacro.Services;

namespace MabinogiMacro;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        LogHelper.InitLogger();
    }
}
