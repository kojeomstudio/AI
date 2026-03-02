using System;
using System.Windows;

namespace CascViewerWPF
{
    public partial class App : System.Windows.Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            AppDomain.CurrentDomain.UnhandledException += (s, ev) => 
                LogFatalError(ev.ExceptionObject as Exception, "AppDomain.UnhandledException");
            
            System.Windows.Application.Current.DispatcherUnhandledException += (s, ev) => 
            {
                LogFatalError(ev.Exception, "Dispatcher.UnhandledException");
                ev.Handled = true;
            };

            if (IntPtr.Size != 8)
            {
                MessageBox.Show("This application must run as a 64-bit process to work with CascLib.dll.", "Architecture Mismatch", MessageBoxButton.OK, MessageBoxImage.Error);
                System.Windows.Application.Current.Shutdown();
            }
        }

        private void LogFatalError(Exception? ex, string source)
        {
            string message = ex?.ToString() ?? "Unknown error";
            MessageBox.Show($"A fatal error occurred ({source}):\n\n{message}", "Fatal Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
