using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;

namespace DataGenerator
{
    public class Program
    {
        [DllImport("kernel32.dll")]
        private static extern bool AttachConsole(int dwProcessId);
        private const int ATTACH_PARENT_PROCESS = -1;

        [STAThread]
        public static void Main(string[] args)
        {
            if (args.Length > 0 && (args.Contains("--cli") || args.Contains("-c")))
            {
                // Attach to parent console if running as CLI
                AttachConsole(ATTACH_PARENT_PROCESS);
                
                RunCli();
            }
            else
            {
                // Run WPF App
                var app = new App();
                app.InitializeComponent();
                app.Run();
            }
        }

        private static void RunCli()
        {
            Console.WriteLine("========================================");
            Console.WriteLine(" UnrealWorld Data Generator (CLI Mode)");
            Console.WriteLine("========================================");

            var generator = new DataGeneratorCore();
            generator.OnLog = (msg) => Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {msg}");
            
            generator.Run();

            Console.WriteLine("========================================");
            Console.WriteLine(" Generation process finished.");
        }
    }
}
