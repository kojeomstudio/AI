using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;

namespace ExcelToJsonExporter
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
                AttachConsole(ATTACH_PARENT_PROCESS);
                RunCli(args);
            }
            else
            {
                var app = new App();
                app.InitializeComponent();
                app.Run();
            }
        }

        private static void RunCli(string[] args)
        {
            Console.WriteLine("=============================================");
            Console.WriteLine(" Excel to JSON Exporter (CLI Mode)");
            Console.WriteLine("=============================================");

            string? inputPath = null;
            string? outputPath = null;

            for (int i = 0; i < args.Length; i++)
            {
                if ((args[i] == "-i" || args[i] == "--input") && i + 1 < args.Length)
                    inputPath = args[++i];
                else if ((args[i] == "-o" || args[i] == "--output") && i + 1 < args.Length)
                    outputPath = args[++i];
            }

            if (string.IsNullOrEmpty(inputPath))
            {
                Console.WriteLine("Usage: ExcelToJsonExporter --cli -i <input.xlsx> [-o <output_dir>]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  -i, --input   Input Excel file path (.xlsx)");
                Console.WriteLine("  -o, --output  Output directory (default: same as input file)");
                return;
            }

            if (!System.IO.File.Exists(inputPath))
            {
                Console.WriteLine("Error: File not found: " + inputPath);
                return;
            }

            if (string.IsNullOrEmpty(outputPath))
            {
                outputPath = System.IO.Path.GetDirectoryName(inputPath);
            }

            var core = new ExcelToJsonCore();
            core.OnLog = (msg) => Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {msg}");

            try
            {
                core.Run(inputPath, outputPath!);
                Console.WriteLine("=============================================");
                Console.WriteLine(" Export completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
                Environment.Exit(1);
            }
        }
    }
}
