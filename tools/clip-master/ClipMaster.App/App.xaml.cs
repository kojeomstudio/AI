using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using ClipMaster.App.Core;

namespace ClipMaster.App;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        if (e.Args.Length > 0)
        {
            await RunCliAsync(e.Args);
            Shutdown();
            return;
        }

        base.OnStartup(e);
    }

    private async System.Threading.Tasks.Task RunCliAsync(string[] args)
    {
        string videoPath = null;
        string audioPath = null;
        string outputDir = null;
        double volume = 1.0;
        double fadeIn = 0;
        double fadeOut = 0;
        string suffix = "merged";

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--video": case "-v": videoPath = args[++i]; break;
                case "--audio": case "-a": audioPath = args[++i]; break;
                case "--output": case "-o": outputDir = args[++i]; break;
                case "--volume": double.TryParse(args[++i], System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out volume); break;
                case "--fade-in": double.TryParse(args[++i], System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out fadeIn); break;
                case "--fade-out": double.TryParse(args[++i], System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out fadeOut); break;
                case "--suffix": suffix = args[++i]; break;
                case "--help": case "-h":
                    PrintHelp();
                    return;
            }
        }

        if (string.IsNullOrEmpty(videoPath) || string.IsNullOrEmpty(audioPath) || string.IsNullOrEmpty(outputDir))
        {
            Console.WriteLine("Error: Missing required arguments.");
            PrintHelp();
            return;
        }

        var ffmpeg = new FFMpegHelper("ffmpeg");
        
        // Handle multiple videos if videoPath is a directory or a comma-separated list
        var videos = new List<string>();
        if (Directory.Exists(videoPath))
        {
            videos.AddRange(Directory.GetFiles(videoPath, "*.mp4")
                .Concat(Directory.GetFiles(videoPath, "*.avi"))
                .Concat(Directory.GetFiles(videoPath, "*.mov"))
                .Concat(Directory.GetFiles(videoPath, "*.mkv")));
        }
        else if (File.Exists(videoPath))
        {
            videos.Add(videoPath);
        }
        else
        {
            Console.WriteLine($"Error: Video path not found: {videoPath}");
            return;
        }

        if (!Directory.Exists(outputDir)) Directory.CreateDirectory(outputDir);

        Console.WriteLine($"Processing {videos.Count} videos...");

        foreach (var video in videos)
        {
            string fileName = Path.GetFileNameWithoutExtension(video);
            string ext = Path.GetExtension(video);
            string outputPath = Path.Combine(outputDir, $"{fileName}_{suffix}{ext}");

            Console.WriteLine($"Merging: {Path.GetFileName(video)}");
            double duration = await ffmpeg.GetDurationAsync(video);
            bool success = await ffmpeg.MergeAudioToVideoAsync(video, audioPath, outputPath, volume, fadeIn, fadeOut, duration);

            if (success) Console.WriteLine($"Successfully saved to: {outputPath}");
            else Console.WriteLine($"Failed to merge: {video}");
        }

        Console.WriteLine("Done.");
    }

    private void PrintHelp()
    {
        Console.WriteLine("ClipMaster CLI Usage:");
        Console.WriteLine("  --video, -v <path>      Path to a video file or directory containing videos.");
        Console.WriteLine("  --audio, -a <path>      Path to the audio file to merge.");
        Console.WriteLine("  --output, -o <path>     Output directory.");
        Console.WriteLine("  --volume <value>        Audio volume (default: 1.0).");
        Console.WriteLine("  --fade-in <seconds>     Fade-in duration in seconds (default: 0).");
        Console.WriteLine("  --fade-out <seconds>    Fade-out duration in seconds (default: 0).");
        Console.WriteLine("  --suffix <string>       Suffix for the output file (default: merged).");
        Console.WriteLine("  --help, -h              Show this help message.");
    }
}

