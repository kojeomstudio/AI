using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace ClipMaster.App.Core
{
    public class FFMpegHelper
    {
        private readonly string _ffmpegPath;

        public FFMpegHelper(string ffmpegPath)
        {
            _ffmpegPath = ffmpegPath;
        }

        public async Task<bool> MergeAudioToVideoAsync(string videoPath, string audioPath, string outputPath, double volume = 1.0, double fadeIn = 0, double fadeOut = 0, double duration = 0)
        {
            string filter = $"[1:a]volume={volume}";
            if (fadeIn > 0) filter += $",afade=t=in:st=0:d={fadeIn}";
            if (fadeOut > 0 && duration > fadeOut) filter += $",afade=t=out:st={duration - fadeOut}:d={fadeOut}";
            filter += "[aout]";

            // Properly escape quotes for the command line
            string args = $"-i \"{videoPath}\" -i \"{audioPath}\" -filter_complex \"{filter}\" -map 0:v -map \"[aout]\" -c:v copy -shortest -y \"{outputPath}\"";

            return await RunFFMpegAsync(args);
        }

        private async Task<bool> RunFFMpegAsync(string arguments)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = _ffmpegPath,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (var process = new Process { StartInfo = startInfo })
            {
                process.Start();
                await Task.Run(() => process.WaitForExit());
                return process.ExitCode == 0;
            }
        }

        public async Task<double> GetDurationAsync(string filePath)
        {
            string ffprobePath = _ffmpegPath.Replace("ffmpeg.exe", "ffprobe.exe");
            if (!File.Exists(ffprobePath)) ffprobePath = "ffprobe";

            var startInfo = new ProcessStartInfo
            {
                FileName = ffprobePath,
                Arguments = $"-v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{filePath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using (var process = new Process { StartInfo = startInfo })
                {
                    process.Start();
                    string output = await process.StandardOutput.ReadToEndAsync();
                    process.WaitForExit();
                    if (double.TryParse(output.Trim(), out double duration))
                    {
                        return duration;
                    }
                }
            }
            catch { }
            return 0;
        }
    }
}
