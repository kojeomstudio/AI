using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Text;

namespace ClipMaster.App.Core
{
    public class FFMpegHelper
    {
        private readonly string _ffmpegPath;
        private readonly string _logPath;

        public FFMpegHelper(string ffmpegPath)
        {
            _ffmpegPath = ffmpegPath;
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _logPath = Path.Combine(baseDir, "clip_master.log");
        }

        private void Log(string message)
        {
            try
            {
                string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                File.AppendAllText(_logPath, $"[{timestamp}] {message}{Environment.NewLine}");
            }
            catch { }
        }

        public async Task<bool> MergeAudioToVideoAsync(string videoPath, string audioPath, string outputPath, double volume = 1.0, double fadeIn = 0, double fadeOut = 0, double duration = 0)
        {
            Log($"Starting Merge: Video={videoPath}, Audio={audioPath}, Output={outputPath}");
            
            // Re-calculate duration if it's 0
            if (duration <= 0)
            {
                duration = await GetDurationAsync(videoPath);
                Log($"Re-calculated Duration: {duration}");
            }

            string volumeStr = volume.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
            string filter = $"[1:a]volume={volumeStr}";
            
            if (fadeIn > 0)
            {
                string fadeInStr = fadeIn.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
                filter += $",afade=t=in:st=0:d={fadeInStr}";
            }
            
            if (fadeOut > 0 && duration > fadeOut)
            {
                string start = (duration - fadeOut).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
                string fadeOutStr = fadeOut.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
                filter += $",afade=t=out:st={start}:d={fadeOutStr}";
            }
            filter += "[aout]";

            // Properly escape quotes for the command line
            // Use -c:a aac to ensure compatibility if we are filtering
            string args = $"-i \"{videoPath}\" -i \"{audioPath}\" -filter_complex \"{filter}\" -map 0:v -map \"[aout]\" -c:v copy -c:a aac -shortest -y \"{outputPath}\"";
            
            Log($"Running FFmpeg with args: {args}");
            bool result = await RunFFMpegAsync(args);
            Log($"Merge Result: {result}");
            return result;
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

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            using (var process = new Process { StartInfo = startInfo })
            {
                process.OutputDataReceived += (s, e) => { if (e.Data != null) outputBuilder.AppendLine(e.Data); };
                process.ErrorDataReceived += (s, e) => { if (e.Data != null) errorBuilder.AppendLine(e.Data); };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                
                await Task.Run(() => process.WaitForExit());
                
                string output = outputBuilder.ToString();
                string error = errorBuilder.ToString();
                
                if (!string.IsNullOrEmpty(output)) Log("FFmpeg Output: " + output);
                if (!string.IsNullOrEmpty(error)) Log("FFmpeg Error: " + error);
                
                return process.ExitCode == 0;
            }
        }

        public async Task<double> GetDurationAsync(string filePath)
        {
            string ffprobePath = "ffprobe";
            if (_ffmpegPath.EndsWith("ffmpeg.exe", StringComparison.OrdinalIgnoreCase))
            {
                ffprobePath = _ffmpegPath.Replace("ffmpeg.exe", "ffprobe.exe");
            }
            else if (_ffmpegPath.EndsWith("ffmpeg", StringComparison.OrdinalIgnoreCase))
            {
                ffprobePath = _ffmpegPath.Replace("ffmpeg", "ffprobe");
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = ffprobePath,
                Arguments = $"-v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{filePath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            Log($"Running FFprobe: {ffprobePath} {startInfo.Arguments}");

            try
            {
                using (var process = new Process { StartInfo = startInfo })
                {
                    process.Start();
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    process.WaitForExit();
                    
                    if (!string.IsNullOrEmpty(error)) Log("FFprobe Error: " + error);

                    if (double.TryParse(output.Trim(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double duration))
                    {
                        return duration;
                    }
                }
            }
            catch (Exception ex)
            {
                Log("FFprobe Exception: " + ex.Message);
            }
            return 0;
        }
    }
}
