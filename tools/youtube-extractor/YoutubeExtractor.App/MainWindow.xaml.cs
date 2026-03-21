using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using YoutubeExplode;
using YoutubeExplode.Videos.Streams;

namespace YoutubeExtractor.App
{
    public partial class MainWindow : Window
    {
        private readonly YoutubeClient _youtube = new YoutubeClient();
        private string _ffmpegPath = "ffmpeg";

        public MainWindow()
        {
            InitializeComponent();
            OutputPathTxt.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyVideos), "YoutubeExtractor");
            DetectFFMpeg();
        }

        private void DetectFFMpeg()
        {
            // Try to find ffmpeg in tools/bin/clip-master
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string clipMasterBin = Path.GetFullPath(Path.Combine(baseDir, "..", "clip-master", "ffmpeg.exe"));
            
            if (File.Exists(clipMasterBin))
            {
                _ffmpegPath = clipMasterBin;
            }
        }

        private void Browse_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFolderDialog();
            if (dialog.ShowDialog() == true)
            {
                OutputPathTxt.Text = dialog.FolderName;
            }
        }

        private async void StartDownload_Click(object sender, RoutedEventArgs e)
        {
            var urls = UrlListTxt.Text.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries)
                                      .Select(u => u.Trim())
                                      .Where(u => !string.IsNullOrEmpty(u))
                                      .ToList();

            if (urls.Count == 0)
            {
                MessageBox.Show("Please enter at least one YouTube URL.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string outputDir = OutputPathTxt.Text;
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            bool isAudioOnly = AudioRb.IsChecked == true;
            ((Button)sender).IsEnabled = false;
            DownloadProgress.Value = 0;
            DownloadProgress.Maximum = urls.Count;

            int successCount = 0;
            int failCount = 0;

            foreach (var url in urls)
            {
                try
                {
                    StatusTxt.Text = $"Getting metadata: {url}";
                    var video = await _youtube.Videos.GetAsync(url);
                    string sanitizedTitle = string.Join("_", video.Title.Split(Path.GetInvalidFileNameChars()));
                    
                    var streamManifest = await _youtube.Videos.Streams.GetManifestAsync(video.Id);

                    if (isAudioOnly)
                    {
                        await DownloadAudioAsync(streamManifest, sanitizedTitle, outputDir);
                    }
                    else
                    {
                        await DownloadHighestVideoAsync(streamManifest, sanitizedTitle, outputDir);
                    }

                    successCount++;
                }
                catch (Exception ex)
                {
                    failCount++;
                    Console.WriteLine($"Error downloading {url}: {ex.Message}");
                }

                DownloadProgress.Value++;
            }

            StatusTxt.Text = "Finished";
            ((Button)sender).IsEnabled = true;
            MessageBox.Show($"Download Complete.\nSuccess: {successCount}\nFailed: {failCount}", "Complete", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private async Task DownloadAudioAsync(StreamManifest manifest, string title, string outputDir)
        {
            StatusTxt.Text = $"Downloading Audio: {title}";
            var audioStream = manifest.GetAudioOnlyStreams().GetWithHighestBitrate();
            string tempAudio = Path.Combine(outputDir, $"{Guid.NewGuid()}.tmp");
            string finalPath = Path.Combine(outputDir, $"{title}.mp3");

            await _youtube.Videos.Streams.DownloadAsync(audioStream, tempAudio);

            // Convert to MP3 using FFmpeg
            string args = $"-i \"{tempAudio}\" -q:a 0 -map a -y \"{finalPath}\"";
            await RunFFMpegAsync(args);

            if (File.Exists(tempAudio)) File.Delete(tempAudio);
        }

        private async Task DownloadHighestVideoAsync(StreamManifest manifest, string title, string outputDir)
        {
            StatusTxt.Text = $"Downloading Video & Audio: {title}";
            var videoStream = manifest.GetVideoOnlyStreams().GetWithHighestVideoQuality();
            var audioStream = manifest.GetAudioOnlyStreams().GetWithHighestBitrate();

            string tempVid = Path.Combine(outputDir, $"{Guid.NewGuid()}_v.tmp");
            string tempAud = Path.Combine(outputDir, $"{Guid.NewGuid()}_a.tmp");
            string finalPath = Path.Combine(outputDir, $"{title}.mp4");

            var downloadVideoTask = _youtube.Videos.Streams.DownloadAsync(videoStream, tempVid).AsTask();
            var downloadAudioTask = _youtube.Videos.Streams.DownloadAsync(audioStream, tempAud).AsTask();

            await Task.WhenAll(downloadVideoTask, downloadAudioTask);

            StatusTxt.Text = $"Muxing: {title}";
            // Mux video and audio using FFmpeg
            string args = $"-i \"{tempVid}\" -i \"{tempAud}\" -c copy -y \"{finalPath}\"";
            await RunFFMpegAsync(args);

            if (File.Exists(tempVid)) File.Delete(tempVid);
            if (File.Exists(tempAud)) File.Delete(tempAud);
        }

        private async Task RunFFMpegAsync(string arguments)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = _ffmpegPath,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true
            };

            using (var process = new Process { StartInfo = startInfo })
            {
                process.Start();
                await process.WaitForExitAsync();
            }
        }
    }
}
