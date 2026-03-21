using System;
using System.Collections.Generic;
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

        public MainWindow()
        {
            InitializeComponent();
            OutputPathTxt.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyVideos), "YoutubeExtractor");
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

            ((Button)sender).IsEnabled = false;
            DownloadProgress.Value = 0;
            DownloadProgress.Maximum = urls.Count;

            int successCount = 0;
            int failCount = 0;

            foreach (var url in urls)
            {
                try
                {
                    StatusTxt.Text = $"Downloading: {url}";
                    var video = await _youtube.Videos.GetAsync(url);
                    
                    // Sanitize file name
                    string fileName = string.Join("_", video.Title.Split(Path.GetInvalidFileNameChars())) + ".mp4";
                    string filePath = Path.Combine(outputDir, fileName);

                    var streamManifest = await _youtube.Videos.Streams.GetManifestAsync(video.Id);
                    
                    // Get highest quality muxed stream (video + audio in one file)
                    // For 1080p+, you'd need FFmpeg to mux separate video/audio streams.
                    // This library provides muxed streams up to 720p usually.
                    var streamInfo = streamManifest.GetMuxedStreams().GetWithHighestVideoQuality();

                    if (streamInfo != null)
                    {
                        await _youtube.Videos.Streams.DownloadAsync(streamInfo, filePath, new Progress<double>(p => {
                            // Individual progress could be shown here if needed
                        }));
                        successCount++;
                    }
                    else
                    {
                        failCount++;
                    }
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
    }
}
