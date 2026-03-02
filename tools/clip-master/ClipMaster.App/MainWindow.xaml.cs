using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;
using ClipMaster.App.Core;

namespace ClipMaster.App
{
    public partial class MainWindow : Window
    {
        private FFMpegHelper _ffmpeg;
        private List<string> _videos = new List<string>();

        public MainWindow()
        {
            InitializeComponent();
            _ffmpeg = new FFMpegHelper("ffmpeg"); // Assumes ffmpeg is in path or in local folder
        }

        private void AddVideos_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Multiselect = true,
                Filter = "Video Files|*.mp4;*.avi;*.mov;*.mkv|All files|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                foreach (var file in openFileDialog.FileNames)
                {
                    if (!_videos.Contains(file))
                    {
                        _videos.Add(file);
                        VideoList.Items.Add(Path.GetFileName(file));
                    }
                }
            }
        }

        private void ClearVideos_Click(object sender, RoutedEventArgs e)
        {
            _videos.Clear();
            VideoList.Items.Clear();
        }

        private void SelectAudio_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Audio Files|*.mp3;*.wav;*.m4a;*.aac|All files|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                AudioPathTxt.Text = openFileDialog.FileName;
            }
        }

        private void BrowseOutput_Click(object sender, RoutedEventArgs e)
        {
            // Use WinForms FolderBrowserDialog or a modern alternative
            var dialog = new Microsoft.Win32.OpenFolderDialog();
            if (dialog.ShowDialog() == true)
            {
                OutputPathTxt.Text = dialog.FolderName;
            }
        }

        private async void StartBatch_Click(object sender, RoutedEventArgs e)
        {
            if (_videos.Count == 0 || string.IsNullOrEmpty(AudioPathTxt.Text) || string.IsNullOrEmpty(OutputPathTxt.Text))
            {
                MessageBox.Show("Please select videos, audio, and output directory.", "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!double.TryParse(FadeInTxt.Text, out double fadeIn) || !double.TryParse(FadeOutTxt.Text, out double fadeOut))
            {
                MessageBox.Show("Invalid Fade values.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            double volume = VolumeSlider.Value;
            string audioPath = AudioPathTxt.Text;
            string outputDir = OutputPathTxt.Text;

            GlobalProgressBar.Value = 0;
            GlobalProgressBar.Maximum = _videos.Count;
            StatusTxt.Text = "Processing...";

            int successCount = 0;
            int failCount = 0;

            foreach (var video in _videos)
            {
                try
                {
                    string fileName = Path.GetFileNameWithoutExtension(video);
                    string ext = Path.GetExtension(video);
                    string outputPath = Path.Combine(outputDir, $"{fileName}_edited{ext}");

                    double duration = await _ffmpeg.GetDurationAsync(video);
                    bool success = await _ffmpeg.MergeAudioToVideoAsync(video, audioPath, outputPath, volume, fadeIn, fadeOut, duration);

                    if (success) successCount++;
                    else failCount++;
                }
                catch (Exception ex)
                {
                    failCount++;
                    Console.WriteLine(ex.Message);
                }

                GlobalProgressBar.Value++;
                StatusTxt.Text = $"Processing: {GlobalProgressBar.Value}/{_videos.Count}";
            }

            StatusTxt.Text = $"Done: {successCount} Success, {failCount} Failed";
            MessageBox.Show($"Batch processing completed.\nSuccess: {successCount}\nFailed: {failCount}", "Batch Complete", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }
}
