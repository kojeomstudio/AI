using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace CascViewerWPF
{
    public partial class MainWindow : Window
    {
        public ObservableCollection<CascNode> CascNodes { get; set; } = new ObservableCollection<CascNode>();

        public MainWindow()
        {
            InitializeComponent();
            CascTreeView.ItemsSource = CascNodes;
        }

        private void BrowseButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                PathTextBox.Text = dialog.SelectedPath;
            }
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            string path = PathTextBox.Text;
            if (string.IsNullOrEmpty(path) || !Directory.Exists(path))
            {
                System.Windows.MessageBox.Show("Please select a valid D2R directory.");
                return;
            }

            CascNodes.Clear();
            StatusText.Text = "Loading CASC...";

            try
            {
                IntPtr hStorage;
                bool success = CascLibWrapper.CascOpenStorage(path, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage);
                
                if (success)
                {
                    StatusText.Text = "CASC Storage Opened. Scanning files...";
                    PopulateTree(hStorage);
                    CascLibWrapper.CascCloseStorage(hStorage);
                    StatusText.Text = "CASC Loaded Successfully.";
                }
                else
                {
                    StatusText.Text = "Failed to open CASC storage.";
                    System.Windows.MessageBox.Show("Failed to open CASC storage.");
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
            }
        }

        private void PopulateTree(IntPtr hStorage)
        {
            CascLibWrapper.CASC_FIND_DATA findData = new CascLibWrapper.CASC_FIND_DATA();
            IntPtr hFind = CascLibWrapper.CascFindFirstFile(hStorage, "*", ref findData, null);

            if (hFind != IntPtr.Zero)
            {
                do
                {
                    if (!string.IsNullOrEmpty(findData.szFileName))
                    {
                        AddFileToTree(findData.szFileName, findData.dwFileSize);
                    }
                } while (CascLibWrapper.CascFindNextFile(hFind, ref findData));

                CascLibWrapper.CascFindClose(hFind);
            }
        }

        private CascNode? selectedNode;

        private void CascTreeView_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            selectedNode = e.NewValue as CascNode;
            if (selectedNode != null && selectedNode.IsFile)
            {
                FileNameText.Text = $"Name: {selectedNode.Name}";
                FileSizeText.Text = $"Size: {selectedNode.Size}";
                FileTypeText.Text = $"Type: {selectedNode.Type}";
                ExtractButton.IsEnabled = true;
            }
            else
            {
                FileNameText.Text = "Name: -";
                FileSizeText.Text = "Size: -";
                FileTypeText.Text = "Type: -";
                ExtractButton.IsEnabled = false;
            }
        }

        private void ExtractFile_Click(object sender, RoutedEventArgs e)
        {
            if (selectedNode == null || !selectedNode.IsFile || string.IsNullOrEmpty(selectedNode.FullPath))
            {
                return;
            }

            var saveDialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = selectedNode.Name,
                Filter = $"All Files (*.*)|*.*"
            };

            if (saveDialog.ShowDialog() == true)
            {
                StatusText.Text = $"Extracting {selectedNode.Name}...";
                
                IntPtr hStorage;
                if (CascLibWrapper.CascOpenStorage(PathTextBox.Text, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                {
                    bool success = CascLibWrapper.CascExtractFile(hStorage, selectedNode.FullPath!, saveDialog.FileName, 0);
                    CascLibWrapper.CascCloseStorage(hStorage);

                    if (success)
                    {
                        StatusText.Text = "Extraction Complete.";
                        System.Windows.MessageBox.Show("File extracted successfully.");
                    }
                    else
                    {
                        StatusText.Text = "Extraction Failed.";
                        System.Windows.MessageBox.Show("Failed to extract file.");
                    }
                }
            }
        }

        private void AddFileToTree(string filePath, uint fileSize)
        {
            string[] parts = filePath.Split(new char[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries);
            ObservableCollection<CascNode> currentLevel = CascNodes;

            for (int i = 0; i < parts.Length; i++)
            {
                string part = parts[i];
                bool isFile = (i == parts.Length - 1);

                CascNode? node = null;
                foreach (var n in currentLevel)
                {
                    if (n.Name == part)
                    {
                        node = n;
                        break;
                    }
                }

                if (node == null)
                {
                    node = new CascNode { Name = part };
                    if (isFile)
                    {
                        node.Size = FormatSize(fileSize);
                        node.Type = Path.GetExtension(part).ToUpper().TrimStart('.');
                        node.FullPath = filePath;
                        node.IsFile = true;
                    }
                    currentLevel.Add(node);
                }

                currentLevel = node.Children;
            }
        }

        private string FormatSize(uint bytes)
        {
            string[] units = { "B", "KB", "MB", "GB" };
            double size = bytes;
            int unitIndex = 0;
            while (size >= 1024 && unitIndex < units.Length - 1)
            {
                size /= 1024;
                unitIndex++;
            }
            return $"{size:F2} {units[unitIndex]}";
        }
    }

    public class CascNode
    {
        public string? Name { get; set; }
        public string? Size { get; set; }
        public string? Type { get; set; }
        public string? FullPath { get; set; }
        public bool IsFile { get; set; }
        public ObservableCollection<CascNode> Children { get; set; } = new ObservableCollection<CascNode>();
    }
}
