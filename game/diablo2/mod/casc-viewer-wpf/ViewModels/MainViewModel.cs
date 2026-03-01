using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using CascViewerWPF.Commands;
using CascViewerWPF.Models;

using CascViewerWPF.Services;

namespace CascViewerWPF.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private string _d2rPath = string.Empty;
        private string _statusText = "Ready";
        private bool _isLoading;
        private CascNode? _selectedNode;
        private int _totalFiles;
        private int _currentFiles;
        private double _progressValue;

        public ObservableCollection<LogEntry> Logs => LogService.Instance.Logs;

        public string D2RPath
        {
            get => _d2rPath;
            set => SetProperty(ref _d2rPath, value);
        }

        public string StatusText
        {
            get => _statusText;
            set => SetProperty(ref _statusText, value);
        }

        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                if (SetProperty(ref _isLoading, value))
                {
                    OnPropertyChanged(nameof(IsBusy));
                    OnPropertyChanged(nameof(CanBrowse));
                }
            }
        }

        public bool IsBusy => IsLoading;
        public bool CanBrowse => !IsLoading;

        public int TotalFiles
        {
            get => _totalFiles;
            set => SetProperty(ref _totalFiles, value);
        }

        public int CurrentFiles
        {
            get => _currentFiles;
            set => SetProperty(ref _currentFiles, value);
        }

        public double ProgressValue
        {
            get => _progressValue;
            set => SetProperty(ref _progressValue, value);
        }

        public CascNode? SelectedNode
        {
            get => _selectedNode;
            set
            {
                if (SetProperty(ref _selectedNode, value))
                {
                    OnPropertyChanged(nameof(CanExtract));
                }
            }
        }

        public bool CanExtract => SelectedNode != null;

        public ObservableCollection<CascNode> CascNodes { get; } = new ObservableCollection<CascNode>();

        public ICommand BrowseCommand { get; }
        public ICommand LoadCommand { get; }
        public ICommand ExtractCommand { get; }

        public MainViewModel()
        {
            BrowseCommand = new RelayCommand(_ => Browse(), _ => CanBrowse);
            LoadCommand = new RelayCommand(_ => LoadCasc(), _ => !string.IsNullOrEmpty(D2RPath) && !IsLoading);
            ExtractCommand = new RelayCommand(_ => ExtractSelected(), _ => CanExtract);
            
            LogService.Instance.Log("MainViewModel initialized.");
        }

        private void Browse()
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                D2RPath = dialog.SelectedPath;
                LogService.Instance.Log($"Path selected: {D2RPath}");
            }
        }

        private async void LoadCasc()
        {
            if (!Directory.Exists(D2RPath))
            {
                System.Windows.MessageBox.Show("Please select a valid D2R directory.");
                return;
            }

            IsLoading = true;
            StatusText = "Opening CASC storage...";
            LogService.Instance.Log($"Loading CASC from {D2RPath}...");
            CascNodes.Clear();
            ProgressValue = 0;
            CurrentFiles = 0;

            try
            {
                await Task.Run(() =>
                {
                    IntPtr hStorage;
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        // Get Total File Count
                        uint lengthNeeded;
                        byte[] buffer = new byte[4];
                        if (CascLibWrapper.CascGetStorageInfo(hStorage, CascLibWrapper.CascStorageFileCount, buffer, 4, out lengthNeeded))
                        {
                            TotalFiles = BitConverter.ToInt32(buffer, 0);
                            LogService.Instance.Log($"Total files in storage: {TotalFiles}");
                        }

                        UpdateStatus("Scanning files...");
                        PopulateTree(hStorage);
                        CascLibWrapper.CascCloseStorage(hStorage);
                        UpdateStatus("CASC Loaded Successfully.");
                    }
                    else
                    {
                        UpdateStatus("Failed to open CASC storage.");
                        LogService.Instance.Log("Failed to open CASC storage.", LogLevel.Error);
                    }
                });
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}");
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
                LogService.Instance.Log($"Critical error: {ex.Message}", LogLevel.Error);
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void UpdateStatus(string message)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() => StatusText = message);
        }

        private void PopulateTree(IntPtr hStorage)
        {
            CascLibWrapper.CASC_FIND_DATA findData = new CascLibWrapper.CASC_FIND_DATA();
            IntPtr hFind = CascLibWrapper.CascFindFirstFile(hStorage, "*", ref findData, null);

            if (hFind != IntPtr.Zero)
            {
                int processed = 0;
                do
                {
                    if (!string.IsNullOrEmpty(findData.szFileName))
                    {
                        var fileName = findData.szFileName;
                        var fileSize = findData.dwFileSize;
                        System.Windows.Application.Current.Dispatcher.Invoke(() => AddFileToTree(fileName, fileSize));
                    }
                    processed++;
                    
                    if (processed % 100 == 0 || processed == TotalFiles)
                    {
                        double progress = TotalFiles > 0 ? (double)processed / TotalFiles * 100 : 0;
                        System.Windows.Application.Current.Dispatcher.Invoke(() => 
                        {
                            CurrentFiles = processed;
                            ProgressValue = progress;
                            StatusText = $"Scanning: {processed} / {TotalFiles} ({progress:F1}%)";
                        });
                    }
                } while (CascLibWrapper.CascFindNextFile(hFind, ref findData));

                CascLibWrapper.CascFindClose(hFind);
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

        private async void ExtractSelected()
        {
            if (SelectedNode == null) return;

            if (SelectedNode.IsFile)
            {
                ExtractSingleFile(SelectedNode);
            }
            else
            {
                await ExtractFolder(SelectedNode);
            }
        }

        private void ExtractSingleFile(CascNode node)
        {
            var saveDialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = node.Name,
                Filter = $"All Files (*.*)|*.*"
            };

            if (saveDialog.ShowDialog() == true)
            {
                LogService.Instance.Log($"Extracting file: {node.FullPath}");
                IntPtr hStorage;
                if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                {
                    bool success = CascLibWrapper.CascExtractFile(hStorage, node.FullPath!, saveDialog.FileName, 0);
                    CascLibWrapper.CascCloseStorage(hStorage);
                    if (success) System.Windows.MessageBox.Show("File extracted.");
                }
            }
        }

        private async Task ExtractFolder(CascNode folderNode)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = $"Select target folder to extract '{folderNode.Name}'"
            };

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                string targetBase = Path.Combine(dialog.SelectedPath, folderNode.Name!);
                IsLoading = true;
                StatusText = $"Extracting folder: {folderNode.Name}...";
                
                await Task.Run(() => 
                {
                    IntPtr hStorage;
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        ExtractNodeRecursive(hStorage, folderNode, targetBase);
                        CascLibWrapper.CascCloseStorage(hStorage);
                        UpdateStatus("Folder extraction complete.");
                    }
                });
                IsLoading = false;
                System.Windows.MessageBox.Show("Folder extraction complete.");
            }
        }

        private void ExtractNodeRecursive(IntPtr hStorage, CascNode node, string targetPath)
        {
            if (node.IsFile)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);
                CascLibWrapper.CascExtractFile(hStorage, node.FullPath!, targetPath, 0);
            }
            else
            {
                foreach (var child in node.Children)
                {
                    ExtractNodeRecursive(hStorage, child, Path.Combine(targetPath, child.Name!));
                }
            }
        }
    }
}
