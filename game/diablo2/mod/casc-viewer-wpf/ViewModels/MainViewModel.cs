using System;
using System.Collections.Generic;
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
        private string _searchMask = "data\\*";
        private string _statusText = "Ready";
        private string _loadingStage = "Ready";
        private bool _isLoading;
        private CascNode? _selectedNode;
        private int _totalFiles;
        private int _currentFiles;
        private double _progressValue;

        public string Version => $"v{System.Reflection.Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "1.0.0"}";
        public ObservableCollection<LogEntry> Logs => LogService.Instance.Logs;

        public string D2RPath
        {
            get => _d2rPath;
            set => SetProperty(ref _d2rPath, value);
        }

        public string SearchMask
        {
            get => _searchMask;
            set => SetProperty(ref _searchMask, value);
        }

        public string StatusText
        {
            get => _statusText;
            set => SetProperty(ref _statusText, value);
        }

        public string LoadingStage
        {
            get => _loadingStage;
            set => SetProperty(ref _loadingStage, value);
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
                MessageBox.Show("Please select a valid D2R directory.");
                return;
            }

            IsLoading = true;
            LoadingStage = "Phase 1: Opening Storage...";
            StatusText = "Initializing CASC storage...";
            LogService.Instance.Log($"Loading CASC (Mask: {SearchMask}) from {D2RPath}...");
            
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
                        // Get Total Blocks for estimate
                        uint lengthNeeded;
                        byte[] buffer = new byte[4];
                        if (CascLibWrapper.CascGetStorageInfo(hStorage, CascLibWrapper.CascStorageTotalFileCount, buffer, 4, out lengthNeeded))
                        {
                            TotalFiles = BitConverter.ToInt32(buffer, 0);
                        }

                        Application.Current.Dispatcher.Invoke(() => LoadingStage = "Phase 2: Mapping Virtual Files...");
                        UpdateStatus("Building file hierarchy...");
                        
                        PopulateTreeOptimized(hStorage);
                        
                        CascLibWrapper.CascCloseStorage(hStorage);
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
                MessageBox.Show($"Error: {ex.Message}");
                LogService.Instance.Log($"Critical error: {ex.Message}", LogLevel.Error);
            }
            finally
            {
                IsLoading = false;
                LoadingStage = "Ready";
            }
        }

        private void UpdateStatus(string message)
        {
            Application.Current.Dispatcher.Invoke(() => StatusText = message);
        }

        private void PopulateTreeOptimized(IntPtr hStorage)
        {
            CascLibWrapper.CASC_FIND_DATA findData = new CascLibWrapper.CASC_FIND_DATA();
            IntPtr hFind = CascLibWrapper.CascFindFirstFile(hStorage, SearchMask, ref findData, null);

            if (hFind != IntPtr.Zero)
            {
                int processed = 0;
                var tempRootNodes = new List<CascNode>();
                var tempRootLookup = new Dictionary<string, CascNode>(StringComparer.OrdinalIgnoreCase);

                do
                {
                    if (!string.IsNullOrEmpty(findData.szFileName))
                    {
                        AddFileToInternalTree(tempRootNodes, tempRootLookup, findData.szFileName, findData.dwFileSize);
                    }
                    processed++;
                    
                    if (processed % 5000 == 0)
                    {
                        UpdateStatusOnUI(processed);
                    }
                } while (CascLibWrapper.CascFindNextFile(hFind, ref findData));

                CascLibWrapper.CascFindClose(hFind);
                
                Application.Current.Dispatcher.Invoke(() => 
                {
                    foreach (var node in tempRootNodes)
                    {
                        CascNodes.Add(node);
                    }
                    CurrentFiles = processed;
                    ProgressValue = 100;
                    StatusText = $"Load Complete. {processed} files mapped.";
                    LogService.Instance.Log($"Mapping complete. {processed} files added to tree.");
                });
            }
            else
            {
                LogService.Instance.Log($"No files found matching mask: {SearchMask}", LogLevel.Warning);
            }
        }

        private void UpdateStatusOnUI(int processed)
        {
            Application.Current.Dispatcher.BeginInvoke(new Action(() => 
            {
                CurrentFiles = processed;
                StatusText = $"Discovered {processed} assets...";
            }));
        }

        private void AddFileToInternalTree(List<CascNode> rootList, Dictionary<string, CascNode> rootLookup, string filePath, uint fileSize)
        {
            string[] parts = filePath.Split(new char[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) return;

            CascNode currentNode;
            string firstPart = parts[0];

            // 1. Handle Root
            if (!rootLookup.TryGetValue(firstPart, out currentNode!))
            {
                currentNode = new CascNode { Name = firstPart, IsFile = (parts.Length == 1) };
                rootLookup[firstPart] = currentNode;
                rootList.Add(currentNode);
            }

            // 2. Handle Hierarchy
            for (int i = 1; i < parts.Length; i++)
            {
                string part = parts[i];
                bool isFile = (i == parts.Length - 1);

                var nextNode = currentNode.GetOrCreateChild(part, isFile);
                currentNode = nextNode!;

                if (isFile)
                {
                    currentNode.Size = FormatSize(fileSize);
                    currentNode.Type = Path.GetExtension(part).ToUpper().TrimStart('.');
                    currentNode.FullPath = filePath;
                }
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
                    if (success) MessageBox.Show("File extracted.");
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
                MessageBox.Show("Folder extraction complete.");
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
