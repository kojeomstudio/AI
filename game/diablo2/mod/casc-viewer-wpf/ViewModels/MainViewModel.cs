using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using CascViewerWPF.Commands;
using CascViewerWPF.Models;
using CascViewerWPF.Services;

namespace CascViewerWPF.ViewModels
{
    /// <summary>
    /// Main ViewModel responsible for managing the application state, 
    /// handling CASC storage operations, and coordinating UI updates.
    /// </summary>
    public class MainViewModel : ViewModelBase
    {
        #region Fields
        private string _d2rPath = string.Empty;
        private string _searchMask = "*";
        private string _statusText = "Ready";
        private string _loadingStage = "Ready";
        private bool _isLoading;
        private CascNode? _selectedNode;
        private int _totalFiles;
        private int _currentFiles;
        private double _progressValue;
        #endregion

        #region Properties
        /// <summary>
        /// Gets the current application version.
        /// </summary>
        public string Version => $"v{System.Reflection.Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "1.0.0"}";

        /// <summary>
        /// Gets the collection of log entries for the Log Viewer.
        /// </summary>
        public ObservableCollection<LogEntry> Logs => LogService.Instance.Logs;

        /// <summary>
        /// Gets or sets the path to the Diablo II Resurrected installation.
        /// </summary>
        public string D2RPath
        {
            get => _d2rPath;
            set => SetProperty(ref _d2rPath, value);
        }

        /// <summary>
        /// Gets or sets the mask used for searching files within CASC storage.
        /// </summary>
        public string SearchMask
        {
            get => _searchMask;
            set => SetProperty(ref _searchMask, value);
        }

        /// <summary>
        /// Gets or sets the current status message displayed in the status bar.
        /// </summary>
        public string StatusText
        {
            get => _statusText;
            set => SetProperty(ref _statusText, value);
        }

        /// <summary>
        /// Gets or sets the current loading stage name (e.g., Phase 1, Phase 2).
        /// </summary>
        public string LoadingStage
        {
            get => _loadingStage;
            set => SetProperty(ref _loadingStage, value);
        }

        /// <summary>
        /// Gets or sets a value indicating whether a long-running operation is in progress.
        /// </summary>
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

        /// <summary>
        /// Total files expected to be processed (for progress estimation).
        /// </summary>
        public int TotalFiles
        {
            get => _totalFiles;
            set => SetProperty(ref _totalFiles, value);
        }

        /// <summary>
        /// Current number of files processed.
        /// </summary>
        public int CurrentFiles
        {
            get => _currentFiles;
            set => SetProperty(ref _currentFiles, value);
        }

        /// <summary>
        /// Percentage progress (0-100).
        /// </summary>
        public double ProgressValue
        {
            get => _progressValue;
            set => SetProperty(ref _progressValue, value);
        }

        /// <summary>
        /// Currently selected node in the TreeView.
        /// </summary>
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

        /// <summary>
        /// Root collection of CASC nodes for the TreeView.
        /// </summary>
        public ObservableCollection<CascNode> CascNodes { get; } = new ObservableCollection<CascNode>();
        #endregion

        #region Commands
        public ICommand BrowseCommand { get; }
        public ICommand LoadCommand { get; }
        public ICommand ExtractCommand { get; }
        public ICommand CopyPathCommand { get; }
        public ICommand ClearLogsCommand { get; }
        #endregion

        public MainViewModel()
        {
            BrowseCommand = new RelayCommand(_ => Browse(), _ => CanBrowse);
            LoadCommand = new RelayCommand(_ => LoadCasc(), _ => !string.IsNullOrEmpty(D2RPath) && !IsLoading);
            ExtractCommand = new RelayCommand(_ => ExtractSelected(), _ => CanExtract);
            CopyPathCommand = new RelayCommand(_ => CopyPath(), _ => SelectedNode != null);
            ClearLogsCommand = new RelayCommand(_ => LogService.Instance.Logs.Clear());
            
            LogService.Instance.Log("MainViewModel initialized.");
        }

        #region Methods
        /// <summary>
        /// Opens a folder browser dialog to select the D2R directory.
        /// </summary>
        private void Browse()
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                D2RPath = dialog.SelectedPath;
                LogService.Instance.Log($"Path selected: {D2RPath}");
            }
        }

        /// <summary>
        /// Asynchronously loads and maps the CASC storage based on the current path and mask.
        /// </summary>
        private async void LoadCasc()
        {
            if (!Directory.Exists(D2RPath))
            {
                System.Windows.MessageBox.Show("Please select a valid D2R directory.");
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
                    // Attempt to open the CASC storage locally.
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        LogService.Instance.Log("CASC storage opened successfully.");
                        
                        // Fetch total file count for progress estimation.
                        uint lengthNeeded;
                        byte[] buffer = new byte[4];
                        if (CascLibWrapper.CascGetStorageInfo(hStorage, CascLibWrapper.CascStorageTotalFileCount, buffer, 4, out lengthNeeded))
                        {
                            TotalFiles = BitConverter.ToInt32(buffer, 0);
                            LogService.Instance.Log($"Storage file count (estimate): {TotalFiles}");
                        }

                        // Phase 2: Iterate through files and build the hierarchical tree.
                        System.Windows.Application.Current.Dispatcher.Invoke(() => LoadingStage = "Phase 2: Mapping Virtual Files...");
                        UpdateStatus("Building file hierarchy...");
                        
                        PopulateTreeOptimized(hStorage);
                        
                        CascLibWrapper.CascCloseStorage(hStorage);
                    }
                    else
                    {
                        UpdateStatus("Failed to open CASC storage.");
                        LogService.Instance.Log("Failed to open CASC storage. Check if the path is correct and files are not locked.", LogLevel.Error);
                    }
                });
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}");
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
                LogService.Instance.Log($"Critical error during CASC load: {ex.Message}", LogLevel.Error);
            }
            finally
            {
                IsLoading = false;
                LoadingStage = "Ready";
            }
        }

        /// <summary>
        /// Updates the status text on the UI thread.
        /// </summary>
        private void UpdateStatus(string message)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() => StatusText = message);
        }

        /// <summary>
        /// Iterates through the CASC storage files using the search mask and populates the internal tree structure.
        /// </summary>
        private void PopulateTreeOptimized(IntPtr hStorage)
        {
            CascLibWrapper.CASC_FIND_DATA findData = new CascLibWrapper.CASC_FIND_DATA();
            IntPtr hFind = CascLibWrapper.CascFindFirstFile(hStorage, SearchMask, ref findData, null);

            if (hFind != IntPtr.Zero)
            {
                int processed = 0;
                int namedFiles = 0;
                var tempRootNodes = new List<CascNode>();
                var tempRootLookup = new Dictionary<string, CascNode>(StringComparer.OrdinalIgnoreCase);

                try
                {
                    do
                    {
                        string fileName = findData.szFileName;
                        if (!string.IsNullOrEmpty(fileName))
                        {
                            AddFileToInternalTree(tempRootNodes, tempRootLookup, fileName, findData.FileSize);
                            namedFiles++;
                        }
                        
                        processed++;
                        
                        // Periodic UI update to show progress without overwhelming the dispatcher.
                        if (processed % 10000 == 0)
                        {
                            UpdateStatusOnUI(processed);
                        }

                        // Safety break: If we exceed a reasonable number of files (e.g., 10 million), 
                        // something is likely wrong with the storage or finding logic.
                        if (processed > 10_000_000)
                        {
                            LogService.Instance.Log("Safety break triggered: processed over 10 million entries. Stopping scan.", LogLevel.Warning);
                            break;
                        }
                    } while (CascLibWrapper.CascFindNextFile(hFind, ref findData));
                }
                finally
                {
                    CascLibWrapper.CascFindClose(hFind);
                }
                
                // Finalize the tree on the UI thread.
                System.Windows.Application.Current.Dispatcher.Invoke(() => 
                {
                    foreach (var node in tempRootNodes)
                    {
                        CascNodes.Add(node);
                    }
                    CurrentFiles = processed;
                    ProgressValue = 100;
                    StatusText = $"Load Complete. {processed} entries ( {namedFiles} named files) mapped.";
                    LogService.Instance.Log($"Mapping complete. Processed {processed} entries, found {namedFiles} named files.");
                });
            }
            else
            {
                LogService.Instance.Log($"No files found matching mask: {SearchMask}", LogLevel.Warning);
                UpdateStatus("No files found.");
            }
        }

        private void UpdateStatusOnUI(int processed)
        {
            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() => 
            {
                CurrentFiles = processed;
                StatusText = $"Discovered {processed} assets...";
            }));
        }

        /// <summary>
        /// Parses a virtual file path and adds it to the hierarchical tree structure.
        /// </summary>
        private void AddFileToInternalTree(List<CascNode> rootList, Dictionary<string, CascNode> rootLookup, string filePath, ulong fileSize)
        {
            string[] parts = filePath.Split(new char[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) return;

            CascNode currentNode;
            string firstPart = parts[0];

            // 1. Resolve or create the root level node.
            if (!rootLookup.TryGetValue(firstPart, out currentNode!))
            {
                currentNode = new CascNode { Name = firstPart, IsFile = (parts.Length == 1) };
                rootLookup[firstPart] = currentNode;
                rootList.Add(currentNode);
            }

            // 2. Traverse or build the remaining hierarchy.
            for (int i = 1; i < parts.Length; i++)
            {
                string part = parts[i];
                bool isFile = (i == parts.Length - 1);

                var nextNode = currentNode.GetOrCreateChild(part, isFile);
                currentNode = nextNode!;

                // If it's the leaf node (file), set its metadata.
                if (isFile)
                {
                    currentNode.Size = FormatSize(fileSize);
                    currentNode.Type = Path.GetExtension(part).ToUpper().TrimStart('.');
                    currentNode.FullPath = filePath;
                }
            }
        }

        /// <summary>
        /// Formats a file size in bytes into a human-readable string (B, KB, MB, GB).
        /// </summary>
        private string FormatSize(ulong bytes)
        {
            if (bytes == 0xFFFFFFFFFFFFFFFF) return "N/A";

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

        /// <summary>
        /// Handles the extraction logic for the currently selected node.
        /// </summary>
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

        /// <summary>
        /// Copies the full virtual path of the selected node to the clipboard.
        /// </summary>
        private void CopyPath()
        {
            if (SelectedNode?.FullPath != null)
            {
                System.Windows.Clipboard.SetText(SelectedNode.FullPath);
                LogService.Instance.Log($"Path copied to clipboard: {SelectedNode.FullPath}");
            }
            else if (SelectedNode != null)
            {
                System.Windows.Clipboard.SetText(SelectedNode.Name ?? string.Empty);
                LogService.Instance.Log($"Name copied to clipboard: {SelectedNode.Name}");
            }
        }

        /// <summary>
        /// Opens a save dialog to extract a single file from CASC.
        /// </summary>
        private void ExtractSingleFile(CascNode node)
        {
            var saveDialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = node.Name,
                Filter = $"All Files (*.*)|*.*",
                Title = $"Extract {node.Name}"
            };

            if (saveDialog.ShowDialog() == true)
            {
                LogService.Instance.Log($"Extracting file: {node.FullPath}");
                IntPtr hStorage;
                if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                {
                    bool success = CascLibWrapper.CascExtractFile(hStorage, node.FullPath!, saveDialog.FileName, 0);
                    CascLibWrapper.CascCloseStorage(hStorage);
                    
                    if (success)
                    {
                        LogService.Instance.Log($"Extraction successful: {saveDialog.FileName}");
                        System.Windows.MessageBox.Show("File extracted successfully.");
                    }
                    else
                    {
                        LogService.Instance.Log($"Extraction failed for: {node.FullPath}", LogLevel.Error);
                        System.Windows.MessageBox.Show("Failed to extract file.");
                    }
                }
            }
        }

        /// <summary>
        /// Opens a folder browser to extract an entire virtual folder recursively.
        /// </summary>
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
                LogService.Instance.Log($"Extracting folder {folderNode.Name} to {targetBase}...");
                
                await Task.Run(() => 
                {
                    IntPtr hStorage;
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        ExtractNodeRecursive(hStorage, folderNode, targetBase);
                        CascLibWrapper.CascCloseStorage(hStorage);
                        UpdateStatus("Folder extraction complete.");
                        LogService.Instance.Log($"Folder extraction complete: {folderNode.Name}");
                    }
                });
                IsLoading = false;
                System.Windows.MessageBox.Show("Folder extraction complete.");
            }
        }

        /// <summary>
        /// Recursively extracts nodes from CASC storage to the local filesystem.
        /// </summary>
        private void ExtractNodeRecursive(IntPtr hStorage, CascNode node, string targetPath)
        {
            if (node.IsFile)
            {
                if (string.IsNullOrEmpty(node.FullPath)) return;

                Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);
                CascLibWrapper.CascExtractFile(hStorage, node.FullPath, targetPath, 0);
            }
            else
            {
                foreach (var child in node.Children.ToList())
                {
                    ExtractNodeRecursive(hStorage, child, Path.Combine(targetPath, child.Name!));
                }
            }
        }
        #endregion
    }
}
