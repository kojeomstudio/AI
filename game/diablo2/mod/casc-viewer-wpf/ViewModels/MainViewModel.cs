using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
        private string _previewText = string.Empty;
        private bool _isLoading;
        private CascNode? _selectedNode;
        private int _totalFiles;
        private int _currentFiles;
        private double _progressValue;
        #endregion

        #region Properties
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

        public string PreviewText
        {
            get => _previewText;
            set => SetProperty(ref _previewText, value);
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
                    UpdatePreview();
                }
            }
        }

        private async void UpdatePreview()
        {
            if (SelectedNode == null || !SelectedNode.IsFile || string.IsNullOrEmpty(SelectedNode.FullPath))
            {
                PreviewText = string.Empty;
                return;
            }

            string ext = Path.GetExtension(SelectedNode.FullPath).ToLower();
            if (ext == ".json" || ext == ".txt" || ext == ".text" || ext == ".xml")
            {
                PreviewText = "Loading preview...";
                try
                {
                    await Task.Run(() =>
                    {
                        IntPtr hStorage;
                        if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                        {
                            IntPtr hFile;
                            if (CascLibWrapper.CascOpenFile(hStorage, SelectedNode.FullPath, 0, 0, out hFile))
                            {
                                byte[] buffer = new byte[128 * 1024]; // Max 128KB for preview
                                uint bytesRead;
                                if (CascLibWrapper.CascReadFile(hFile, buffer, (uint)buffer.Length, out bytesRead))
                                {
                                    string content = System.Text.Encoding.UTF8.GetString(buffer, 0, (int)bytesRead);
                                    
                                    if (ext == ".json")
                                    {
                                        try
                                        {
                                            using var doc = System.Text.Json.JsonDocument.Parse(content);
                                            content = System.Text.Json.JsonSerializer.Serialize(doc, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                                        }
                                        catch { /* Not valid JSON, show raw */ }
                                    }
                                    
                                    System.Windows.Application.Current.Dispatcher.Invoke(() => PreviewText = content);
                                }
                                CascLibWrapper.CascCloseFile(hFile);
                            }
                            CascLibWrapper.CascCloseStorage(hStorage);
                        }
                    });
                }
                catch (Exception ex)
                {
                    PreviewText = $"Error loading preview: {ex.Message}";
                }
            }
            else
            {
                PreviewText = $"(Preview not available for {ext} files)";
            }
        }

        public bool CanExtract => SelectedNode != null;
        public ObservableCollection<CascNode> CascNodes { get; } = new ObservableCollection<CascNode>();
        #endregion

        #region Commands
        public ICommand BrowseCommand { get; }
        public ICommand LoadCommand { get; }
        public ICommand ExtractCommand { get; }
        public ICommand CopyPathCommand { get; }
        public ICommand ClearLogsCommand { get; }
        public ICommand OpenLogFolderCommand { get; }
        #endregion

        public MainViewModel()
        {
            BrowseCommand = new RelayCommand(_ => Browse(), _ => CanBrowse);
            LoadCommand = new RelayCommand(_ => LoadCasc(), _ => !string.IsNullOrEmpty(D2RPath) && !IsLoading);
            ExtractCommand = new RelayCommand(_ => ExtractSelected(), _ => CanExtract);
            CopyPathCommand = new RelayCommand(_ => CopyPath(), _ => SelectedNode != null);
            ClearLogsCommand = new RelayCommand(_ => LogService.Instance.Logs.Clear());
            OpenLogFolderCommand = new RelayCommand(_ => OpenLogFolder());
            
            LogService.Instance.Log("MainViewModel initialized.");
        }

        #region Methods
        private void OpenLogFolder()
        {
            try
            {
                string logFile = LogService.Instance.LogFilePath;
                string? folder = Path.GetDirectoryName(logFile);
                if (folder != null && Directory.Exists(folder))
                {
                    System.Diagnostics.Process.Start("explorer.exe", folder);
                }
            }
            catch (Exception ex)
            {
                LogService.Instance.Log($"Failed to open log folder: {ex.Message}", LogLevel.Error);
            }
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
                    try 
                    {
                        if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                        {
                            LogService.Instance.Log("CASC storage opened successfully.");
                            
                            uint lengthNeeded;
                            byte[] buffer = new byte[4];
                            if (CascLibWrapper.CascGetStorageInfo(hStorage, CascLibWrapper.CascStorageTotalFileCount, buffer, 4, out lengthNeeded))
                            {
                                TotalFiles = BitConverter.ToInt32(buffer, 0);
                            }

                            System.Windows.Application.Current.Dispatcher.Invoke(() => LoadingStage = "Phase 2: Mapping Virtual Files...");
                            UpdateStatus("Building file hierarchy...");
                            
                            PopulateTreeOptimized(hStorage);
                            
                            CascLibWrapper.CascCloseStorage(hStorage);
                        }
                        else
                        {
                            UpdateStatus("Failed to open CASC storage.");
                            LogService.Instance.Log($"Failed to open CASC storage. Win32Error: {Marshal.GetLastWin32Error()}", LogLevel.Error);
                        }
                    }
                    catch (EntryPointNotFoundException ex)
                    {
                        LogService.Instance.Log($"DLL Entry Point Not Found: {ex.Message}", LogLevel.Error);
                        UpdateStatus("DLL compatibility error.");
                    }
                    catch (DllNotFoundException ex)
                    {
                        LogService.Instance.Log($"DLL Not Found: {ex.Message}", LogLevel.Error);
                        UpdateStatus("CascLib.dll missing.");
                    }
                });
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}");
                LogService.Instance.Log($"Critical error during CASC load: {ex.Message}", LogLevel.Error);
            }
            finally
            {
                IsLoading = false;
                LoadingStage = "Ready";
            }
        }

        private void UpdateStatus(string message)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() => StatusText = message);
        }

        private void PopulateTreeOptimized(IntPtr hStorage)
        {
            CascLibWrapper.CASC_FIND_DATA findData = new CascLibWrapper.CASC_FIND_DATA();
            IntPtr hFind = IntPtr.Zero;
            
            try 
            {
                hFind = CascLibWrapper.CascFindFirstFile(hStorage, SearchMask, ref findData, null);
                if (hFind != IntPtr.Zero)
                {
                    int processed = 0;
                    int namedFiles = 0;
                    var tempRootNodes = new List<CascNode>();
                    var tempRootLookup = new Dictionary<string, CascNode>(StringComparer.OrdinalIgnoreCase);

                    do
                    {
                        string fileName = findData.szFileName;
                        if (!string.IsNullOrEmpty(fileName))
                        {
                            AddFileToInternalTree(tempRootNodes, tempRootLookup, fileName, findData.FileSize);
                            namedFiles++;
                        }
                        
                        processed++;
                        if (processed % 10000 == 0) UpdateStatusOnUI(processed);
                        if (processed > 10_000_000) break;

                    } while (CascLibWrapper.CascFindNextFile(hFind, ref findData));

                    CascLibWrapper.CascFindClose(hFind);
                    
                    System.Windows.Application.Current.Dispatcher.Invoke(() => 
                    {
                        foreach (var node in tempRootNodes) CascNodes.Add(node);
                        CurrentFiles = processed;
                        ProgressValue = 100;
                        StatusText = $"Load Complete. {processed} entries mapped.";
                        LogService.Instance.Log($"Mapping complete. Processed {processed} entries, found {namedFiles} named files.");
                    });
                }
                else
                {
                    LogService.Instance.Log($"No files found matching mask: {SearchMask}", LogLevel.Warning);
                }
            }
            catch (Exception ex)
            {
                LogService.Instance.Log($"Error during file scanning: {ex.Message}", LogLevel.Error);
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

        private void AddFileToInternalTree(List<CascNode> rootList, Dictionary<string, CascNode> rootLookup, string filePath, ulong fileSize)
        {
            string[] parts = filePath.Split(new char[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) return;

            CascNode currentNode;
            string firstPart = parts[0];

            if (!rootLookup.TryGetValue(firstPart, out currentNode!))
            {
                currentNode = new CascNode { Name = firstPart, IsFile = (parts.Length == 1) };
                rootLookup[firstPart] = currentNode;
                rootList.Add(currentNode);
            }

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

        private async void ExtractSelected()
        {
            if (SelectedNode == null) return;
            if (SelectedNode.IsFile) ExtractSingleFile(SelectedNode);
            else await ExtractFolder(SelectedNode);
        }

        private void CopyPath()
        {
            if (SelectedNode?.FullPath != null)
            {
                System.Windows.Clipboard.SetText(SelectedNode.FullPath);
                LogService.Instance.Log($"Path copied to clipboard: {SelectedNode.FullPath}");
            }
        }

        private void ExtractSingleFile(CascNode node)
        {
            if (string.IsNullOrEmpty(node.FullPath)) return;

            var saveDialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = node.Name,
                Filter = "All Files (*.*)|*.*",
                Title = $"Extract {node.Name}"
            };

            if (saveDialog.ShowDialog() == true)
            {
                LogService.Instance.Log($"Attempting to extract file: {node.FullPath}");
                try 
                {
                    IntPtr hStorage;
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        bool success = ExtractFileInternal(hStorage, node.FullPath, saveDialog.FileName);
                        CascLibWrapper.CascCloseStorage(hStorage);
                        
                        if (success)
                        {
                            LogService.Instance.Log($"Extraction successful: {saveDialog.FileName}");
                            System.Windows.MessageBox.Show("File extracted successfully.");
                        }
                        else
                        {
                            LogService.Instance.Log($"Extraction failed for: {node.FullPath}", LogLevel.Error);
                            System.Windows.MessageBox.Show("Extraction failed. Check logs.");
                        }
                    }
                }
                catch (Exception ex)
                {
                    LogService.Instance.Log($"Unexpected error during extraction: {ex.Message}", LogLevel.Error);
                }
            }
        }

        /// <summary>
        /// Manually extracts a file by opening, reading and writing it to disk.
        /// CascLib.dll doesn't always export a single 'Extract' function.
        /// </summary>
        private bool ExtractFileInternal(IntPtr hStorage, string fileName, string targetPath)
        {
            IntPtr hFile = IntPtr.Zero;
            try
            {
                // LocaleFlags = 0, OpenFlags = 0
                if (CascLibWrapper.CascOpenFile(hStorage, fileName, 0, 0, out hFile))
                {
                    // Ensure directory exists
                    string? dir = Path.GetDirectoryName(targetPath);
                    if (dir != null) Directory.CreateDirectory(dir);

                    using (var fs = new FileStream(targetPath, FileMode.Create, FileAccess.Write))
                    {
                        byte[] buffer = new byte[64 * 1024]; // 64KB buffer
                        uint bytesRead;
                        
                        // Check if it's a JSON file to format it
                        string ext = Path.GetExtension(fileName).ToLower();
                        if (ext == ".json")
                        {
                            var ms = new MemoryStream();
                            while (CascLibWrapper.CascReadFile(hFile, buffer, (uint)buffer.Length, out bytesRead) && bytesRead > 0)
                            {
                                ms.Write(buffer, 0, (int)bytesRead);
                            }
                            
                            string rawJson = System.Text.Encoding.UTF8.GetString(ms.ToArray());
                            try
                            {
                                using var doc = System.Text.Json.JsonDocument.Parse(rawJson);
                                string formattedJson = System.Text.Json.JsonSerializer.Serialize(doc, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                                byte[] formattedBytes = System.Text.Encoding.UTF8.GetBytes(formattedJson);
                                fs.Write(formattedBytes, 0, formattedBytes.Length);
                            }
                            catch
                            {
                                // If not valid JSON, write raw
                                byte[] rawBytes = ms.ToArray();
                                fs.Write(rawBytes, 0, rawBytes.Length);
                            }
                        }
                        else
                        {
                            // Standard streaming for other files
                            while (CascLibWrapper.CascReadFile(hFile, buffer, (uint)buffer.Length, out bytesRead) && bytesRead > 0)
                            {
                                fs.Write(buffer, 0, (int)bytesRead);
                            }
                        }
                    }
                    return true;
                }
                else
                {
                    LogService.Instance.Log($"CascOpenFile failed for {fileName}. Win32Error: {Marshal.GetLastWin32Error()}", LogLevel.Error);
                    return false;
                }
            }
            catch (Exception ex)
            {
                LogService.Instance.Log($"Error in ExtractFileInternal for {fileName}: {ex.Message}", LogLevel.Error);
                return false;
            }
            finally
            {
                if (hFile != IntPtr.Zero) CascLibWrapper.CascCloseFile(hFile);
            }
        }

        private async Task ExtractFolder(CascNode folderNode)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                string targetBase = Path.Combine(dialog.SelectedPath, folderNode.Name!);
                IsLoading = true;
                LogService.Instance.Log($"Starting recursive extraction: {folderNode.Name}");
                
                await Task.Run(() => 
                {
                    IntPtr hStorage;
                    try 
                    {
                        if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                        {
                            ExtractNodeRecursive(hStorage, folderNode, targetBase);
                            CascLibWrapper.CascCloseStorage(hStorage);
                            LogService.Instance.Log($"Folder extraction complete: {folderNode.Name}");
                        }
                    }
                    catch (Exception ex)
                    {
                        LogService.Instance.Log($"Error during folder extraction: {ex.Message}", LogLevel.Error);
                    }
                });
                IsLoading = false;
                System.Windows.MessageBox.Show("Extraction task finished.");
            }
        }

        private void ExtractNodeRecursive(IntPtr hStorage, CascNode node, string targetPath)
        {
            if (node.IsFile)
            {
                if (string.IsNullOrEmpty(node.FullPath)) return;
                ExtractFileInternal(hStorage, node.FullPath, targetPath);
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
