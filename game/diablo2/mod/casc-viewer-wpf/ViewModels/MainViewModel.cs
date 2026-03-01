using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using CascViewerWPF.Commands;
using CascViewerWPF.Models;

namespace CascViewerWPF.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private string _d2rPath = string.Empty;
        private string _statusText = "Ready";
        private bool _isLoading;
        private CascNode? _selectedNode;

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
                }
            }
        }

        public bool IsBusy => IsLoading;

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

        public bool CanExtract => SelectedNode?.IsFile ?? false;

        public ObservableCollection<CascNode> CascNodes { get; } = new ObservableCollection<CascNode>();

        public ICommand BrowseCommand { get; }
        public ICommand LoadCommand { get; }
        public ICommand ExtractCommand { get; }

        public MainViewModel()
        {
            BrowseCommand = new RelayCommand(_ => Browse());
            LoadCommand = new RelayCommand(_ => LoadCasc(), _ => !string.IsNullOrEmpty(D2RPath) && !IsLoading);
            ExtractCommand = new RelayCommand(_ => ExtractSelected(), _ => CanExtract);
        }

        private void Browse()
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                D2RPath = dialog.SelectedPath;
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
            CascNodes.Clear();

            try
            {
                await Task.Run(() =>
                {
                    IntPtr hStorage;
                    if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                    {
                        UpdateStatus("Scanning files...");
                        PopulateTree(hStorage);
                        CascLibWrapper.CascCloseStorage(hStorage);
                        UpdateStatus("CASC Loaded Successfully.");
                    }
                    else
                    {
                        UpdateStatus("Failed to open CASC storage.");
                    }
                });
            }
            catch (Exception ex)
            {
                StatusText = $"Error: {ex.Message}";
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
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
                do
                {
                    if (!string.IsNullOrEmpty(findData.szFileName))
                    {
                        var fileName = findData.szFileName;
                        var fileSize = findData.dwFileSize;
                        System.Windows.Application.Current.Dispatcher.Invoke(() => AddFileToTree(fileName, fileSize));
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

        private void ExtractSelected()
        {
            if (SelectedNode == null || !SelectedNode.IsFile || string.IsNullOrEmpty(SelectedNode.FullPath))
                return;

            var saveDialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = SelectedNode.Name,
                Filter = $"All Files (*.*)|*.*"
            };

            if (saveDialog.ShowDialog() == true)
            {
                StatusText = $"Extracting {SelectedNode.Name}...";
                
                IntPtr hStorage;
                if (CascLibWrapper.CascOpenStorage(D2RPath, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage))
                {
                    bool success = CascLibWrapper.CascExtractFile(hStorage, SelectedNode.FullPath!, saveDialog.FileName, 0);
                    CascLibWrapper.CascCloseStorage(hStorage);

                    if (success)
                    {
                        StatusText = "Extraction Complete.";
                        System.Windows.MessageBox.Show("File extracted successfully.");
                    }
                    else
                    {
                        StatusText = "Extraction Failed.";
                        System.Windows.MessageBox.Show("Failed to extract file.");
                    }
                }
            }
        }
    }
}
