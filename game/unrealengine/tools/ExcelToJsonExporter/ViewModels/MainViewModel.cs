using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using ExcelToJsonExporter.Models;
using ExcelToJsonExporter.Services;
using Microsoft.Win32;

namespace ExcelToJsonExporter.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private readonly ExcelReader _excelReader = new();
        private readonly JsonExporter _jsonExporter = new();

        private string _inputPath = "";
        public string InputPath
        {
            get => _inputPath;
            set => SetProperty(ref _inputPath, value);
        }

        private string _outputPath = "";
        public string OutputPath
        {
            get => _outputPath;
            set => SetProperty(ref _outputPath, value);
        }

        private ObservableCollection<SheetPreview> _sheets = new();
        public ObservableCollection<SheetPreview> Sheets
        {
            get => _sheets;
            set => SetProperty(ref _sheets, value);
        }

        private SheetPreview? _selectedSheet;
        public SheetPreview? SelectedSheet
        {
            get => _selectedSheet;
            set => SetProperty(ref _selectedSheet, value);
        }

        private ObservableCollection<string> _logMessages = new();
        public ObservableCollection<string> LogMessages
        {
            get => _logMessages;
            set => SetProperty(ref _logMessages, value);
        }

        private bool _isExportEnabled;
        public bool IsExportEnabled
        {
            get => _isExportEnabled;
            set => SetProperty(ref _isExportEnabled, value);
        }

        public ICommand BrowseInputCommand { get; }
        public ICommand BrowseOutputCommand { get; }
        public ICommand ExportCommand { get; }

        public MainViewModel()
        {
            BrowseInputCommand = new RelayCommand(_ => BrowseInput());
            BrowseOutputCommand = new RelayCommand(_ => BrowseOutput());
            ExportCommand = new RelayCommand(_ => Export(), _ => IsExportEnabled);
        }

        private void BrowseInput()
        {
            var dialog = new OpenFileDialog
            {
                Title = "엑셀 파일 선택",
                Filter = "Excel Files (*.xlsx)|*.xlsx|All Files (*.*)|*.*",
                Multiselect = false
            };

            if (dialog.ShowDialog() == true)
            {
                InputPath = dialog.FileName;

                if (string.IsNullOrEmpty(OutputPath))
                {
                    OutputPath = Path.GetDirectoryName(dialog.FileName) ?? "";
                }

                LoadPreview();
            }
        }

        private void BrowseOutput()
        {
            var dialog = new OpenFolderDialog
            {
                Title = "출력 디렉토리 선택"
            };

            if (dialog.ShowDialog() == true)
            {
                OutputPath = dialog.FolderName;
            }
        }

        private void LoadPreview()
        {
            LogMessages.Clear();
            Sheets.Clear();
            SelectedSheet = null;
            IsExportEnabled = false;

            if (!File.Exists(InputPath))
            {
                Log("오류: 파일을 찾을 수 없습니다 - " + InputPath);
                return;
            }

            Log("파일 읽는 중: " + Path.GetFileName(InputPath));

            try
            {
                var previews = _excelReader.Read(InputPath);
                foreach (var preview in previews)
                {
                    Sheets.Add(preview);

                    if (preview.IsValid)
                        Log($"  시트 '{preview.SheetName}': {preview.Columns.Count} 컬럼, {preview.TotalRows} 데이터 행");
                    else
                        Log($"  시트 '{preview.SheetName}': 건너뜀 - {preview.ErrorMessage}");
                }

                if (Sheets.Count > 0)
                {
                    SelectedSheet = Sheets.FirstOrDefault(s => s.IsValid);
                    IsExportEnabled = Sheets.Any(s => s.IsValid);
                }

                Log($"총 {Sheets.Count}개 시트 로드 완료");
            }
            catch (Exception ex)
            {
                Log("오류: " + ex.Message);
            }
        }

        private void Export()
        {
            if (string.IsNullOrEmpty(InputPath) || string.IsNullOrEmpty(OutputPath))
                return;

            Log("JSON 변환 시작...");

            try
            {
                var results = _jsonExporter.Export(InputPath, OutputPath);

                int successCount = 0;
                int failCount = 0;

                foreach (var r in results)
                {
                    if (r.Success)
                    {
                        Log($"  완료: {r.SheetName} → {Path.GetFileName(r.OutputPath)} ({r.RowCount} 행)");
                        successCount++;
                    }
                    else
                    {
                        Log($"  실패: {r.SheetName} - {r.ErrorMessage}");
                        failCount++;
                    }
                }

                Log($"변환 완료: 성공 {successCount}개, 실패 {failCount}개");
            }
            catch (Exception ex)
            {
                Log("오류: " + ex.Message);
            }
        }

        private void Log(string message)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss");
            LogMessages.Add($"[{timestamp}] {message}");
        }
    }
}
