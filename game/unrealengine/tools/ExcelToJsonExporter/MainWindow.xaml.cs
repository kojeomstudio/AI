using System;
using System.Collections.Generic;
using System.Data;
using System.Windows;
using System.Windows.Controls;
using ExcelToJsonExporter.ViewModels;

namespace ExcelToJsonExporter
{
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel;
        private readonly Dictionary<string, string> _columnHeaders = new();

        public MainWindow()
        {
            InitializeComponent();
            _viewModel = new MainViewModel();
            DataContext = _viewModel;

            _viewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(MainViewModel.SelectedSheet))
                {
                    UpdateDataGridColumns();
                }
            };

            PreviewDataGrid.AutoGeneratingColumn += PreviewDataGrid_AutoGeneratingColumn;
        }

        private void PreviewDataGrid_AutoGeneratingColumn(object? sender, DataGridAutoGeneratingColumnEventArgs e)
        {
            if (_viewModel.SelectedSheet == null) return;

            if (_columnHeaders.TryGetValue(e.PropertyName, out string? header))
            {
                e.Column.Header = header;
            }
        }

        private void UpdateDataGridColumns()
        {
            _columnHeaders.Clear();
            var sheet = _viewModel.SelectedSheet;
            if (sheet == null || sheet.Rows.Count == 0) return;

            var dataTable = new DataTable();

            for (int i = 0; i < sheet.Columns.Count; i++)
            {
                string colName = sheet.Columns[i].ColumnName;
                string header = $"{colName} ({sheet.Columns[i].DataType})";
                dataTable.Columns.Add(colName, typeof(string));
                _columnHeaders[colName] = header;
            }

            foreach (var row in sheet.Rows)
            {
                var dataRow = dataTable.NewRow();
                for (int i = 0; i < Math.Min(row.Count, dataTable.Columns.Count); i++)
                {
                    dataRow[i] = row[i];
                }
                dataTable.Rows.Add(dataRow);
            }

            PreviewDataGrid.ItemsSource = dataTable.DefaultView;
        }
    }
}
