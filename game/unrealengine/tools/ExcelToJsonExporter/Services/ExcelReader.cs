using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using ExcelToJsonExporter.Models;
using OfficeOpenXml;

namespace ExcelToJsonExporter.Services
{
    public class ExcelReader
    {
        public List<SheetPreview> Read(string filePath, int previewRows = 100)
        {
            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;

            var sheets = new List<SheetPreview>();

            using (var package = new ExcelPackage(new FileInfo(filePath)))
            {
                foreach (var worksheet in package.Workbook.Worksheets)
                {
                    var preview = ReadSheet(worksheet, previewRows);
                    sheets.Add(preview);
                }
            }

            return sheets;
        }

        private SheetPreview ReadSheet(ExcelWorksheet worksheet, int previewRows)
        {
            var preview = new SheetPreview
            {
                SheetName = worksheet.Name
            };

            try
            {
                int rowCount = worksheet.Dimension?.Rows ?? 0;
                int colCount = worksheet.Dimension?.Columns ?? 0;

                if (rowCount < 2 || colCount < 1)
                {
                    preview.IsValid = false;
                    preview.ErrorMessage = "시트에 데이터가 없거나 헤더 행만 있습니다.";
                    return preview;
                }

                for (int col = 1; col <= colCount; col++)
                {
                    var cellValue = worksheet.Cells[1, col].Text?.Trim() ?? "";
                    if (string.IsNullOrEmpty(cellValue))
                        continue;

                    var colDef = ColumnDefinition.Parse(cellValue);
                    preview.Columns.Add(colDef);
                }

                if (preview.Columns.Count == 0)
                {
                    preview.IsValid = false;
                    preview.ErrorMessage = "유효한 헤더 컬럼을 찾을 수 없습니다.";
                    return preview;
                }

                int dataStartRow = DetermineDataStartRow(worksheet, colCount);

                for (int row = dataStartRow; row <= rowCount; row++)
                {
                    var values = new List<string>();
                    bool rowIsEmpty = true;

                    for (int col = 1; col <= colCount; col++)
                    {
                        string val = worksheet.Cells[row, col].Text?.Trim() ?? "";
                        values.Add(val);
                        if (!string.IsNullOrEmpty(val))
                            rowIsEmpty = false;
                    }

                    if (rowIsEmpty)
                        continue;

                    if (IsSeparatorRow(values))
                        continue;

                    var rowCollection = new ObservableCollection<string>(values);
                    preview.Rows.Add(rowCollection);

                    preview.TotalRows++;

                    if (preview.TotalRows >= previewRows)
                        break;
                }

                if (preview.TotalRows == 0)
                {
                    preview.IsValid = false;
                    preview.ErrorMessage = "데이터 행을 찾을 수 없습니다.";
                }
            }
            catch (Exception ex)
            {
                preview.IsValid = false;
                preview.ErrorMessage = $"읽기 오류: {ex.Message}";
            }

            return preview;
        }

        private int DetermineDataStartRow(ExcelWorksheet worksheet, int colCount)
        {
            if (worksheet.Dimension == null || worksheet.Dimension.Rows < 2)
                return 2;

            for (int row = 2; row <= Math.Min(5, worksheet.Dimension.Rows); row++)
            {
                bool isSeparator = true;
                for (int col = 1; col <= colCount; col++)
                {
                    string val = worksheet.Cells[row, col].Text?.Trim() ?? "";
                    if (!IsSeparatorCell(val))
                    {
                        isSeparator = false;
                        break;
                    }
                }
                if (!isSeparator)
                    return row;
            }

            return 2;
        }

        private bool IsSeparatorCell(string value)
        {
            if (string.IsNullOrEmpty(value)) return true;
            return value.All(c => c == '-' || c == '=' || c == '_' || c == '~' || c == ':');
        }

        private bool IsSeparatorRow(List<string> values)
        {
            return values.All(v => IsSeparatorCell(v));
        }
    }
}
