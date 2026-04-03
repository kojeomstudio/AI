using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ExcelToJsonExporter.Models;
using Newtonsoft.Json;
using OfficeOpenXml;

namespace ExcelToJsonExporter.Services
{
    public class JsonExporter
    {
        public List<ExportResult> Export(string excelPath, string outputDir)
        {
            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;

            var results = new List<ExportResult>();

            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            using (var package = new ExcelPackage(new FileInfo(excelPath)))
            {
                foreach (var worksheet in package.Workbook.Worksheets)
                {
                    try
                    {
                        var result = ExportSheet(worksheet, outputDir);
                        results.Add(result);
                    }
                    catch (Exception ex)
                    {
                        results.Add(new ExportResult
                        {
                            SheetName = worksheet.Name,
                            Success = false,
                            ErrorMessage = ex.Message
                        });
                    }
                }
            }

            return results;
        }

        private ExportResult ExportSheet(ExcelWorksheet worksheet, string outputDir)
        {
            var result = new ExportResult
            {
                SheetName = worksheet.Name
            };

            int rowCount = worksheet.Dimension?.Rows ?? 0;
            int colCount = worksheet.Dimension?.Columns ?? 0;

            if (rowCount < 2 || colCount < 1)
            {
                result.Success = false;
                result.ErrorMessage = "데이터가 없습니다.";
                return result;
            }

            var columns = new List<ColumnDefinition>();
            for (int col = 1; col <= colCount; col++)
            {
                string cellValue = worksheet.Cells[1, col].Text?.Trim() ?? "";
                if (string.IsNullOrEmpty(cellValue))
                    continue;
                columns.Add(ColumnDefinition.Parse(cellValue));
            }

            if (columns.Count == 0)
            {
                result.Success = false;
                result.ErrorMessage = "유효한 헤더를 찾을 수 없습니다.";
                return result;
            }

            var entries = new List<Dictionary<string, object>>();
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

                if (rowIsEmpty || IsSeparatorRow(values))
                    continue;

                var entry = new Dictionary<string, object>();
                for (int i = 0; i < columns.Count; i++)
                {
                    string rawValue = i < values.Count ? values[i] : "";
                    entry[columns[i].ColumnName] = ParseValue(columns[i].DataType, rawValue);
                }
                entries.Add(entry);
            }

            string jsonOutput = JsonConvert.SerializeObject(entries, Formatting.Indented);
            string fileName = SanitizeFileName(worksheet.Name);
            string outputPath = Path.Combine(outputDir, fileName + ".json");

            File.WriteAllText(outputPath, jsonOutput);

            result.OutputPath = outputPath;
            result.RowCount = entries.Count;
            result.Success = true;

            return result;
        }

        private object ParseValue(string dataType, string rawValue)
        {
            if (string.IsNullOrEmpty(rawValue))
                return "";

            try
            {
                switch (dataType.ToLower())
                {
                    case "int":
                    case "int32":
                        if (int.TryParse(rawValue, out int intVal)) return intVal;
                        return rawValue;

                    case "float":
                    case "single":
                        if (float.TryParse(rawValue, out float floatVal)) return floatVal;
                        return rawValue;

                    case "double":
                        if (double.TryParse(rawValue, out double doubleVal)) return doubleVal;
                        return rawValue;

                    case "bool":
                    case "boolean":
                        if (bool.TryParse(rawValue, out bool boolVal)) return boolVal;
                        if (rawValue.Equals("1", StringComparison.Ordinal)) return true;
                        if (rawValue.Equals("0", StringComparison.Ordinal)) return false;
                        return rawValue;

                    case "long":
                    case "int64":
                        if (long.TryParse(rawValue, out long longVal)) return longVal;
                        return rawValue;

                    default:
                        return rawValue;
                }
            }
            catch
            {
                return rawValue;
            }
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

        private string SanitizeFileName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries));
        }
    }
}
