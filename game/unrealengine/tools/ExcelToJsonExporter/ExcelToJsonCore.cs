using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ExcelToJsonExporter.Models;
using ExcelToJsonExporter.Services;
using Newtonsoft.Json;
using OfficeOpenXml;

namespace ExcelToJsonExporter
{
    public class ExcelToJsonCore
    {
        public Action<string>? OnLog;

        private void Log(string message)
        {
            OnLog?.Invoke(message);
        }

        public void Run(string inputPath, string outputDir)
        {
            Log("Excel to JSON 변환을 시작합니다...");

            if (!File.Exists(inputPath))
            {
                Log("오류: 파일을 찾을 수 없습니다 - " + inputPath);
                return;
            }

            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;

            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            using (var package = new ExcelPackage(new FileInfo(inputPath)))
            {
                int totalSheets = package.Workbook.Worksheets.Count;
                Log($"파일: {Path.GetFileName(inputPath)} ({totalSheets}개 시트)");

                int successCount = 0;
                int failCount = 0;

                foreach (var worksheet in package.Workbook.Worksheets)
                {
                    try
                    {
                        var result = ProcessSheet(worksheet, outputDir);
                        if (result.Success)
                        {
                            Log($"  완료: {result.SheetName} → {Path.GetFileName(result.OutputPath)} ({result.RowCount} 행)");
                            successCount++;
                        }
                        else
                        {
                            Log($"  건너뜀: {result.SheetName} - {result.ErrorMessage}");
                            failCount++;
                        }
                    }
                    catch (Exception ex)
                    {
                        Log($"  오류: {worksheet.Name} - {ex.Message}");
                        failCount++;
                    }
                }

                Log($"변환 완료: 성공 {successCount}개, 실패 {failCount}개");
            }
        }

        private ExportResult ProcessSheet(ExcelWorksheet worksheet, string outputDir)
        {
            var result = new ExportResult { SheetName = worksheet.Name };

            int rowCount = worksheet.Dimension?.Rows ?? 0;
            int colCount = worksheet.Dimension?.Columns ?? 0;

            if (rowCount < 2 || colCount < 1)
            {
                result.Success = false;
                result.ErrorMessage = "데이터가 부족합니다 (최소 헤더 + 데이터 1행 필요)";
                return result;
            }

            var columns = new List<ColumnDefinition>();
            for (int col = 1; col <= colCount; col++)
            {
                string cellValue = worksheet.Cells[1, col].Text?.Trim() ?? "";
                if (string.IsNullOrEmpty(cellValue)) continue;
                columns.Add(ColumnDefinition.Parse(cellValue));
            }

            if (columns.Count == 0)
            {
                result.Success = false;
                result.ErrorMessage = "유효한 헤더를 찾을 수 없습니다";
                return result;
            }

            int dataStartRow = FindDataStartRow(worksheet, colCount);
            var entries = new List<Dictionary<string, object>>();

            for (int row = dataStartRow; row <= rowCount; row++)
            {
                var values = new List<string>();
                bool isEmpty = true;

                for (int col = 1; col <= colCount; col++)
                {
                    string val = worksheet.Cells[row, col].Text?.Trim() ?? "";
                    values.Add(val);
                    if (!string.IsNullOrEmpty(val)) isEmpty = false;
                }

                if (isEmpty || IsSeparatorRow(values)) continue;

                var entry = new Dictionary<string, object>();
                for (int i = 0; i < columns.Count; i++)
                {
                    string raw = i < values.Count ? values[i] : "";
                    entry[columns[i].ColumnName] = ParseValue(columns[i].DataType, raw);
                }
                entries.Add(entry);
            }

            string json = JsonConvert.SerializeObject(entries, Formatting.Indented);
            string fileName = SanitizeFileName(worksheet.Name);
            string outputPath = Path.Combine(outputDir, fileName + ".json");
            File.WriteAllText(outputPath, json);

            result.OutputPath = outputPath;
            result.RowCount = entries.Count;
            result.Success = true;
            return result;
        }

        private int FindDataStartRow(ExcelWorksheet ws, int colCount)
        {
            for (int row = 2; row <= Math.Min(5, ws.Dimension?.Rows ?? 2); row++)
            {
                bool isSep = true;
                for (int col = 1; col <= colCount; col++)
                {
                    string v = ws.Cells[row, col].Text?.Trim() ?? "";
                    if (v.Length > 0 && !v.All(c => c == '-' || c == '=' || c == '_' || c == '~' || c == ':'))
                    {
                        isSep = false;
                        break;
                    }
                }
                if (!isSep) return row;
            }
            return 2;
        }

        private bool IsSeparatorRow(List<string> values)
        {
            return values.All(v =>
                string.IsNullOrEmpty(v) ||
                v.All(c => c == '-' || c == '=' || c == '_' || c == '~' || c == ':'));
        }

        private object ParseValue(string dataType, string rawValue)
        {
            if (string.IsNullOrEmpty(rawValue)) return "";
            try
            {
                switch (dataType.ToLower())
                {
                    case "int": case "int32":
                        return int.TryParse(rawValue, out int iv) ? iv : rawValue;
                    case "float": case "single":
                        return float.TryParse(rawValue, out float fv) ? fv : rawValue;
                    case "double":
                        return double.TryParse(rawValue, out double dv) ? dv : rawValue;
                    case "bool": case "boolean":
                        if (bool.TryParse(rawValue, out bool bv)) return bv;
                        if (rawValue == "1") return true;
                        if (rawValue == "0") return false;
                        return rawValue;
                    case "long": case "int64":
                        return long.TryParse(rawValue, out long lv) ? lv : rawValue;
                    default:
                        return rawValue;
                }
            }
            catch { return rawValue; }
        }

        private string SanitizeFileName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries));
        }
    }
}
