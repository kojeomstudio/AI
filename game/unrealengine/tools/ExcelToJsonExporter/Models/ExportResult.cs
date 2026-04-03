namespace ExcelToJsonExporter.Models
{
    public class ExportResult
    {
        public string SheetName { get; set; } = "";
        public string OutputPath { get; set; } = "";
        public int RowCount { get; set; }
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
