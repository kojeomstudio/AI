using System.Collections.ObjectModel;

namespace ExcelToJsonExporter.Models
{
    public class SheetPreview
    {
        public string SheetName { get; set; } = "";
        public ObservableCollection<ColumnDefinition> Columns { get; set; } = new();
        public ObservableCollection<ObservableCollection<string>> Rows { get; set; } = new();
        public int TotalRows { get; set; }
        public bool IsValid { get; set; } = true;
        public string? ErrorMessage { get; set; }
    }
}
