using OfficeOpenXml;

namespace ExcelToJsonExporter.Models
{
    public class ColumnDefinition
    {
        public string DataType { get; set; } = "string";
        public string ColumnName { get; set; } = "";
        public string DisplayName { get; set; } = "";

        public static ColumnDefinition Parse(string headerText)
        {
            var def = new ColumnDefinition();

            int colonIndex = headerText.IndexOf(':');
            if (colonIndex > 0)
            {
                def.DataType = headerText.Substring(0, colonIndex).Trim().ToLower();
                def.ColumnName = headerText.Substring(colonIndex + 1).Trim();
            }
            else
            {
                def.DataType = "string";
                def.ColumnName = headerText.Trim();
            }

            def.DisplayName = def.ColumnName;
            return def;
        }
    }
}
