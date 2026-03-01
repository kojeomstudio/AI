using System.Collections.ObjectModel;

namespace CascViewerWPF.Models
{
    public class CascNode
    {
        public string? Name { get; set; }
        public string? Size { get; set; }
        public string? Type { get; set; }
        public string? FullPath { get; set; }
        public bool IsFile { get; set; }
        public bool IsFolder => !IsFile;
        public ObservableCollection<CascNode> Children { get; set; } = new ObservableCollection<CascNode>();
        
        // UI Helpers
        public string Icon => IsFile ? "📄" : "📁";
    }
}
