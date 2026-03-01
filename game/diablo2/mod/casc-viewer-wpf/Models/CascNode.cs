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
        public ObservableCollection<CascNode> Children { get; } = new ObservableCollection<CascNode>();
        
        // Fast lookup for tree building (not for binding)
        private readonly Dictionary<string, CascNode> _childrenLookup = new Dictionary<string, CascNode>(System.StringComparer.OrdinalIgnoreCase);

        public CascNode? GetOrCreateChild(string name, bool isFile)
        {
            if (_childrenLookup.TryGetValue(name, out var existing))
                return existing;

            var newNode = new CascNode { Name = name, IsFile = isFile };
            _childrenLookup[name] = newNode;
            Children.Add(newNode);
            return newNode;
        }

        // UI Helpers
        public string Icon => IsFile ? "📄" : "📁";
    }
}
