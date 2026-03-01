using System.Collections.ObjectModel;
using System.Collections.Generic;

namespace CascViewerWPF.Models
{
    /// <summary>
    /// Represents a single node in the hierarchical file tree (either a folder or a file).
    /// </summary>
    public class CascNode
    {
        /// <summary>
        /// Display name of the node (file name or folder name).
        /// </summary>
        public string? Name { get; set; }

        /// <summary>
        /// Human-readable size string (e.g., "1.24 MB").
        /// </summary>
        public string? Size { get; set; }

        /// <summary>
        /// File extension/type (e.g., "DCC", "TXT").
        /// </summary>
        public string? Type { get; set; }

        /// <summary>
        /// The full virtual path within the CASC storage.
        /// </summary>
        public string? FullPath { get; set; }

        /// <summary>
        /// True if this node represents a file; false if it's a folder.
        /// </summary>
        public bool IsFile { get; set; }

        /// <summary>
        /// True if this node represents a folder.
        /// </summary>
        public bool IsFolder => !IsFile;

        /// <summary>
        /// Collection of child nodes (sub-folders and files).
        /// </summary>
        public ObservableCollection<CascNode> Children { get; } = new ObservableCollection<CascNode>();
        
        /// <summary>
        /// Fast lookup dictionary to manage child nodes during tree construction.
        /// This avoids O(N) searches when building large hierarchies.
        /// </summary>
        private readonly Dictionary<string, CascNode> _childrenLookup = new Dictionary<string, CascNode>(System.StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Retrieves an existing child node by name or creates a new one if it doesn't exist.
        /// </summary>
        /// <param name="name">Name of the child node.</param>
        /// <param name="isFile">Whether the child is a file or folder.</param>
        /// <returns>The found or created CascNode.</returns>
        public CascNode? GetOrCreateChild(string name, bool isFile)
        {
            if (_childrenLookup.TryGetValue(name, out var existing))
                return existing;

            var newNode = new CascNode { Name = name, IsFile = isFile };
            _childrenLookup[name] = newNode;
            Children.Add(newNode);
            return newNode;
        }

        /// <summary>
        /// UI helper to provide an icon based on the node type.
        /// </summary>
        public string Icon => IsFile ? "📄" : "📁";
    }
}
