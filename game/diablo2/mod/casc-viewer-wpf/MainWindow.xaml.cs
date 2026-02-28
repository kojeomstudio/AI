using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace CascViewerWPF
{
    public partial class MainWindow : Window
    {
        public ObservableCollection<CascNode> CascNodes { get; set; } = new ObservableCollection<CascNode>();

        public MainWindow()
        {
            InitializeComponent();
            CascTreeView.ItemsSource = CascNodes;
        }

        private void BrowseButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                PathTextBox.Text = dialog.SelectedPath;
            }
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            string path = PathTextBox.Text;
            if (string.IsNullOrEmpty(path) || !Directory.Exists(path))
            {
                System.Windows.MessageBox.Show("Please select a valid D2R directory.");
                return;
            }

            CascNodes.Clear();
            StatusText.Text = "Loading CASC...";

            try
            {
                // Placeholder: Here you would use CascLibWrapper to open and read
                IntPtr hStorage;
                bool success = CascLibWrapper.CascOpenStorage(path, CascLibWrapper.CASC_OPEN_LOCAL, out hStorage);
                
                if (success)
                {
                    StatusText.Text = "CASC Storage Opened Successfully.";
                    // Populate Tree logic would go here
                    AddDummyData();
                    CascLibWrapper.CascCloseStorage(hStorage);
                }
                else
                {
                    StatusText.Text = "Failed to open CASC storage. Ensure CascLib.dll is present.";
                    System.Windows.MessageBox.Show("Failed to open CASC storage.");
                    // Fallback to dummy data for UI demo
                    AddDummyData();
                }
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
                AddDummyData();
            }
        }

        private void AddDummyData()
        {
            var root = new CascNode { Name = "data" };
            var global = new CascNode { Name = "global" };
            var excel = new CascNode { Name = "excel" };
            
            excel.Children.Add(new CascNode { Name = "weapons.txt", Size = "45 KB", Type = "TXT" });
            excel.Children.Add(new CascNode { Name = "armor.txt", Size = "32 KB", Type = "TXT" });
            
            global.Children.Add(excel);
            root.Children.Add(global);
            
            CascNodes.Add(root);
        }
    }

    public class CascNode
    {
        public string? Name { get; set; }
        public string? Size { get; set; }
        public string? Type { get; set; }
        public ObservableCollection<CascNode> Children { get; set; } = new ObservableCollection<CascNode>();
    }
}
