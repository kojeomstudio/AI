using System.Windows;
using CascViewerWPF.ViewModels;
using CascViewerWPF.Models;

namespace CascViewerWPF
{
    public partial class MainWindow : Window
    {
        public MainViewModel ViewModel { get; }

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new MainViewModel();
            DataContext = ViewModel;
        }

        private void CascTreeView_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            ViewModel.SelectedNode = e.NewValue as CascNode;
        }
    }
}
