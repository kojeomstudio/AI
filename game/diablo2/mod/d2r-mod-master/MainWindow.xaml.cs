using System.Windows;
using D2RModMaster.ViewModels;
using D2RModMaster.Models;

namespace D2RModMaster
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
