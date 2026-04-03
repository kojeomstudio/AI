using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using Newtonsoft.Json;

namespace DataGenerator
{
    public partial class MainWindow : Window
    {
        private DataGeneratorCore Generator;

        public MainWindow()
        {
            InitializeComponent();
            Generator = new DataGeneratorCore();
            Generator.OnLog = (msg) => {
                this.Dispatcher.Invoke(() => {
                    this.Log(msg);
                });
            };
        }

        private void OnGenerateButtonClick(object sender, RoutedEventArgs e)
        {
            this.LogBox.Items.Clear();
            this.Log("Starting generation process (WPF)...");
            Generator.Run();
        }

        private void Log(string Message)
        {
            this.LogBox.Items.Add("[" + DateTime.Now.ToString("HH:mm:ss") + "] " + Message);
            if (this.LogBox.Items.Count > 0)
            {
                this.LogBox.ScrollIntoView(this.LogBox.Items[this.LogBox.Items.Count - 1]);
            }
        }
    }
}
