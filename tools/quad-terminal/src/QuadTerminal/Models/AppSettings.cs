using System;

namespace QuadTerminal.Models;

public class AppSettings
{
    public PaneSettings[] Panes { get; set; } = CreateDefaultPanes();
    public string Layout { get; set; } = "Quad";
    public WindowBoundsSettings? WindowBounds { get; set; }

    private static PaneSettings[] CreateDefaultPanes()
    {
        string dir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return
        [
            new PaneSettings { WorkingDirectory = dir, Shell = "powershell.exe", Enabled = true },
            new PaneSettings { WorkingDirectory = dir, Shell = "powershell.exe", Enabled = true },
            new PaneSettings { WorkingDirectory = dir, Shell = "powershell.exe", Enabled = true },
            new PaneSettings { WorkingDirectory = dir, Shell = "powershell.exe", Enabled = true }
        ];
    }
}

public class PaneSettings
{
    public string WorkingDirectory { get; set; } = "";
    public string Shell { get; set; } = "powershell.exe";
    public bool Enabled { get; set; } = true;
}

public class WindowBoundsSettings
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Width { get; set; }
    public double Height { get; set; }
    public string WindowState { get; set; } = "Normal";
}
