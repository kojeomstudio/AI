using System.Diagnostics;
using System.Runtime.InteropServices;
using MabinogiMacro.Models;
using MabinogiMacro.Native;
using Serilog;

namespace MabinogiMacro.Services;

public class InputManager
{
    private static readonly Dictionary<string, ushort> KeyMap = new()
    {
        ["space"] = Win32.VK_SPACE,
        ["enter"] = Win32.VK_RETURN,
        ["esc"] = Win32.VK_ESCAPE,
        ["escape"] = Win32.VK_ESCAPE,
        ["left"] = Win32.VK_LEFT,
        ["up"] = Win32.VK_UP,
        ["right"] = Win32.VK_RIGHT,
        ["down"] = Win32.VK_DOWN,
        ["tab"] = Win32.VK_TAB,
        ["shift"] = Win32.VK_SHIFT,
        ["ctrl"] = Win32.VK_CONTROL,
    };

    public IntPtr Hwnd { get; private set; }
    public uint ProcessId { get; private set; }
    public string WindowTitle { get; }

    public InputManager(string windowTitle)
    {
        WindowTitle = windowTitle;
        FindTargetWindow();
    }

    public bool FindTargetWindow()
    {
        Hwnd = Win32.FindWindow(null, WindowTitle);
        if (Hwnd == IntPtr.Zero)
        {
            Log.Warning("Window not found: {WindowTitle}", WindowTitle);
            ProcessId = 0;
            return false;
        }

        Win32.GetWindowThreadProcessId(Hwnd, out uint pid);
        ProcessId = pid;
        Log.Information("Window found: {Title} (HWND={Hwnd}, PID={PID})", WindowTitle, Hwnd, ProcessId);
        return true;
    }

    public bool MonitorProcess()
    {
        if (Hwnd == IntPtr.Zero || !Win32.IsWindow(Hwnd))
        {
            Log.Warning("Window handle invalid, re-searching...");
            return FindTargetWindow();
        }

        try
        {
            var proc = Process.GetProcessById((int)ProcessId);
            if (proc.HasExited)
            {
                Log.Warning("Process {PID} has exited, re-searching...", ProcessId);
                return FindTargetWindow();
            }
        }
        catch (ArgumentException)
        {
            Log.Warning("Process {PID} not found, re-searching...", ProcessId);
            return FindTargetWindow();
        }

        return true;
    }

    public WindowInfo? GetWindowInfo()
    {
        if (Hwnd == IntPtr.Zero) return null;

        if (!Win32.GetWindowRect(Hwnd, out var windowRect)) return null;
        Win32.GetClientRect(Hwnd, out var clientRect);

        return new WindowInfo
        {
            Hwnd = Hwnd,
            Left = windowRect.Left,
            Top = windowRect.Top,
            Right = windowRect.Right,
            Bottom = windowRect.Bottom,
            Width = windowRect.Right - windowRect.Left,
            Height = windowRect.Bottom - windowRect.Top,
            ClientWidth = clientRect.Right,
            ClientHeight = clientRect.Bottom,
        };
    }

    public bool Click(int x, int y, InputMethod method = InputMethod.PostMessage, string button = "left")
    {
        return method switch
        {
            InputMethod.SendInput => ClickSendInput(x, y, button),
            InputMethod.PostMessage => ClickPostMessage(x, y, button),
            InputMethod.SendMessage => ClickSendMessage(x, y, button),
            _ => false,
        };
    }

    public bool SendKey(string key, InputMethod method = InputMethod.PostMessage)
    {
        var vk = GetVkCode(key);
        if (vk == 0) return false;

        return method switch
        {
            InputMethod.SendInput => KeySendInput(vk),
            InputMethod.PostMessage => KeyPostMessage(vk),
            InputMethod.SendMessage => KeySendMessage(vk),
            _ => false,
        };
    }

    private bool ClickSendInput(int x, int y, string button)
    {
        Win32.SetForegroundWindow(Hwnd);
        Thread.Sleep(100);

        Win32.SetCursorPos(x, y);
        Thread.Sleep(50);

        int screenW = NativeMethods.GetSystemMetrics(0);
        int screenH = NativeMethods.GetSystemMetrics(1);
        if (screenW == 0) screenW = 1920;
        if (screenH == 0) screenH = 1080;

        var inputs = new Win32.INPUT[3];
        inputs[0].type = Win32.INPUT_MOUSE;
        inputs[0].u.mi.dwFlags = Win32.MOUSEEVENTF_ABSOLUTE | Win32.MOUSEEVENTF_MOVE;
        inputs[0].u.mi.dx = (int)(x * 65535 / screenW);
        inputs[0].u.mi.dy = (int)(y * 65535 / screenH);
        inputs[0].u.mi.time = 0;
        inputs[0].u.mi.dwExtraInfo = IntPtr.Zero;

        uint downFlag = button == "right" ? (uint)Win32.MOUSEEVENTF_RIGHTDOWN : (uint)Win32.MOUSEEVENTF_LEFTDOWN;
        uint upFlag = button == "right" ? (uint)Win32.MOUSEEVENTF_RIGHTUP : (uint)Win32.MOUSEEVENTF_LEFTUP;

        inputs[1].type = Win32.INPUT_MOUSE;
        inputs[1].u.mi.dwFlags = downFlag;
        inputs[1].u.mi.time = 0;
        inputs[1].u.mi.dwExtraInfo = IntPtr.Zero;

        inputs[2].type = Win32.INPUT_MOUSE;
        inputs[2].u.mi.dwFlags = upFlag;
        inputs[2].u.mi.time = 0;
        inputs[2].u.mi.dwExtraInfo = IntPtr.Zero;

        var result = Win32.SendInput(3, inputs, Marshal.SizeOf<Win32.INPUT>());
        Log.Debug("SendInput click at ({X},{Y}) {Button}", x, y, button);
        return result == 3;
    }

    private bool ClickPostMessage(int x, int y, string button)
    {
        var info = GetWindowInfo();
        if (info == null) return false;

        int clientX = x - info.Left;
        int clientY = y - info.Top;
        var lParam = Win32.MakeLParam(clientX, clientY);

        uint downMsg = button == "right" ? (uint)Win32.WM_RBUTTONDOWN : (uint)Win32.WM_LBUTTONDOWN;
        uint upMsg = button == "right" ? (uint)Win32.WM_RBUTTONUP : (uint)Win32.WM_LBUTTONUP;
        var wParam = (IntPtr)(button == "right" ? Win32.MK_RBUTTON : Win32.MK_LBUTTON);

        Win32.PostMessage(Hwnd, downMsg, wParam, lParam);
        Thread.Sleep(50);
        Win32.PostMessage(Hwnd, upMsg, IntPtr.Zero, lParam);

        Log.Debug("PostMessage click at ({X},{Y}) client({CX},{CY}) {Button}", x, y, clientX, clientY, button);
        return true;
    }

    private bool ClickSendMessage(int x, int y, string button)
    {
        var info = GetWindowInfo();
        if (info == null) return false;

        int clientX = x - info.Left;
        int clientY = y - info.Top;
        var lParam = Win32.MakeLParam(clientX, clientY);

        uint downMsg = button == "right" ? (uint)Win32.WM_RBUTTONDOWN : (uint)Win32.WM_LBUTTONDOWN;
        uint upMsg = button == "right" ? (uint)Win32.WM_RBUTTONUP : (uint)Win32.WM_LBUTTONUP;
        var wParam = (IntPtr)(button == "right" ? Win32.MK_RBUTTON : Win32.MK_LBUTTON);

        Win32.SendMessage(Hwnd, downMsg, wParam, lParam);
        Thread.Sleep(50);
        Win32.SendMessage(Hwnd, upMsg, IntPtr.Zero, lParam);

        Log.Debug("SendMessage click at ({X},{Y}) client({CX},{CY}) {Button}", x, y, clientX, clientY, button);
        return true;
    }

    private bool KeySendInput(ushort vk)
    {
        Win32.SetForegroundWindow(Hwnd);
        Thread.Sleep(100);

        var inputs = new Win32.INPUT[2];
        inputs[0].type = Win32.INPUT_KEYBOARD;
        inputs[0].u.ki.wVk = vk;
        inputs[0].u.ki.dwFlags = 0;
        inputs[0].u.ki.time = 0;
        inputs[0].u.ki.dwExtraInfo = IntPtr.Zero;

        inputs[1].type = Win32.INPUT_KEYBOARD;
        inputs[1].u.ki.wVk = vk;
        inputs[1].u.ki.dwFlags = 0x0002;
        inputs[1].u.ki.time = 0;
        inputs[1].u.ki.dwExtraInfo = IntPtr.Zero;

        var result = Win32.SendInput(2, inputs, Marshal.SizeOf<Win32.INPUT>());
        Log.Debug("SendInput key: VK={VK}", vk);
        return result == 2;
    }

    private bool KeyPostMessage(ushort vk)
    {
        Win32.PostMessage(Hwnd, Win32.WM_KEYDOWN, (IntPtr)vk, IntPtr.Zero);
        Thread.Sleep(50);
        Win32.PostMessage(Hwnd, Win32.WM_KEYUP, (IntPtr)vk, IntPtr.Zero);
        Log.Debug("PostMessage key: VK={VK}", vk);
        return true;
    }

    private bool KeySendMessage(ushort vk)
    {
        Win32.SendMessage(Hwnd, Win32.WM_KEYDOWN, (IntPtr)vk, IntPtr.Zero);
        Thread.Sleep(50);
        Win32.SendMessage(Hwnd, Win32.WM_KEYUP, (IntPtr)vk, IntPtr.Zero);
        Log.Debug("SendMessage key: VK={VK}", vk);
        return true;
    }

    private static ushort GetVkCode(string key)
    {
        if (string.IsNullOrEmpty(key)) return 0;
        key = key.ToLowerInvariant();
        if (KeyMap.TryGetValue(key, out var vk)) return vk;
        if (key.Length == 1 && char.IsLetterOrDigit(key[0])) return (ushort)char.ToUpper(key[0]);
        return 0;
    }
}

public record WindowInfo
{
    public IntPtr Hwnd { get; init; }
    public int Left { get; init; }
    public int Top { get; init; }
    public int Right { get; init; }
    public int Bottom { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
    public int ClientWidth { get; init; }
    public int ClientHeight { get; init; }
}

internal static class NativeMethods
{
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int nIndex);
}
