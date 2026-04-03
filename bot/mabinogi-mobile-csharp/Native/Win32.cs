using System.Runtime.InteropServices;

namespace MabinogiMacro.Native;

public static class Win32
{
    private const string User32 = "user32.dll";
    private const string Gdi32 = "gdi32.dll";
    private const string Kernel32 = "kernel32.dll";

    public const int WM_LBUTTONDOWN = 0x0201;
    public const int WM_LBUTTONUP = 0x0202;
    public const int WM_RBUTTONDOWN = 0x0204;
    public const int WM_RBUTTONUP = 0x0205;
    public const int WM_KEYDOWN = 0x0100;
    public const int WM_KEYUP = 0x0101;
    public const int WM_MOUSEMOVE = 0x0200;

    public const int MK_LBUTTON = 0x0001;
    public const int MK_RBUTTON = 0x0002;

    public const int VK_SPACE = 0x20;
    public const int VK_RETURN = 0x0D;
    public const int VK_ESCAPE = 0x1B;
    public const int VK_LEFT = 0x25;
    public const int VK_UP = 0x26;
    public const int VK_RIGHT = 0x27;
    public const int VK_DOWN = 0x28;
    public const int VK_TAB = 0x09;
    public const int VK_SHIFT = 0x10;
    public const int VK_CONTROL = 0x11;

    public const int MOUSEEVENTF_LEFTDOWN = 0x0002;
    public const int MOUSEEVENTF_LEFTUP = 0x0004;
    public const int MOUSEEVENTF_RIGHTDOWN = 0x0008;
    public const int MOUSEEVENTF_RIGHTUP = 0x0010;
    public const int MOUSEEVENTF_ABSOLUTE = 0x8000;
    public const int MOUSEEVENTF_MOVE = 0x0001;

    public const int INPUT_MOUSE = 0;
    public const int INPUT_KEYBOARD = 1;

    public const int SW_RESTORE = 9;
    public const int SRCCOPY = 0x00CC0020;
    public const int DIB_RGB_COLORS = 0;

    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT
    {
        public uint type;
        public INPUTUNION u;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct INPUTUNION
    {
        [FieldOffset(0)] public MOUSEINPUT mi;
        [FieldOffset(0)] public KEYBDINPUT ki;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT
    {
        public int dx;
        public int dy;
        public uint mouseData;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT
    {
        public ushort wVk;
        public ushort wScan;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT
    {
        public int Left, Top, Right, Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BITMAPINFOHEADER
    {
        public uint biSize;
        public int biWidth;
        public int biHeight;
        public ushort biPlanes;
        public ushort biBitCount;
        public uint biCompression;
        public uint biSizeImage;
        public int biXPelsPerMeter;
        public int biYPelsPerMeter;
        public uint biClrUsed;
        public uint biClrImportant;
    }

    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport(User32, SetLastError = true)]
    public static extern IntPtr FindWindow(string? lpClassName, string lpWindowName);

    [DllImport(User32)]
    public static extern bool IsWindow(IntPtr hWnd);

    [DllImport(User32)]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport(User32)]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport(User32)]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport(User32)]
    public static extern bool GetClientRect(IntPtr hWnd, out RECT lpRect);

    [DllImport(User32)]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport(User32)]
    public static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

    [DllImport(User32)]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

    [DllImport(User32)]
    public static extern bool SetCursorPos(int X, int Y);

    [DllImport(User32)]
    public static extern void mouse_event(uint dwFlags, int dx, int dy, uint cButtons, uint dwExtraInfo);

    [DllImport(User32)]
    public static extern uint SendInput(uint nInputs, [In] INPUT[] pInputs, int cbSize);

    [DllImport(User32)]
    public static extern IntPtr GetWindowDC(IntPtr hWnd);

    [DllImport(User32)]
    public static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

    [DllImport(User32)]
    public static extern IntPtr CreateCompatibleDC(IntPtr hDC);

    [DllImport(User32)]
    public static extern IntPtr CreateCompatibleBitmap(IntPtr hDC, int width, int height);

    [DllImport(User32)]
    public static extern IntPtr SelectObject(IntPtr hDC, IntPtr hObject);

    [DllImport(Gdi32)]
    public static extern bool BitBlt(IntPtr hdcDest, int xDest, int yDest, int wDest, int hDest,
        IntPtr hdcSrc, int xSrc, int ySrc, int rop);

    [DllImport(Gdi32)]
    public static extern int GetDIBits(IntPtr hdc, IntPtr hbmp, uint uStartScan, uint cScanLines,
        [Out] byte[] lpvBits, ref BITMAPINFOHEADER lpbi, uint uUsage);

    [DllImport(User32)]
    public static extern bool DeleteObject(IntPtr hObject);

    [DllImport(User32)]
    public static extern bool DeleteDC(IntPtr hDC);

    [DllImport(Kernel32)]
    public static extern bool ProcessIdToSessionId(uint dwProcessId, out uint pSessionId);

    [DllImport(User32)]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport(User32, CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

    [DllImport(User32)]
    public static extern int GetSystemMetrics(int nIndex);

    public static IntPtr MakeLParam(int loWord, int hiWord)
    {
        return (IntPtr)((loWord & 0xFFFF) | ((hiWord & 0xFFFF) << 16));
    }

    public static (int X, int Y) LParamToCoords(IntPtr lParam)
    {
        int val = lParam.ToInt32();
        return (val & 0xFFFF, (val >> 16) & 0xFFFF);
    }

    public static ushort MakeWParam(int loWord, int hiWord)
    {
        return (ushort)((loWord & 0xFFFF) | ((hiWord & 0xFFFF) << 16));
    }
}
