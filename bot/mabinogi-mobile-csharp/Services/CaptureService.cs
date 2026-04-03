using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MabinogiMacro.Native;
using Serilog;

namespace MabinogiMacro.Services;

public class CaptureService
{
    [DllImport("gdi32.dll")]
    private static extern int GetDIBits(IntPtr hdc, IntPtr hbmp, uint uStartScan, uint cScanLines,
        byte[] lpvBits, ref BITMAPINFO lpbmi, uint uUsage);

    [StructLayout(LayoutKind.Sequential)]
    private struct BITMAPINFO
    {
        public BITMAPINFOHEADER bmiHeader;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1)]
        public int[] bmiColors;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BITMAPINFOHEADER
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

    public Bitmap? CaptureWindow(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero) return null;

        if (!Win32.GetWindowRect(hwnd, out var rect)) return null;
        int width = rect.Right - rect.Left;
        int height = rect.Bottom - rect.Top;
        if (width <= 0 || height <= 0) return null;

        var hdcSrc = Win32.GetWindowDC(hwnd);
        var hdcDest = Win32.CreateCompatibleDC(hdcSrc);
        var hBitmap = Win32.CreateCompatibleBitmap(hdcSrc, width, height);
        if (hBitmap == IntPtr.Zero)
        {
            Win32.ReleaseDC(hwnd, hdcSrc);
            Win32.DeleteDC(hdcDest);
            return null;
        }

        Win32.SelectObject(hdcDest, hBitmap);
        Win32.BitBlt(hdcDest, 0, 0, width, height, hdcSrc, 0, 0, Win32.SRCCOPY);

        var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        var bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height),
            ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

        var bmi = new BITMAPINFO();
        bmi.bmiHeader.biSize = (uint)Marshal.SizeOf<BITMAPINFOHEADER>();
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = height; // positive = bottom-up DIB
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = 0;
        bmi.bmiHeader.biSizeImage = (uint)(width * height * 4);
        bmi.bmiColors = new int[1];

        var pixels = new byte[width * height * 4];
        GetDIBits(hdcDest, hBitmap, 0, (uint)height, pixels, ref bmi, 0);

        Marshal.Copy(pixels, 0, bmpData.Scan0, pixels.Length);

        bitmap.UnlockBits(bmpData);
        Win32.SelectObject(hdcDest, IntPtr.Zero);
        Win32.DeleteObject(hBitmap);
        Win32.DeleteDC(hdcDest);
        Win32.ReleaseDC(hwnd, hdcSrc);

        return bitmap;
    }

    public Bitmap? CaptureWindow(string windowTitle)
    {
        var hwnd = Win32.FindWindow(null, windowTitle);
        if (hwnd == IntPtr.Zero)
        {
            Log.Warning("CaptureWindow: window not found '{WindowTitle}'", windowTitle);
            return null;
        }

        Win32.ShowWindow(hwnd, Win32.SW_RESTORE);
        return CaptureWindow(hwnd);
    }
}
