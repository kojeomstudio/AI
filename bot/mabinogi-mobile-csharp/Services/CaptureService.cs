using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MabinogiMacro.Native;
using Serilog;

namespace MabinogiMacro.Services;

public class CaptureService
{
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

        var bmi = new Win32.BITMAPINFO();
        bmi.bmiHeader.biSize = (uint)Marshal.SizeOf<Win32.BITMAPINFOHEADER>();
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = height;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = 0;
        bmi.bmiHeader.biSizeImage = (uint)(width * height * 4);
        bmi.bmiColors = new int[1];

        var pixels = new byte[width * height * 4];
        Win32.GetDIBits(hdcDest, hBitmap, 0, (uint)height, pixels, ref bmi, Win32.DIB_RGB_COLORS);

        unsafe
        {
            var dest = (byte*)bmpData.Scan0;
            for (int i = 0; i < pixels.Length; i += 4)
            {
                dest[i] = pixels[i + 2];
                dest[i + 1] = pixels[i + 1];
                dest[i + 2] = pixels[i];
                dest[i + 3] = 255;
            }
        }

        bitmap.UnlockBits(bmpData);
        Win32.SelectObject(hdcDest, IntPtr.Zero);
        Win32.DeleteObject(hBitmap);
        Win32.DeleteDC(hdcDest);
        Win32.ReleaseDC(hwnd, hdcSrc);

        return bitmap;
    }

    public byte[]? CaptureWindowToBgr(IntPtr hwnd)
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

        var bmi = new Win32.BITMAPINFO();
        bmi.bmiHeader.biSize = (uint)Marshal.SizeOf<Win32.BITMAPINFOHEADER>();
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = 0;
        bmi.bmiHeader.biSizeImage = (uint)(width * height * 4);
        bmi.bmiColors = new int[1];

        var bgra = new byte[width * height * 4];
        Win32.GetDIBits(hdcDest, hBitmap, 0, (uint)height, bgra, ref bmi, Win32.DIB_RGB_COLORS);

        var bgr = new byte[width * height * 3];
        int srcIdx = 0;
        for (int i = 0; i < bgr.Length; i += 3)
        {
            bgr[i] = bgra[srcIdx];
            bgr[i + 1] = bgra[srcIdx + 1];
            bgr[i + 2] = bgra[srcIdx + 2];
            srcIdx += 4;
        }

        Win32.SelectObject(hdcDest, IntPtr.Zero);
        Win32.DeleteObject(hBitmap);
        Win32.DeleteDC(hdcDest);
        Win32.ReleaseDC(hwnd, hdcSrc);

        return bgr;
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
