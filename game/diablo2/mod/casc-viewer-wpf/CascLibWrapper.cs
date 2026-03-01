using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CascViewerWPF
{
    public class CascLibWrapper
    {
        private const string CascLibDll = "CascLib.dll";

        // CASC_OPEN_FLAGS
        public const uint CASC_OPEN_CASCPORTAL = 0x0001;
        public const uint CASC_OPEN_ONLINE     = 0x0002;
        public const uint CASC_OPEN_LOCAL      = 0x0004;

        [DllImport(CascLibDll, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern bool CascOpenStorage(string szStoragePath, uint dwFlags, out IntPtr phStorage);

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascCloseStorage(IntPtr hStorage);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        public struct CASC_FIND_DATA
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
            public string szFileName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
            public string szPlainName;
            public uint dwFileIndex;
            public uint dwFileSize;
            public uint dwFileAttributes;
            public uint dwLocaleFlags;
            public uint dwContentFlags;
        }

        [DllImport(CascLibDll, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern IntPtr CascFindFirstFile(IntPtr hStorage, string szMask, ref CASC_FIND_DATA pFindData, string? szListFile);

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascFindNextFile(IntPtr hFind, ref CASC_FIND_DATA pFindData);

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascFindClose(IntPtr hFind);

        [DllImport(CascLibDll, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern bool CascOpenFile(IntPtr hStorage, string szFileName, uint dwLocaleFlags, uint dwOpenFlags, out IntPtr phFile);

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascReadFile(IntPtr hFile, byte[] lpBuffer, uint dwToRead, out uint pdwRead);

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascCloseFile(IntPtr hFile);

        [DllImport(CascLibDll, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern bool CascExtractFile(IntPtr hStorage, string szFileName, string szLocalFileName, uint dwLocaleFlags);

        // CASC_STORAGE_INFO_CLASS
        public const int CascStorageFileCount = 0;
        public const int CascStorageFeatures  = 1;
        public const int CascStorageGameInfo  = 2;

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascGetStorageInfo(IntPtr hStorage, int InfoClass, byte[] pvStorageInfo, uint cbStorageInfo, out uint pcbLengthNeeded);
    }
}
