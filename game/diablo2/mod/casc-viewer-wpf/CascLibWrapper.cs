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

        [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi)]
        public unsafe struct CASC_FIND_DATA
        {
            [FieldOffset(0)]
            public fixed byte _szFileName[260]; // MAX_PATH = 260

            [FieldOffset(260)]
            public fixed byte CKey[16];

            [FieldOffset(276)]
            public fixed byte EKey[16];

            // Padding 4 bytes for 8-byte alignment of next field on x64 (292 -> 296)
            [FieldOffset(296)]
            public ulong TagBitMask;

            [FieldOffset(304)]
            public ulong FileSize;

            [FieldOffset(312)]
            public IntPtr szPlainName;

            [FieldOffset(320)]
            public uint dwFileDataId;

            [FieldOffset(324)]
            public uint dwLocaleFlags;

            [FieldOffset(328)]
            public uint dwContentFlags;

            [FieldOffset(332)]
            public uint dwSpanCount;

            // Bit-field and Enum following...
            // Since we only care about szFileName and FileSize for tree building,
            // we've ensured those offsets are correct. 
            // Total size should be around 352 bytes on x64.

            public string szFileName
            {
                get
                {
                    fixed (byte* p = _szFileName)
                    {
                        return Marshal.PtrToStringAnsi((IntPtr)p) ?? string.Empty;
                    }
                }
            }
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
        public const int CascStorageLocalFileCount   = 0;
        public const int CascStorageTotalFileCount   = 1;
        public const int CascStorageFeatures         = 2;
        public const int CascStorageInstalledLocales = 3;
        public const int CascStorageProduct          = 4;
        public const int CascStorageTags             = 5;

        [DllImport(CascLibDll, SetLastError = true)]
        public static extern bool CascGetStorageInfo(IntPtr hStorage, int InfoClass, byte[] pvStorageInfo, uint cbStorageInfo, out uint pcbLengthNeeded);
    }
}
