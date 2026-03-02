using System;
using System.Runtime.InteropServices;
using System.Text;

namespace D2RModMaster
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
        public unsafe struct CASC_FIND_DATA
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
            public string szFileName;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] CKey;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] EKey;

            public ulong TagBitMask;
            public ulong FileSize;
            public IntPtr szPlainName;
            public uint dwFileDataId;
            public uint dwLocaleFlags;
            public uint dwContentFlags;
            public uint dwSpanCount;
            
            // Bit-field and other trailing members can be omitted if not accessed
            // but we need enough padding for the structure size
            private uint _bitFields;
            private uint _nameType;
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
