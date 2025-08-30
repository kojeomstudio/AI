using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServerCore
{
    public class RecvBuffer
    {
        private ArraySegment<byte> _buffer;

        private int _readPos;
        private int _writePos;

        public RecvBuffer(int bufferSize)
        {
            _buffer = new ArraySegment<byte>(new byte[bufferSize], 0, bufferSize);
        }

        public int DataSize { get { return _writePos - _readPos; } }
        public int FreeSize { get { return _buffer.Count - _writePos; } }

        /// <summary>
        /// ReadSegement == DataSegment
        /// </summary>
        public ArraySegment<byte> ReadSegment // DataSegment
        {
            get { return new ArraySegment<byte>(_buffer.Array, _buffer.Offset + _readPos, DataSize); }
        }

        /// <summary>
        /// WriteSegment == RecvSegment
        /// </summary>
        public ArraySegment<byte> WriteSegment // RecvSegment
        {
            get { return new ArraySegment<byte>(_buffer.Array, _buffer.Offset + _writePos, FreeSize); }
        }

        public void Clean()
        {
            int dataSize = DataSize;
            if (dataSize == 0)
            {
                // 남은 데이터가 없으면, 버퍼를 비운다.
                _readPos = _writePos = 0;
            }
            else
            {
                // 남은 데이터가 있으면, 버퍼의 맨 앞으로 옮긴다.
                Array.Copy(_buffer.Array, _buffer.Offset + _readPos, _buffer.Array, _buffer.Offset, dataSize);
                _readPos = 0;
                _writePos = dataSize;
            }
        }

        public bool OnRead(int numOfBytes)
        {
            if (numOfBytes > DataSize)
            {
                return false;
            }

            _readPos += numOfBytes;
            return true;
        }

        public bool OnWrite(int numOfBytes)
        {
            if (numOfBytes > FreeSize)
            {
                return false;
            }

            _writePos += numOfBytes;
            return true;
        }
}
