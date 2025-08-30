﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace ServerCore
{
    public abstract class Session
    {
        private Socket _socket;
        private int _disconnected = 0;

        private RecvBuffer _recvBuffer = new RecvBuffer(65535);

        private SocketAsyncEventArgs _sendArgs = new SocketAsyncEventArgs();
        private SocketAsyncEventArgs _recvArgs = new SocketAsyncEventArgs();

        private object _lock = new object();
        private Queue<byte[]> _sendQueue = new Queue<byte[]>();
        private bool _pending = false;

        private List<ArraySegment<byte>> _pendinglist = new List<ArraySegment<byte>>();

        public void Start(Socket socket)
        {
            _socket = socket;

            
            _recvArgs.Completed += new EventHandler<SocketAsyncEventArgs>(OnRecvCompleted);
            _sendArgs.Completed += new EventHandler<SocketAsyncEventArgs>(OnSendCompleted);

            RegisterRecv(_recvArgs);
        }

        // Handlers.
        public abstract void OnConnected(EndPoint endPoint);

        public abstract int OnReceive(ArraySegment<byte> buffer);
        public abstract void OnSend(int numOfBytes);
        public abstract void OnDisconnected(EndPoint endPoint);
        // ~Handlers.

        public void Send(byte[] sendBuff)
        {
            lock(_lock)
            {
                _sendQueue.Enqueue(sendBuff);
                if (_pending == false)
                {
                    RegisterSend();
                }
            }
        }

        public void Disconnect()
        {
            if(Interlocked.Exchange(ref _disconnected, 1) == 1)
            {
                return;
            }

            OnDisconnected(_socket.RemoteEndPoint);

            _socket.Shutdown(SocketShutdown.Both);
            _socket.Close();
        }

        #region Network
        private void RegisterSend()
        {
            _pending = true;

            
            while (_sendQueue.Count > 0)
            {
                byte[] buff = _sendQueue.Dequeue();
                _pendinglist.Add(new ArraySegment<byte>(buff, 0, buff.Length));
            }
            
            _sendArgs.BufferList = _pendinglist;

            bool pending = _socket.SendAsync(_sendArgs);
            if(pending == false)
            {
                OnSendCompleted(null, _sendArgs);
            }
        }

        private void OnSendCompleted(object? sender, SocketAsyncEventArgs args)
        {
            lock(_lock)
            {
                if (args.BytesTransferred > 0 && args.SocketError == SocketError.Success)
                {
                    try
                    {
                        _sendArgs.BufferList = null;
                        _pendinglist.Clear();

                        OnSend(_sendArgs.BytesTransferred);

                        if (_sendQueue.Count > 0)
                        {
                            RegisterSend();
                        }
                        else
                        {
                            _pending = false;
                        }
                            
                    }
                    catch (Exception ex)
                    {
                        ServerLogger.Instance.Log(LogLevel.Error, $"OnSendCompleted Faield : {ex.ToString()}");
                    }
                }
                else
                {
                    //ServerLogger.Instance.Log(LogLevel.Error, $"OnSendCompleted Failed : {args.SocketError}");
                    Disconnect();
                }
            }
        }
        private void RegisterRecv(SocketAsyncEventArgs args)
        {
            _recvBuffer.Clean();

            ArraySegment<byte> segment = _recvBuffer.WriteSegment;
            _recvArgs.SetBuffer(segment.Array, segment.Offset, segment.Count);

            bool pending = _socket.ReceiveAsync(args);
            if (pending == false)
            {
                OnRecvCompleted(null, args);
            }
        }

        private void OnRecvCompleted(object? sender, SocketAsyncEventArgs args)
        {
            if(args.BytesTransferred > 0 && args.SocketError == SocketError.Success)
            {
                try
                {
                    // Write Cursor moved.
                    if (_recvBuffer.OnWrite(args.BytesTransferred) == false)
                    {
                        Disconnect();
                        return;
                    }

                    int processLen = OnReceive(_recvBuffer.ReadSegment);
                    if(processLen < 0 || _recvBuffer.DataSize < processLen)
                    {
                        Disconnect();
                        return;
                    }

                    if(_recvBuffer.OnRead(processLen) == false)
                    {
                        Disconnect();
                        return;
                    }

                    RegisterRecv(args);
                }
                catch (Exception ex)
                {
                    ServerLogger.Instance.Log(LogLevel.Error, $"OnRecvCompleted Faield : {ex.ToString()}");
                }
            }
            else
            {
                Disconnect();
            }
        }
        #endregion
    }
}
