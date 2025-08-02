using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace ServerCore
{
    public class Session
    {
        private Socket _socket;
        private int _disconnected = 0;

        private SocketAsyncEventArgs _sendArgs = new SocketAsyncEventArgs();

        private object _lock = new object();
        private Queue<byte[]> _sendQueue = new Queue<byte[]>();
        private bool _pending = false;

        public void Start(Socket socket)
        {
            _socket = socket;

            SocketAsyncEventArgs recvArgs = new SocketAsyncEventArgs();
            recvArgs.Completed += new EventHandler<SocketAsyncEventArgs>(OnRecvCompleted);
            recvArgs.SetBuffer(new byte[1024], 0, 1024);

            _sendArgs.Completed += new EventHandler<SocketAsyncEventArgs>(OnSendCompleted);

            RegisterRecv(recvArgs);
        }

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

            _socket.Shutdown(SocketShutdown.Both);
            _socket.Close();
        }

        #region Network
        private void RegisterSend()
        {
            _pending = true;

            byte[] buff = _sendQueue.Dequeue();
            _sendArgs.SetBuffer(buff, 0, buff.Length);

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
                    string recvData = Encoding.UTF8.GetString(args.Buffer, args.Offset, args.BytesTransferred);
                    ServerLogger.Instance.Log(LogLevel.Info, $"[From Client] {recvData}");
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
