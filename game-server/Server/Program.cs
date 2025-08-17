using ServerCore;
using System.Net;
using System.Text;

namespace KojeomGameServer
{
    class GameSession : Session
    {
        public override void OnConnected(EndPoint endPoint)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"OnConnected EndPoint : {endPoint}");

            byte[] sendBuffer = Encoding.UTF8.GetBytes("Welcome to MMOPRG Server~!");
            Send(sendBuffer);

            Thread.Sleep(1000);

            Disconnect();
        }

        public override void OnDisconnected(EndPoint endPoint)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"OnDisconnected EndPoint : {endPoint}");
        }

        public override void OnReceive(ArraySegment<byte> buffer)
        {

            string recvData = Encoding.UTF8.GetString(buffer.Array, buffer.Offset, buffer.Count);
            ServerLogger.Instance.Log(LogLevel.Info, $"[From Client] {recvData}");

        }

        public override void OnSend(int numOfBytes)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"Transferred Bytes : {numOfBytes}");
        }
    }

    internal class Program
    {
        static Listener _listener = new Listener();

        static void Main(string[] args)
        {
            string host = Dns.GetHostName();
            IPHostEntry ipHost = Dns.GetHostEntry(host);
            IPAddress ipAddr = ipHost.AddressList[0];
            IPEndPoint endPoint = new IPEndPoint
            (
                ipAddr,
                7777
             );

            _listener.Init(endPoint, () => { return new GameSession(); });

            while (true)
            {

            }
        }
    }
}
