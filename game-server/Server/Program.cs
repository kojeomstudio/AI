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
           
            ArraySegment<byte> openSegment = SendBufferHelper.Open(4096);
            string testMessage = "Welcome to MMOPRG Server~!";
            Array.Copy(Encoding.UTF8.GetBytes(testMessage), 0, openSegment.Array, openSegment.Offset, testMessage.Length);
            ArraySegment<byte> sendBuffer = SendBufferHelper.Close(testMessage.Length);

            Send(sendBuffer);

            Thread.Sleep(1000);

            Disconnect();
        }

        public override void OnDisconnected(EndPoint endPoint)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"OnDisconnected EndPoint : {endPoint}");
        }

        public override int OnReceive(ArraySegment<byte> buffer)
        {

            string recvData = Encoding.UTF8.GetString(buffer.Array, buffer.Offset, buffer.Count);
            ServerLogger.Instance.Log(LogLevel.Info, $"[From Client] {recvData}");

            return buffer.Count;
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
