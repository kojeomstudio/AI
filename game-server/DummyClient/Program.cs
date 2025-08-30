using ServerCore;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace DummyClient
{
    internal class Program
    {
        class GameSession : Session
        {
            public override void OnConnected(EndPoint endPoint)
            {
                Console.WriteLine($"OnConnected EndPoint : {endPoint}");

                for (int i = 0; i < 5; i++)
                {
                    byte[] sendBuffer = Encoding.UTF8.GetBytes("Hello from client!");
                    Send(sendBuffer);

                    ClientLogger.Instance.Info($"Sent {sendBuffer.Length} bytes to server, index : {i}");
                }
            }

            public override void OnDisconnected(EndPoint endPoint)
            {
                Console.WriteLine($"OnDisconnected EndPoint : {endPoint}");
            }

            public override int OnReceive(ArraySegment<byte> buffer)
            {

                string recvData = Encoding.UTF8.GetString(buffer.Array, buffer.Offset, buffer.Count);
                Console.WriteLine($"[From Server] {recvData}");
                return buffer.Count;
            }

            public override void OnSend(int numOfBytes)
            {
                Console.WriteLine($"Transferred Bytes : {numOfBytes}");
            }
        }
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

            Connector connector = new Connector();
            connector.Connect(endPoint, () => { return new GameSession(); });

            while (true)
            {
                Thread.Sleep(1000);

            }
        }
    }
}
