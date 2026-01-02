using ServerCore;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace DummyClient
{
    internal class Program
    {
        public class Packet
        {
            public ushort size;
            public ushort packetId;
        }

        class GameSession : Session
        {
            public override void OnConnected(EndPoint endPoint)
            {
                Console.WriteLine($"OnConnected EndPoint : {endPoint}");

                Packet packet = new Packet() { size = 100, packetId = 10 };

                for (int i = 0; i < 5; i++)
                {
                    ArraySegment<byte> openSegment = SendBufferHelper.Open(4096);

                    byte[] sizeBuffer = BitConverter.GetBytes(packet.size);
                    byte[] packetIdBuffer = BitConverter.GetBytes(packet.packetId);

                    Array.Copy(sizeBuffer, 0, openSegment.Array, openSegment.Offset, sizeBuffer.Length);
                    Array.Copy(packetIdBuffer, 0, openSegment.Array, openSegment.Offset + sizeBuffer.Length, packetIdBuffer.Length);
                    ArraySegment<byte> sendBuffer = SendBufferHelper.Close(packet.size);

                    Send(sendBuffer);

                    ClientLogger.Instance.Info($"Sent {sendBuffer.Count} bytes to server, index : {i}");
                }
            }

            public override void OnDisconnected(EndPoint endPoint)
            {
                Console.WriteLine($"OnDisconnected EndPoint : {endPoint}");
            }

            public override int OnRecv(ArraySegment<byte> buffer)
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
