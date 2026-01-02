using ServerCore;
using System.Net;
using System.Text;

namespace KojeomGameServer
{
    public class Packet
    {
        public ushort size;
        public ushort packetId;
    }

    class GameSession : PacketSession
    {
        public override void OnConnected(EndPoint endPoint)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"OnConnected EndPoint : {endPoint}");

            //Packet packet = new Packet() { size = 100, packetId = 10 };

            //ArraySegment<byte> openSegment = SendBufferHelper.Open(4096);

            //byte[] sizeBuffer = BitConverter.GetBytes(packet.size);
            //byte[] packetIdBuffer = BitConverter.GetBytes(packet.packetId);

            //Array.Copy(sizeBuffer, 0, openSegment.Array, openSegment.Offset, sizeBuffer.Length);
            //Array.Copy(packetIdBuffer, 0, openSegment.Array, openSegment.Offset + sizeBuffer.Length, packetIdBuffer.Length);
            //ArraySegment<byte> sendBuffer = SendBufferHelper.Close(packet.size);

            //Send(sendBuffer);

            Thread.Sleep(5000);

            Disconnect();
        }

        public override void OnDisconnected(EndPoint endPoint)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"OnDisconnected EndPoint : {endPoint}");
        }

        public override void OnRecvPacket(ArraySegment<byte> buffer)
        {
            ushort size = BitConverter.ToUInt16(buffer.Array, buffer.Offset);
            ushort packetId = BitConverter.ToUInt16(buffer.Array, buffer.Offset + PacketSession.HeaderSize);
            
            ServerLogger.Instance.Log(LogLevel.Info, $"RecvPacketId : {packetId}, Size : {size}");
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
