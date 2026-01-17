using ServerCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace DummyClient
{
    public class Packet
    {
        public ushort size;
        public ushort packetId;
    }

    public class PlayerInfoReq : Packet
    {
        public long playerId;
    }

    public class PlayerInfoOk : Packet
    {
        public int hp;
        public int attack;
    }

    public enum PacketID
    {
        PlayerInfoReq = 1,
        PlayerInfoOk = 2,
    }

    class ServerSession : Session
    {
        public override void OnConnected(EndPoint endPoint)
        {
            Console.WriteLine($"OnConnected EndPoint : {endPoint}");

            PlayerInfoReq packet = new PlayerInfoReq() { playerId = 1001, packetId = (ushort)PacketID.PlayerInfoReq };

            //for (int i = 0; i < 5; i++)
            {
                ArraySegment<byte> openSeg = SendBufferHelper.Open(4096);

                byte[] sizeBuffer = BitConverter.GetBytes(packet.size);
                byte[] packetIdBuffer = BitConverter.GetBytes(packet.packetId);
                byte[] playerIdBuffer = BitConverter.GetBytes(packet.playerId);

                ushort count = 0;

                //Array.Copy(sizeBuffer, 0, openSeg.Array, openSeg.Offset + count, 2);
                //count += 2;

                //Array.Copy(packetIdBuffer, 0, openSeg.Array, openSeg.Offset + count, 2);
                //count += 2;

                //Array.Copy(playerIdBuffer, 0, openSeg.Array, openSeg.Offset + count, 8);
                //count += 8;

                bool success = true;
                //success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset, openSeg.Count), packet.size);
                count += 2;
                success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset + count, openSeg.Count - count), packet.packetId);
                count += 2;
                success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset + count, openSeg.Count - count), packet.playerId);
                count += 8;
                success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset, openSeg.Count), count);

                packet.size = count;

                ArraySegment<byte> sendBuffer = SendBufferHelper.Close(packet.size);

                if(success)
                {
                    Send(sendBuffer);
                }

                ClientLogger.Instance.Info($"Sent {sendBuffer.Count} bytes to server");
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
}
