using ServerCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace DummyClient
{
    public abstract class Packet
    {
        public ushort size;
        public ushort packetId;

        public abstract ArraySegment<byte> Write();
        public abstract void Read(ArraySegment<byte> segment);
    }

    public class PlayerInfoReq : Packet
    {
        public long playerId;

        public PlayerInfoReq()
        {
            this.packetId = (ushort)PacketID.PlayerInfoReq;
        }

        public override void Read(ArraySegment<byte> segment)
        {
            ushort count = 0;

            //ushort size = BitConverter.ToUInt16(segment.Array, segment.Offset + count);
            count += 2;
            //this.packetId = BitConverter.ToUInt16(segment.Array, segment.Offset + count);
            count += 2;
            this.playerId = BitConverter.ToInt64(new ReadOnlySpan<byte>(segment.Array, segment.Offset + count, segment.Count - count));

            count += 8;
        }

        public override ArraySegment<byte> Write()
        {
            ArraySegment<byte> openSeg = SendBufferHelper.Open(4096);

            byte[] sizeBuffer = BitConverter.GetBytes(this.size);
            byte[] packetIdBuffer = BitConverter.GetBytes(this.packetId);
            byte[] playerIdBuffer = BitConverter.GetBytes(this.playerId);

            ushort count = 0;
            bool success = true;

            //success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset, openSeg.Count), packet.size);
            count += 2;
            success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset + count, openSeg.Count - count), this.packetId);
            count += 2;
            success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset + count, openSeg.Count - count), this.playerId);
            count += 8;
            success &= BitConverter.TryWriteBytes(new Span<byte>(openSeg.Array, openSeg.Offset, openSeg.Count), count);

            this.size = count;

            if(success == false)
            {
                return null;
            }

            return SendBufferHelper.Close(this.size);
        }
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

            PlayerInfoReq packet = new PlayerInfoReq() { playerId = 1001};

            //for (int i = 0; i < 5; i++)
            {
                ArraySegment<byte> segment = packet.Write();

                if(segment != null)
                {
                    Send(segment);
                }

                ClientLogger.Instance.Info($"Sent {segment.Count} bytes to server");
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
