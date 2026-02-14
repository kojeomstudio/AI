using ServerCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Server
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

            if (success == false)
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

    class ClientSession : PacketSession
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
            ushort count = 0;

            ushort size = BitConverter.ToUInt16(buffer.Array, buffer.Offset + count);
            count +=2;

            ushort packetId = BitConverter.ToUInt16(buffer.Array, buffer.Offset + count);
            count +=2;

            switch((PacketID)packetId)
            {
                case PacketID.PlayerInfoReq:
                    {
                        PlayerInfoReq playerInfoReq = new PlayerInfoReq();
                        playerInfoReq.Read(buffer);

                        ServerLogger.Instance.Log(LogLevel.Info, $"PlayerInfoReq Recv : PlayerId : {playerInfoReq.playerId}");
                    }
                    break;
            }

            ServerLogger.Instance.Log(LogLevel.Info, $"RecvPacketId : {packetId}, Size : {size}");
        }

        public override void OnSend(int numOfBytes)
        {
            ServerLogger.Instance.Log(LogLevel.Info, $"Transferred Bytes : {numOfBytes}");
        }
    }
}
