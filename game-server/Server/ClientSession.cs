using ServerCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Server
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
                        long playerId = BitConverter.ToInt64(buffer.Array, buffer.Offset + count);
                        count += 8;
                        ServerLogger.Instance.Log(LogLevel.Info, $"PlayerInfoReq Recv : PlayerId : {playerId}");
                    }
                    break;
                case PacketID.PlayerInfoOk:
                    {
                        int hp = BitConverter.ToInt32(buffer.Array, buffer.Offset + count);
                        count += 4;
                        int attack = BitConverter.ToInt32(buffer.Array, buffer.Offset + count);
                        count += 4;
                        ServerLogger.Instance.Log(LogLevel.Info, $"PlayerInfoOk Recv : Hp : {hp}, Attack : {attack}");
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
