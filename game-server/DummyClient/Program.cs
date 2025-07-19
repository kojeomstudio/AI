using System.Net;
using System.Net.Sockets;
using System.Text;

namespace DummyClient
{
    internal class Program
    {
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

            while (true)
            {
                Thread.Sleep(1000);

                try
                {
                    Socket socket = new Socket(endPoint.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

                    socket.Connect(endPoint);
                    ClientLogger.Instance.Info($"Connected to {socket.RemoteEndPoint}");

                    for (int i = 0; i < 5; i++)
                    {
                        byte[] sendBuffer = Encoding.UTF8.GetBytes("Hello from client!");
                        int sentBytes = socket.Send(sendBuffer);

                        ClientLogger.Instance.Info($"Sent {sentBytes} bytes to server, index : {i}");
                    }

                    byte[] recvBuffer = new byte[1024];
                    int recvBytes = socket.Receive(recvBuffer);

                    string revcData = Encoding.UTF8.GetString(recvBuffer, 0, recvBytes);
                    ClientLogger.Instance.Info($"Received data: {revcData}");

                    socket.Shutdown(SocketShutdown.Both);
                    socket.Close();
                }
                catch (SocketException ex)
                {
                    ClientLogger.Instance.Error($"Socket error: {ex.Message}");
                    return;
                }
                catch (Exception ex)
                {
                    ClientLogger.Instance.Error($"Unexpected error: {ex.Message}");
                    return;
                }
            }
        }
    }
}
