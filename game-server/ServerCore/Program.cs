using System.Net;
using System.Net.Sockets;
using System.Text;

namespace ServerCore
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

            Socket listenSocket = new Socket
            (
                endPoint.AddressFamily,
                SocketType.Stream,
                ProtocolType.Tcp
            );
            
            try
            {
                listenSocket.Bind(endPoint);
                listenSocket.Listen(10);

                while (true)
                {
                    ServerLogger.Instance.Log(LogLevel.Info, "Waiting for a connection...");

                    Socket clientSocket = listenSocket.Accept();

                    byte[] recvBuffer = new Byte[1024];
                    int recvBytes = clientSocket.Receive(recvBuffer);

                    string recvData = Encoding.UTF8.GetString(recvBuffer, 0, recvBytes);
                    ServerLogger.Instance.Log(LogLevel.Info, $"Received data: {recvData}");

                    byte[] sendBuffer = Encoding.UTF8.GetBytes("Hello from server!");
                    clientSocket.Send(sendBuffer);

                    clientSocket.Shutdown(SocketShutdown.Both);
                    clientSocket.Close();
                }
            }
            catch (Exception ex)
            {
                ServerLogger.Instance.Log(LogLevel.Error, $"Error setting up server: {ex.Message}");
                return;
            }
        }
    }
}
