using System.Net;
using System.Net.Sockets;
using System.Text;

namespace ServerCore
{
    internal class Program
    {
        static Listener _listener = new Listener();

        static void OnAcceptHandler(Socket clientSocket)
        {
            try
            {
                byte[] recvBuffer = new Byte[1024];
                int recvBytes = clientSocket.Receive(recvBuffer);

                string recvData = Encoding.UTF8.GetString(recvBuffer, 0, recvBytes);
                ServerLogger.Instance.Log(LogLevel.Info, $"Received data: {recvData}");

                byte[] sendBuffer = Encoding.UTF8.GetBytes("Hello from server!");
                clientSocket.Send(sendBuffer);

                clientSocket.Shutdown(SocketShutdown.Both);
                clientSocket.Close();
            }
            catch (Exception ex)
            {
                ServerLogger.Instance.Log(LogLevel.Error, $"Error setting up server: {ex.Message}");
                return;
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

            _listener.Init(endPoint, OnAcceptHandler);

            while(true)
            {

            }
        }
    }
}
