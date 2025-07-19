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
                Session session = new Session();
                session.Start(clientSocket);

                byte[] sendBuffer = Encoding.UTF8.GetBytes("Welcome to MMOPRG Server~!");
                session.Send(sendBuffer);

                Thread.Sleep(1000);

                session.Disconnect();
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
