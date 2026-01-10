using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Modding; // Serve per il logging

namespace SyntheticSoulMod
{
    public class SocketServer
    {
        private TcpListener listener;
        private Thread listenerThread;
        private int port;
        private volatile bool running = false;
        private GameStateCapture stateCapture;
        public Action<string> ActionCallback;
        private Mod logger;

        public SocketServer(int port, Mod modLogger = null)
        {
            this.port = port;
            this.logger = modLogger;
        }

        public void Start(GameStateCapture capture)
        {
            this.stateCapture = capture;
            running = true;
            listenerThread = new Thread(ListenForClients);
            listenerThread.IsBackground = true;
            listenerThread.Start();
        }

        private void ListenForClients()
        {
            try
            {
                listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
                listener.Start();
                if(logger!=null) logger.Log($"[Socket] Server avviato su {port}");

                while (running)
                {
                    try
                    {
                        if (!listener.Pending()) { Thread.Sleep(50); continue; }
                        TcpClient client = listener.AcceptTcpClient();
                        if(logger!=null) logger.Log("[Socket] Python Connesso!");
                        HandleClient(client);
                    }
                    catch (Exception ex)
                    {
                        if(logger!=null) logger.Log($"[Socket] Loop Error: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                if(logger!=null) logger.Log($"[Socket] Start Error: {ex.Message}");
            }
            finally
            {
                listener?.Stop();
            }
        }

        private void HandleClient(TcpClient client)
        {
            using (NetworkStream stream = client.GetStream())
            {
                byte[] buffer = new byte[1024];
                while (client.Connected && running)
                {
                    try
                    {
                        // Usa il metodo sicuro senza JsonUtility
                        string jsonState = stateCapture.GetStateJson();
                        byte[] data = Encoding.UTF8.GetBytes(jsonState + "\n");
                        stream.Write(data, 0, data.Length);

                        if (stream.DataAvailable)
                        {
                            int bytesRead = stream.Read(buffer, 0, buffer.Length);
                            if (bytesRead > 0)
                            {
                                string action = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                                ActionCallback?.Invoke(action);
                            }
                        }
                        Thread.Sleep(10); 
                    }
                    catch (Exception ex)
                    {
                        if(logger!=null) logger.Log($"[Socket] Client Error: {ex.Message}");
                        break;
                    }
                }
            }
            if(logger!=null) logger.Log("[Socket] Client Disconnesso.");
        }
    }
}