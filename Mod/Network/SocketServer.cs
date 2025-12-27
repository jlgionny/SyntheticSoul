using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.IO;
using System.Threading;
using Modding;

namespace SyntheticSoulMod
{
    public class SocketServer
    {
        private TcpListener tcpListener;
        private int port;
        private bool isRunning = false;
        public Action<InputHandler.AIAction> ActionCallback { get; set; }

        public SocketServer(int port)
        {
            this.port = port;
            tcpListener = new TcpListener(IPAddress.Loopback, port);
        }

        public void Start(GameStateCapture stateCapture)
        {
            try
            {
                tcpListener.Start();
                isRunning = true;
                Modding.Logger.Log($"[Synthetic Soul] SocketServer ✅ In ascolto su 127.0.0.1:{port}");

                while (isRunning)
                {
                    try
                    {
                        if (tcpListener.Pending())
                        {
                            TcpClient client = tcpListener.AcceptTcpClient();
                            ThreadPool.QueueUserWorkItem(_ => HandleClient(client, stateCapture));
                        }
                        Thread.Sleep(10);
                    }
                    catch { }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[Synthetic Soul] SocketServer ❌ Errore: {e.Message}");
            }
        }

        private void HandleClient(TcpClient client, GameStateCapture stateCapture)
        {
            try
            {
                using (NetworkStream stream = client.GetStream())
                using (StreamReader reader = new StreamReader(stream, new UTF8Encoding(false)))
                using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
                {
                    string gameStateJSON = stateCapture.GetStateAsJSON();
                    writer.WriteLine(gameStateJSON);

                    string actionLine = reader.ReadLine();
                    if (!string.IsNullOrEmpty(actionLine))
                    {
                        Modding.Logger.Log($"[SocketServer] 📩 Azione ricevuta {actionLine}");
                        if (ActionCallback != null)
                        {
                            var action = InputHandler.ParseAction(actionLine);
                            ActionCallback(action);
                        }
                    }
                }
            }
            catch { }
            finally
            {
                client.Close();
            }
        }

        public void Stop()
        {
            isRunning = false;
            try { tcpListener?.Stop(); } catch { }
        }
    }
}
