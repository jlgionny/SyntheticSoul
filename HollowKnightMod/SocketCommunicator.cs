using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Modding;

namespace SyntheticSoulMod
{
    public class SocketCommunicator
    {
        private TcpListener server;
        private TcpClient client;
        private NetworkStream stream;
        private int port;
        private object lockObject = new object();
        private volatile bool isConnected = false;
        private volatile bool isRunning = false;
        private Thread serverThread;

        // Coda comandi (V41)
        private Queue<string> commandQueue = new Queue<string>();

        public SocketCommunicator(int port)
        {
            this.port = port;
        }

        public void StartServer()
        {
            if (isRunning) return;

            isRunning = true;
            serverThread = new Thread(ServerLoop);
            serverThread.IsBackground = true;
            serverThread.Start();

            Modding.Logger.Log($"[SocketCommunicator] Persistent server started on port {port}");
        }

        private void ServerLoop()
        {
            try
            {
                server = new TcpListener(IPAddress.Loopback, port);
                server.Start();

                while (isRunning)
                {
                    try
                    {
                        TcpClient newClient = server.AcceptTcpClient();

                        lock (lockObject)
                        {
                            CloseCurrentClient();
                            client = newClient;
                            client.NoDelay = true; // Importante per ridurre latenza
                            client.ReceiveBufferSize = 4096;
                            client.SendBufferSize = 4096;

                            stream = client.GetStream();

                            // Timeout aumentato a 10s per evitare disconnessioni se Python fa calcoli pesanti
                            stream.ReadTimeout = 10000;
                            stream.WriteTimeout = 10000;

                            isConnected = true;
                            commandQueue.Clear();
                        }

                        DesktopLogger.Log($"[Socket] Client Connected: {client.Client.RemoteEndPoint}");

                        while (isConnected && isRunning)
                        {
                            Thread.Sleep(100);
                        }
                    }
                    catch (Exception ex)
                    {
                        if (isRunning) Modding.Logger.LogError($"Server loop error: {ex.Message}");
                        Thread.Sleep(1000);
                    }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.LogError($"Fatal server error: {e.Message}");
                isRunning = false;
            }
            finally
            {
                server?.Stop();
            }
        }

        private void CloseCurrentClient()
        {
            isConnected = false;
            commandQueue.Clear();
            try { stream?.Close(); stream?.Dispose(); stream = null; } catch { }
            try { client?.Close(); client = null; } catch { }
        }

        public void SendState(GameState state)
        {
            if (!isConnected || stream == null) return;

            lock (lockObject)
            {
                if (!isConnected || stream == null || !stream.CanWrite) return;
                try
                {
                    string json = JsonConvert.SerializeObject(state);
                    byte[] data = Encoding.UTF8.GetBytes(json + "\n");
                    stream.Write(data, 0, data.Length);
                    // Non serve flush aggressivo con NoDelay, ma male non fa
                }
                catch (Exception)
                {
                    CloseCurrentClient();
                }
            }
        }

        public string ReceiveAction()
        {
            if (!isConnected || stream == null) return "IDLE";

            lock (lockObject)
            {
                // 1. Priorità alla Coda: Se abbiamo comandi residui (da burst precedenti), usiamoli.
                if (commandQueue.Count > 0)
                {
                    return commandQueue.Dequeue();
                }

                if (!isConnected || stream == null || !stream.CanRead) return "IDLE";

                try
                {
                    // =================================================================
                    // FIX SINCRONIZZAZIONE (V42): LETTURA BLOCCANTE
                    // Rimosso: if (!stream.DataAvailable) return "IDLE";
                    // Ora Stream.Read BLOCCA il thread finché Python non invia il comando.
                    // Questo costringe il gioco ad andare alla velocità di Python.
                    // =================================================================

                    byte[] buffer = new byte[4096];
                    int bytesRead = stream.Read(buffer, 0, buffer.Length); // <--- QUI SI FERMA E ASPETTA

                    if (bytesRead > 0)
                    {
                        string rawData = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                        string[] commands = rawData.Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                        foreach (string cmd in commands)
                        {
                            string cleanCmd = cmd.Trim().ToUpper();
                            if (!string.IsNullOrEmpty(cleanCmd))
                            {
                                commandQueue.Enqueue(cleanCmd);
                            }
                        }

                        if (commandQueue.Count > 0)
                        {
                            return commandQueue.Dequeue();
                        }
                    }
                }
                catch (System.IO.IOException)
                {
                    // Timeout scaduto (Python non risponde da 10s) -> Restituisci IDLE per sbloccare
                    return "IDLE";
                }
                catch (Exception ex)
                {
                    DesktopLogger.LogError($"Socket Read Error: {ex.Message}");
                    CloseCurrentClient();
                    return "IDLE";
                }
            }
            return "IDLE";
        }

        public void Close()
        {
            isRunning = false;
            lock (lockObject) { CloseCurrentClient(); }
            try { server?.Stop(); } catch { }
        }

        public bool IsConnected => isConnected;
    }
}