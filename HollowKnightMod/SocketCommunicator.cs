using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;

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

        public SocketCommunicator(int port)
        {
            this.port = port;
        }

        /// <summary>
        /// FIX: Persistent server that runs in a loop, accepting multiple connections
        /// </summary>
        public void StartServer()
        {
            if (isRunning)
            {
                Modding.Logger.LogWarn("Server already running!");
                return;
            }

            isRunning = true;

            // FIX: Run server loop on a dedicated background thread
            serverThread = new Thread(ServerLoop);
            serverThread.IsBackground = true;
            serverThread.Start();

            Modding.Logger.Log($"[SocketCommunicator] Persistent server thread started on port {port}");
            DesktopLogger.Log($"SocketCommunicator: Persistent server started on port {port}");
        }

        /// <summary>
        /// FIX: Main server loop - continuously accepts new connections
        /// </summary>
        private void ServerLoop()
        {
            try
            {
                // Initialize TcpListener once
                server = new TcpListener(IPAddress.Loopback, port);
                server.Start();
                
                Modding.Logger.Log($"[SocketCommunicator] TcpListener started on port {port}");
                DesktopLogger.Log($"TcpListener started, waiting for connections...");

                // FIX: Infinite loop to accept multiple client connections
                while (isRunning)
                {
                    try
                    {
                        Modding.Logger.Log("[SocketCommunicator] Waiting for Python agent to connect...");
                        DesktopLogger.Log("Waiting for client connection...");

                        // FIX: Blocking call - waits for new connection
                        // This is OK since we're on a background thread
                        TcpClient newClient = server.AcceptTcpClient();

                        // FIX: Close any existing client before accepting new one
                        lock (lockObject)
                        {
                            CloseCurrentClient();

                            // Setup new client
                            client = newClient;
                            client.NoDelay = true;
                            stream = client.GetStream();
                            stream.ReadTimeout = 100;
                            stream.WriteTimeout = 100;
                            isConnected = true;
                        }

                        string clientEndpoint = client.Client.RemoteEndPoint?.ToString() ?? "Unknown";
                        Modding.Logger.Log($"[SocketCommunicator] ✓ Client connected: {clientEndpoint}");
                        DesktopLogger.Log($"✓ NEW CLIENT CONNECTED: {clientEndpoint} at {DateTime.Now:HH:mm:ss}");

                        // FIX: Monitor connection health
                        MonitorConnection();
                    }
                    catch (SocketException se)
                    {
                        if (isRunning)
                        {
                            Modding.Logger.LogError($"[SocketCommunicator] Socket error in accept loop: {se.Message}");
                            DesktopLogger.LogError($"Socket error: {se.Message}");
                            Thread.Sleep(1000); // Brief pause before retrying
                        }
                    }
                    catch (Exception ex)
                    {
                        if (isRunning)
                        {
                            Modding.Logger.LogError($"[SocketCommunicator] Error in server loop: {ex.Message}");
                            DesktopLogger.LogError($"Server loop error: {ex.Message}");
                            Thread.Sleep(1000);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.LogError($"[SocketCommunicator] Fatal server error: {e.Message}\n{e.StackTrace}");
                DesktopLogger.LogError($"Fatal server error: {e.Message}\n{e.StackTrace}");
                isRunning = false;
            }
            finally
            {
                Modding.Logger.Log("[SocketCommunicator] Server loop ended");
                DesktopLogger.Log("Server loop ended");
            }
        }

        /// <summary>
        /// FIX: Monitor connection and detect disconnections
        /// </summary>
        private void MonitorConnection()
        {
            // This runs on the server thread after accepting a client
            // We don't actively monitor here - disconnection is detected during Send/Receive
            // Just log that monitoring is active
            Modding.Logger.Log("[SocketCommunicator] Connection monitoring active");
        }

        /// <summary>
        /// FIX: Close current client connection (but keep server running)
        /// </summary>
        private void CloseCurrentClient()
        {
            // Must be called inside lock(lockObject)
            if (isConnected)
            {
                Modding.Logger.Log("[SocketCommunicator] Closing current client connection...");
                DesktopLogger.Log($"Closing client connection at {DateTime.Now:HH:mm:ss}");
            }

            isConnected = false;

            try
            {
                stream?.Close();
                stream?.Dispose();
                stream = null;
            }
            catch (Exception e)
            {
                Modding.Logger.LogWarn($"Error closing stream: {e.Message}");
            }

            try
            {
                client?.Close();
                client = null;
            }
            catch (Exception e)
            {
                Modding.Logger.LogWarn($"Error closing client: {e.Message}");
            }
        }

        public void SendState(GameState state)
        {
            if (!isConnected || stream == null)
                return;

            lock (lockObject)
            {
                // Double-check inside lock
                if (!isConnected || stream == null || !stream.CanWrite)
                    return;

                try
                {
                    string json = JsonConvert.SerializeObject(state);
                    byte[] data = Encoding.UTF8.GetBytes(json + "\n");
                    stream.Write(data, 0, data.Length);
                    stream.Flush();
                }
                catch (System.IO.IOException ioEx)
                {
                    // FIX: Connection lost during write
                    Modding.Logger.LogWarn($"[SocketCommunicator] Client disconnected during SendState: {ioEx.Message}");
                    DesktopLogger.LogWarning($"Client disconnected during SendState at {DateTime.Now:HH:mm:ss}");
                    
                    // FIX: Close client but DON'T stop the server - it will accept new connections
                    CloseCurrentClient();
                }
                catch (Exception e)
                {
                    Modding.Logger.LogError($"[SocketCommunicator] Error sending state: {e.Message}");
                    DesktopLogger.LogError($"SendState error: {e.Message}");
                    CloseCurrentClient();
                }
            }
        }

        public string ReceiveAction()
        {
            if (!isConnected || stream == null)
                return "IDLE";

            lock (lockObject)
            {
                // Double-check inside lock
                if (!isConnected || stream == null || !stream.CanRead)
                    return "IDLE";

                try
                {
                    if (!stream.DataAvailable)
                        return "IDLE";

                    byte[] buffer = new byte[1024];
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    
                    if (bytesRead > 0)
                    {
                        string action = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                        return action.Trim();
                    }
                    else
                    {
                        // FIX: Zero bytes read = connection closed
                        Modding.Logger.LogWarn("[SocketCommunicator] Client disconnected (0 bytes read)");
                        DesktopLogger.LogWarning($"Client disconnected during ReceiveAction at {DateTime.Now:HH:mm:ss}");
                        CloseCurrentClient();
                        return "IDLE";
                    }
                }
                catch (System.IO.IOException ioEx)
                {
                    // FIX: Connection lost during read
                    Modding.Logger.LogWarn($"[SocketCommunicator] Client disconnected during ReceiveAction: {ioEx.Message}");
                    DesktopLogger.LogWarning($"Client disconnected during ReceiveAction at {DateTime.Now:HH:mm:ss}");
                    CloseCurrentClient();
                    return "IDLE";
                }
                catch (Exception e)
                {
                    Modding.Logger.LogError($"[SocketCommunicator] Error receiving action: {e.Message}");
                    DesktopLogger.LogError($"ReceiveAction error: {e.Message}");
                    CloseCurrentClient();
                    return "IDLE";
                }
            }
        }

        public void Close()
        {
            Modding.Logger.Log("[SocketCommunicator] Shutting down server...");
            DesktopLogger.Log("SocketCommunicator shutting down");

            // FIX: Signal server loop to stop
            isRunning = false;

            lock (lockObject)
            {
                CloseCurrentClient();
            }

            // FIX: Stop the TcpListener
            try
            {
                server?.Stop();
                server = null;
            }
            catch (Exception e)
            {
                Modding.Logger.LogWarn($"Error stopping server: {e.Message}");
            }

            // FIX: Wait for server thread to finish
            if (serverThread != null && serverThread.IsAlive)
            {
                if (!serverThread.Join(2000)) // Wait up to 2 seconds
                {
                    Modding.Logger.LogWarn("Server thread did not stop gracefully");
                }
            }

            Modding.Logger.Log("[SocketCommunicator] Server shut down complete");
            DesktopLogger.Log("Server shut down complete");
        }

        public bool IsConnected => isConnected;
    }
}