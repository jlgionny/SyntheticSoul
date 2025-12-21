using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.IO;
using System.Threading;
using Modding;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Server TCP semplice.
    /// Ascolta sulla porta specificata e invia il JSON di stato a chi si connette.
    /// </summary>
    public class SocketServer
    {
        private TcpListener tcpListener;
        private int port;
        private bool isRunning = false;

        public SocketServer(int port)
        {
            this.port = port;
            // Ascolta solo su Localhost (127.0.0.1) per sicurezza
            tcpListener = new TcpListener(IPAddress.Loopback, port);
        }

        /// <summary>
        /// Avvia il loop di ascolto. Questo metodo è bloccante, va eseguito in un Thread separato.
        /// </summary>
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
                        // Controlla se c'è una connessione in attesa
                        if (tcpListener.Pending())
                        {
                            TcpClient client = tcpListener.AcceptTcpClient();
                            // Gestisce il client in un task del ThreadPool per non bloccare
                            ThreadPool.QueueUserWorkItem(_ => HandleClient(client, stateCapture));
                        }
                        // Piccola pausa per non consumare il 100% della CPU nel loop
                        Thread.Sleep(10);
                    }
                    catch { /* Gestione errori silenziosa nel loop */ }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[Synthetic Soul] SocketServer ❌ Errore Avvio: {e.Message}");
            }
        }

        private void HandleClient(TcpClient client, GameStateCapture stateCapture)
        {
            try
            {
                using (NetworkStream stream = client.GetStream())
                // new UTF8Encoding(false) DISABILITA il BOM (Byte Order Mark) che causa errori in Python
                using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
                {
                    // Ottieni il JSON aggiornato (Thread-Safe grazie al lock in GameStateCapture)
                    string gameStateJSON = stateCapture.GetStateAsJSON();
                    writer.WriteLine(gameStateJSON);
                }
            }
            catch { /* Il client si è disconnesso o c'è stato un errore di rete */ }
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