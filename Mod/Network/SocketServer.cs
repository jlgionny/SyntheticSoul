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
    /// Server TCP bidirezionale FIXED.
    /// 
    /// FLUSSO CORRETTO:
    /// 1. Client si connette
    /// 2. Server invia JSON (stato di gioco)
    /// 3. Server ASPETTA azione dal client (ReadLine)
    /// 4. Client invia azione (es: "JUMP\n")
    /// 5. Server esegue azione
    /// 6. Connessione chiude
    /// 7. Ripete da capo
    /// </summary>
    public class SocketServer
    {
        private TcpListener tcpListener;
        private int port;
        private bool isRunning = false;
        
        // Callback per notificare quando un'azione è ricevuta
        public delegate void OnActionReceived(InputHandler.AIAction action);
        public OnActionReceived ActionCallback { get; set; }

        public SocketServer(int port)
        {
            this.port = port;
            tcpListener = new TcpListener(IPAddress.Loopback, port);
        }

        /// <summary>
        /// Avvia il loop di ascolto. Eseguito in un Thread separato.
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
                            // Gestisce il client in un task del ThreadPool
                            ThreadPool.QueueUserWorkItem(_ => HandleClient(client, stateCapture));
                        }

                        // Piccola pausa per non consumare il 100% della CPU
                        Thread.Sleep(10);
                    }
                    catch { /* Errori nel loop */ }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[Synthetic Soul] SocketServer ❌ Errore Avvio: {e.Message}");
            }
        }

        /// <summary>
        /// Gestisce un client singolo.
        /// 
        /// SEQUENZA:
        /// 1. Client si connette
        /// 2. Invia lo stato (JSON)
        /// 3. ASPETTA l'azione dal client (ReadLine è BLOCCANTE)
        /// 4. Riceve azione e la esegue
        /// 5. Chiude connessione
        /// </summary>
        private void HandleClient(TcpClient client, GameStateCapture stateCapture)
        {
            try
            {
                using (NetworkStream stream = client.GetStream())
                {
                    // Timeout per evitare che il server resti bloccato
                    stream.ReadTimeout = 5000;  // 5 secondi
                    stream.WriteTimeout = 5000; // 5 secondi

                    using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
                    using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
                    {
                        // === FASE 1: INVIA STATO ATTUALE ===
                        string gameStateJSON = stateCapture.GetStateAsJSON();
                        writer.WriteLine(gameStateJSON);
                        Modding.Logger.Log($"[SocketServer] 📤 Stato inviato al client");

                        // === FASE 2: RICEVI AZIONE DALL'IA (ASPETTA QUI) ===
                        string actionString = reader.ReadLine();
                        
                        if (!string.IsNullOrEmpty(actionString))
                        {
                            // Parsing azione
                            InputHandler.AIAction action = InputHandler.ParseAction(actionString);
                            Modding.Logger.Log($"[SocketServer] 📥 Azione ricevuta: {action}");

                            // Invoca il callback per eseguire l'azione nel gioco
                            ActionCallback?.Invoke(action);
                        }
                        else
                        {
                            Modding.Logger.Log($"[SocketServer] ⚠️ Azione vuota ricevuta");
                        }
                    }
                }
            }
            catch (IOException ioEx)
            {
                Modding.Logger.Log($"[SocketServer] ⚠️ Errore I/O HandleClient: {ioEx.Message}");
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[SocketServer] ❌ Errore HandleClient: {e.Message}");
            }
            finally
            {
                try
                {
                    client.Close();
                }
                catch { }
            }
        }

        public void Stop()
        {
            isRunning = false;
            try { tcpListener?.Stop(); } catch { }
        }
    }
}