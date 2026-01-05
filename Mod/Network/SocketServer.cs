using System;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text;
using System.Threading;
using Modding;

namespace SyntheticSoulMod
{
    public class SocketServer
    {
        private TcpListener _listener;
        private bool _isRunning = false;
        
        // Callback per passare l'azione ricevuta al gioco
        public Action<string> ActionCallback;

        public SocketServer(int port)
        {
            // Ascolta solo su localhost per sicurezza e velocità
            _listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
        }

        public void Start(GameStateCapture capture)
        {
            try
            {
                _listener.Start();
                _isRunning = true;
                Modding.Logger.Log("[SyntheticSoul] Server avviato. In attesa di Python...");

                while (_isRunning)
                {
                    // 1. Accetta il client (Python) - Questo blocca finché Python non si connette
                    try 
                    {
                        using (TcpClient client = _listener.AcceptTcpClient())
                        using (NetworkStream stream = client.GetStream())
                        using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
                        using (StreamWriter writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true })
                        {
                            Modding.Logger.Log("[SyntheticSoul] Python Connesso!");

                            // 2. CICLO DI COMUNICAZIONE PERSISTENTE
                            // Continua a scambiare dati finché il client non si disconnette o crasha
                            while (client.Connected)
                            {
                                // A. Prepara e invia lo stato attuale (JSON)
                                string json = capture.GetJson();
                                writer.WriteLine(json);

                                // B. Aspetta la risposta (Azione)
                                // Questo blocca il thread finché Python non invia qualcosa
                                string response = reader.ReadLine();

                                // Se response è null, significa che Python ha chiuso la connessione
                                if (response == null) 
                                {
                                    Modding.Logger.Log("[SyntheticSoul] Python si è disconnesso.");
                                    break; 
                                }

                                // C. Esegui l'azione ricevuta
                                if (!string.IsNullOrEmpty(response) && ActionCallback != null)
                                {
                                    ActionCallback(response);
                                }
                            }
                        }
                    }
                    catch (IOException) 
                    {
                        // Succede se Python crasha o chiude forzatamente. Ignoriamo e torniamo in ascolto.
                        Modding.Logger.Log("[SyntheticSoul] Connessione persa/interrotta.");
                    }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log("[SyntheticSoul] Server Critical Error: " + e.Message);
            }
            finally
            {
                _listener.Stop();
            }
        }

        public void Stop()
        {
            _isRunning = false;
            _listener.Stop();
        }
    }
}