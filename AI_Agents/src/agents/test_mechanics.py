import socket
import time
import threading

HOST = '127.0.0.1'
PORT = 8888

def listen_and_drain(sock):
    """
    Legge e scarta i dati in arrivo dalla Mod per evitare 
    che il buffer di rete si riempia e faccia cadere la connessione.
    """
    try:
        while True:
            data = sock.recv(4096)
            if not data: break
            # I dati vengono letti e ignorati per tenere il canale libero
    except Exception:
        pass

def test_mechanics():
    print(f"Connessione a {HOST}:{PORT}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Opzione TCP_NODELAY per inviare i comandi istantaneamente
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((HOST, PORT))
            print("Connesso! Avvio thread di ascolto dati...")

            # --- IDRAULICO DI RETE ---
            # Avviamo il thread che pulisce il tubo di connessione in background
            listener = threading.Thread(target=listen_and_drain, args=(s,), daemon=True)
            listener.start()
            
            def send_action(action, duration_sec=0.1):
                """Invia un comando ripetutamente per una certa durata."""
                print(f" -> Invio azione: {action.strip()}")
                end_time = time.time() + duration_sec
                while time.time() < end_time:
                    try:
                        # Il newline \n è fondamentale per il protocollo della Mod
                        msg = f"{action}\n"
                        s.sendall(msg.encode('utf-8'))
                    except Exception as e:
                        print(f"Errore invio: {e}")
                        break
                    time.sleep(0.03) # 30ms tra un invio e l'altro

            # --- INIZIO SEQUENZA TEST ---

            print("\n>>> TEST 1: SALTO (JUMP)")
            send_action("JUMP", 0.3)
            time.sleep(1.0) 

            print(">>> TEST 2: ATTACCO FRONTALE (ATTACK)")
            send_action("ATTACK", 0.1)
            time.sleep(1.0)

            print(">>> TEST 3: ATTACCO IN ALTO (UP ATTACK)")
            # La mod combina "UP" (sguardo) e "ATTACK"
            send_action("UP ATTACK", 0.2)
            time.sleep(1.0)

            print(">>> TEST 4: POGO (SALTO + DOWN ATTACK)")
            # Saltiamo prima per poter colpire verso il basso
            send_action("JUMP", 0.3)
            time.sleep(0.05) 
            send_action("DOWN ATTACK", 0.2)
            time.sleep(1.0)

            print(">>> TEST 5: DASH")
            send_action("DASH", 0.1)
            time.sleep(1.0)

            print(">>> TEST 6: SPELL (CAST_NEUTRAL)")
            send_action("CAST_NEUTRAL", 0.1)
            time.sleep(1.5)

            print("\n>>> TEST 7: CURA (FOCUS)")
            print("!!! Il cavaliere dovrebbe fermarsi, caricare e curarsi (o sprecare anime) !!!")
            
            # Inviamo FOCUS per 2.5 secondi per garantire il tempo di caricamento
            start_focus = time.time()
            while time.time() - start_focus < 2.5:
                try:
                    s.sendall(b"FOCUS\n")
                except Exception as e:
                    print(f"Errore durante focus: {e}")
                    break
                time.sleep(0.03) 

            print(">>> FINE FOCUS. Rilascio tasti.")
            time.sleep(1.0)
            
            print("\nTest completato.")

    except ConnectionRefusedError:
        print("Errore: Impossibile connettersi. Assicurati che Hollow Knight sia aperto e la Mod caricata.")
    except Exception as e:
        print(f"Errore imprevisto: {e}")

if __name__ == "__main__":
    test_mechanics()