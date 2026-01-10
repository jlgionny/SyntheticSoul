import socket
import time
import threading

HOST = '127.0.0.1'
PORT = 8888

def listen_and_drain(sock):
    """Legge e scarta i dati in arrivo dalla Mod."""
    try:
        while True:
            data = sock.recv(4096)
            if not data: break
    except Exception:
        pass

def test_mechanics():
    print(f"Connessione a {HOST}:{PORT}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((HOST, PORT))
            print("Connesso! Avvio test meccaniche LITE...")

            # Avvio thread di ascolto
            listener = threading.Thread(target=listen_and_drain, args=(s,), daemon=True)
            listener.start()
            
            def send_action(action, duration_sec=0.1):
                print(f" -> Invio azione: {action.strip()}")
                end_time = time.time() + duration_sec
                while time.time() < end_time:
                    try:
                        msg = f"{action}\n"
                        s.sendall(msg.encode('utf-8'))
                    except Exception as e:
                        print(f"Errore invio: {e}")
                        break
                    time.sleep(0.03)

            # --- INIZIO SEQUENZA TEST ---

            print("\n>>> TEST 1: SALTO (JUMP)")
            send_action("JUMP", 0.3)
            time.sleep(1.0) 

            print(">>> TEST 2: ATTACCO FRONTALE (ATTACK)")
            send_action("ATTACK", 0.1)
            time.sleep(1.0)

            print(">>> TEST 3: ATTACCO IN ALTO (ATTACK_UP)")
            # La mod ora gestisce automaticamente lo sguardo verso l'alto
            send_action("ATTACK_UP", 0.2)
            time.sleep(1.0)

            print(">>> TEST 4: POGO (SALTO + ATTACK_DOWN)")
            # Saltiamo prima
            send_action("JUMP", 0.3)
            time.sleep(0.05) 
            # Poi attacco giù (la mod gestisce lo sguardo)
            send_action("ATTACK_DOWN", 0.2)
            time.sleep(1.0)

            print(">>> TEST 5: DASH")
            send_action("DASH", 0.1)
            time.sleep(1.0)

            print(">>> TEST 6: SPELL NEUTRA (CAST_NEUTRAL)")
            send_action("CAST_NEUTRAL", 0.1)
            time.sleep(1.5)

            # RIMOSSI: Test Focus, Spell Up, Spell Down

            print("\nTest completato (Versione Lite).")

    except ConnectionRefusedError:
        print("Errore: Impossibile connettersi a Hollow Knight.")
    except Exception as e:
        print(f"Errore imprevisto: {e}")

if __name__ == "__main__":
    test_mechanics()