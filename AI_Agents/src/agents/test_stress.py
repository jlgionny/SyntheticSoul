import socket
import time
import json
import random

HOST = '127.0.0.1'
PORT = 8888

ACTIONS = [
    "MOVE_LEFT", "MOVE_RIGHT", 
    "JUMP", "ATTACK", 
    "ATTACK_UP", "ATTACK_DOWN", # Pogo
    "DASH", "CAST_NEUTRAL"
]

def survival_test():
    print(f"--- AVVIO SURVIVAL TEST (FINO ALLA MORTE) ---")
    print(f"Connessione a {HOST}:{PORT}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((HOST, PORT))
            print(">> Connesso! COMBATTIMENTO INIZIATO!")
            print(">> Premi CTRL+C per interrompere manualmente.")

            start_time = time.time()
            alive = True

            while alive:
                try:
                    # 1. Scegli un'azione casuale (stile AI non addestrata)
                    action = random.choice(ACTIONS)
                    
                    # Logica base: se pogo, saltiamo prima
                    if action == "ATTACK_DOWN":
                        s.sendall(b"JUMP\n")
                        time.sleep(0.1)

                    # Invia comando
                    msg = f"{action}\n"
                    s.sendall(msg.encode('utf-8'))

                    # 2. Ricevi lo stato per controllare la vita
                    # Non usiamo un thread separato qui perché vogliamo leggere la vita ADESSO
                    data = s.recv(4096)
                    if not data:
                        print("Connessione persa.")
                        break

                    # Parsing del JSON (prendiamo l'ultima riga valida)
                    lines = data.decode('utf-8').strip().split('\n')
                    if lines:
                        last_line = lines[-1]
                        try:
                            state = json.loads(last_line)
                            hp = state.get("health", 0)
                            max_hp = state.get("maxHealth", 5)
                            
                            # Barra visiva in console
                            bar = "♥" * hp + "♡" * (max_hp - hp)
                            print(f"\rHP: [{bar}] Action: {action}      ", end="")

                            if hp <= 0:
                                print(f"\n\n>>> SEI MORTO! <<<")
                                print(f"Tempo sopravvissuto: {time.time() - start_time:.2f} secondi")
                                alive = False
                        except json.JSONDecodeError:
                            pass # Ignora pacchetti incompleti

                    # Piccolo delay per non floodare troppo
                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print("\nTest interrotto dall'utente.")
                    break
                except Exception as e:
                    print(f"\nErrore nel loop: {e}")
                    break

    except ConnectionRefusedError:
        print("ERRORE: Hollow Knight non risponde. Assicurati di essere in partita.")
    except Exception as e:
        print(f"ERRORE CRITICO: {e}")

if __name__ == "__main__":
    survival_test()