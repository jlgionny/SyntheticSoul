import socket
import json
import time
import numpy as np # Assicurati di avere numpy se serve per gli spazi, altrimenti ignora

class HollowKnightEnv:
    """
    Environment wrapper per Hollow Knight.
    Comunica con la mod C# via socket.
    """
    
    # Nomi azioni che corrispondono esattamente a ActionExecutor.cs
    ACTIONS = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "UP",
        3: "DOWN",
        4: "JUMP",
        5: "ATTACK",
        6: "DASH",
        7: "SPELL"
    }
    
    def __init__(self, host='localhost', port=5555, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            # Wrapper file per usare readline()
            self.socket_file = self.socket.makefile('r', encoding='utf-8')
            self.connected = True
            print(f"[Env] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Env] Connection failed: {e}")
            # Non raise qui, permettiamo al codice di riprovare se necessario o gestire l'errore dopo
    
    def _send_action(self, action_name):
        try:
            if not self.socket:
                return

            # === FIX CRITICO ===
            # NON inviare JSON. Il C# si aspetta solo la stringa comando + newline.
            # Errato: message = json.dumps({"action": action_name}) + "\n"
            
            message = f"{action_name}\n"
            self.socket.sendall(message.encode('utf-8'))
            
        except BrokenPipeError:
            print("[Env] Broken pipe sending action. Reconnecting...")
            self.connected = False
            self._connect()
        except Exception as e:
            print(f"[Env] Error sending action: {e}")
    
    def _receive_state(self):
        try:
            if not self.socket_file:
                return None

            line = self.socket_file.readline()
            if not line:
                # EOF
                self.connected = False
                return None
                
            line = line.strip()
            if not line:
                return None
                
            state = json.loads(line)
            return state
            
        except json.JSONDecodeError as e:
            print(f"[Env] JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"[Env] Error receiving state: {e}")
            return None
    
    def reset(self):
        # Invia IDLE invece di RESET perché ActionExecutor non ha un case "RESET"
        self._send_action("IDLE")
        
        # Aspetta il primo stato valido
        state = self._receive_state()
        
        # Se il primo stato è None (connessione lenta), riprova un paio di volte
        attempts = 0
        while state is None and attempts < 10:
            time.sleep(0.1)
            state = self._receive_state()
            attempts += 1
            
        # Se ancora None, restituisci un dizionario vuoto o default per evitare crash
        if state is None:
            return {} 
            
        return state
    
    def step(self, action):
        action_name = self.ACTIONS.get(action, "IDLE")
        self._send_action(action_name)
        
        state = self._receive_state()
        
        # Gestione caso disconnessione o errore
        if state is None:
            done = True
            state = {} # Stato vuoto di fallback
        else:
            done = state.get('isDead', False) or state.get('bossDefeated', False)
        
        info = {
            'action_name': action_name,
            'timestamp': state.get('timestamp', 0) if state else 0
        }
        return state, done, info
    
    def close(self):
        if self.socket_file:
            try:
                self.socket_file.close()
            except:
                pass
        if self.socket:
            try:
                self.socket.close()
                print("[Env] Connection closed")
            except:
                pass
        self.connected = False
        self.socket_file = None
        self.socket = None