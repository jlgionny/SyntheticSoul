import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
import math

class HollowKnightEnv(gym.Env):
    """
    Ambiente Custom Gym per Hollow Knight.
    Si connette alla Mod 'SyntheticSoul' via TCP (localhost:8888).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, host='127.0.0.1', port=8888):
        super(HollowKnightEnv, self).__init__()

        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        
        # --- SPAZIO D'AZIONE (9 AZIONI) ---
        # Abbiamo rimosso FOCUS, CAST_UP e CAST_DOWN come richiesto.
        self.action_space = spaces.Discrete(9)
        
        # Mapping: Indice -> Comando stringa per la Mod
        self.action_map = {
            0: "IDLE",
            1: "MOVE_LEFT",
            2: "MOVE_RIGHT",
            3: "JUMP",
            4: "DASH",
            5: "ATTACK",       # Attacco frontale
            6: "ATTACK_UP",    # Attacco in alto (La mod gestisce lo sguardo)
            7: "ATTACK_DOWN",  # Pogo (La mod gestisce lo sguardo)
            8: "CAST_NEUTRAL"  # Vengeful Spirit (La mod gestisce la direzione)
        }

        # --- SPAZIO DI OSSERVAZIONE ---
        # Definiamo cosa vede l'AI. I valori devono essere normalizzati o grezzi.
        # Esempio vettore (dimensione 12 + 5 per il nemico più vicino = 17):
        # [HeroX, HeroY, HeroVX, HeroVY, Health, MaxHealth, Soul, OnGround, OnWall, DmgTaken, CanDash, CanJump, 
        #  EnemyX, EnemyY, EnemyDist, EnemyHP, EnemyType(OneHot? per ora 0/1)]
        
        # Per semplicità usiamo un box generico. La dimensione dipende da quanti dati estraiamo.
        # Qui ipotizzo un vettore di 15 float.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # Stato interno
        self.current_state = None
        self.last_health = -1
        self.total_reward = 0

    def connect(self):
        """Stabilisce la connessione con la Mod di Hollow Knight."""
        if self.connected:
            return
        
        print(f"[ENV] Connessione a {self.host}:{self.port}...")
        while not self.connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.host, self.port))
                self.connected = True
                print("[ENV] Connesso!")
            except ConnectionRefusedError:
                print("[ENV] In attesa del gioco... (Riprovo in 2s)")
                time.sleep(2)
            except Exception as e:
                print(f"[ENV] Errore connessione: {e}")
                time.sleep(2)

    def _get_state_from_socket(self):
        """Legge il JSON inviato dalla Mod e lo converte in dizionario."""
        if not self.connected:
            return None
        
        try:
            # Legge finché non trova il carattere newline (protocollo della Mod)
            buffer = b""
            while b"\n" not in buffer:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionResetError("Connessione chiusa dal server.")
                buffer += chunk
            
            # Decodifica l'ultima riga completa (potrebbero essercene più di una nel buffer)
            lines = buffer.decode('utf-8').strip().split('\n')
            last_line = lines[-1] # Prendiamo lo stato più recente
            
            return json.loads(last_line)
        except Exception as e:
            print(f"[ENV] Errore lettura socket: {e}")
            self.connected = False
            self.sock.close()
            return None

    def _process_observation(self, state_json):
        """Converte il JSON grezzo in un vettore numpy per la Rete Neurale."""
        if state_json is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Dati Eroe
        x = state_json.get("x", 0.0)
        y = state_json.get("y", 0.0)
        vx = state_json.get("vx", 0.0)
        vy = state_json.get("vy", 0.0)
        hp = state_json.get("health", 0)
        max_hp = state_json.get("maxHealth", 1)
        soul = state_json.get("soul", 0)
        ground = 1.0 if state_json.get("onGround", False) else 0.0
        wall = 1.0 if state_json.get("touchingWall", False) else 0.0
        dmg = 1.0 if state_json.get("damageTaken", False) else 0.0

        # Dati Nemico (Prendiamo solo il più vicino per semplificare l'input all'AI)
        enemies = state_json.get("enemies", [])
        ex, ey, edist, ehp = 0.0, 0.0, 0.0, 0.0
        
        if enemies:
            # Ordina per distanza e prendi il primo
            enemies.sort(key=lambda e: e.get("Distance", 9999))
            nearest = enemies[0]
            ex = nearest.get("X", 0.0)
            ey = nearest.get("Y", 0.0)
            edist = nearest.get("Distance", 0.0)
            ehp = nearest.get("HP", 0)
        
        # Normalizzazione basilare (opzionale ma consigliata per le Reti Neurali)
        # Qui lasciamo i valori grezzi per chiarezza, la rete imparerà comunque.
        # Costruiamo il vettore di osservazione (dimensione 15)
        obs = np.array([
            x, y, vx, vy, 
            hp, max_hp, soul, 
            ground, wall, dmg,
            ex, ey, edist, ehp,
            1.0 if enemies else 0.0 # Flag: c'è un nemico?
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self, state_json):
        """Logica di ricompensa semplificata."""
        if state_json is None:
            return 0.0
        
        reward = 0.0
        
        # 1. Penalità Danno (-10 se vieni colpito)
        if state_json.get("damageTaken", False):
            reward -= 10.0
            
        # 2. Ricompensa Sopravvivenza (piccolo premio per ogni frame vivo)
        reward += 0.01 
        
        # 3. Ricompensa per avvicinamento al nemico (shaping)
        # (Qui servirebbe logica complessa per confrontare la distanza precedente, 
        #  per ora lasciamo semplice per evitare bug)
        
        # 4. Ricompensa per Anima guadagnata (significa che abbiamo colpito un nemico)
        # Nota: Questo è un modo indiretto per sapere se abbiamo colpito, dato che colpire dà anima.
        # Bisognerebbe tracciare l'anima precedente.
        
        return reward

    def step(self, action_idx):
        # 1. Invia Azione
        if not self.connected:
            self.connect()
            
        command = self.action_map.get(action_idx, "IDLE")
        try:
            msg = f"{command}\n".encode('utf-8')
            self.sock.sendall(msg)
        except Exception:
            self.connected = False

        # 2. Ricevi Nuovo Stato
        state_json = self._get_state_from_socket()
        
        # 3. Elabora Osservazione
        observation = self._process_observation(state_json)
        
        # 4. Calcola Reward
        reward = self._calculate_reward(state_json)
        
        # 5. Check Terminazione (Morte)
        terminated = False
        if state_json:
            current_hp = state_json.get("health", 0)
            if current_hp <= 0:
                terminated = True
                reward -= 50 # Penalità grossa per la morte

        truncated = False # Usato se il tempo scade (opzionale)
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Gestione connessione
        if not self.connected:
            self.connect()
        
        # Invia comando IDLE per sicurezza
        try:
            self.sock.sendall(b"IDLE\n")
            # Attendi un attimo per stabilizzare
            time.sleep(0.1)
        except:
            self.connected = False
            self.connect()

        # Ottieni stato iniziale
        state_json = self._get_state_from_socket()
        observation = self._process_observation(state_json)
        
        return observation, {}

    def render(self):
        pass

    def close(self):
        if self.sock:
            self.sock.close()
        self.connected = False