import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time

class HollowKnightEnv(gym.Env):
    """
    Ambiente Gymnasium per Hollow Knight.
    Gestisce connessione Socket, Action Mapping (12 azioni) e Reward Shaping.
    """
    
    def __init__(self, host='127.0.0.1', port=8888):
        super(HollowKnightEnv, self).__init__()
        
        # --- SPAZIO D'AZIONE (12 Azioni) ---
        self.action_space = spaces.Discrete(12)
        
        # Mappa interi -> stringhe per il C#
        self.action_map = {
            0: "IDLE", 
            1: "MOVE_LEFT", 
            2: "MOVE_RIGHT", 
            3: "JUMP", 
            4: "DASH", 
            5: "ATTACK",
            6: "ATTACK_UP",
            7: "ATTACK_DOWN",
            8: "CAST_NEUTRAL",
            9: "CAST_UP",
            10: "CAST_DOWN",
            11: "FOCUS"
        }

        # --- SPAZIO OSSERVAZIONE ---
        # 10 features normalizzate
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.host = host
        self.port = port
        self.sock = None
        self._connect()

        self.last_boss_hp = 0
        self.last_player_hp = 0
        self.last_soul = 0

    def _connect(self):
        print(f"Tentativo di connessione a {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((self.host, self.port))
                print("Connesso a Hollow Knight!")
                break
            except ConnectionRefusedError:
                time.sleep(2)

    def step(self, action):
        # 1. Invia Azione
        action_str = self.action_map[int(action)]
        try:
            self.sock.sendall((action_str + "\n").encode('utf-8'))
        except BrokenPipeError:
            self._connect()
            return self.step(action)

        # 2. Ricevi Stato
        json_data = self._receive_json()
        if json_data is None:
            return np.zeros(10, dtype=np.float32), 0, True, False, {}

        # 3. Processa Dati
        obs = self._process_observation(json_data)
        
        # 4. Calcola Reward
        reward = self._calculate_reward(json_data, action)

        # 5. Check fine episodio
        terminated = False
        if json_data['hp'] <= 0:
            terminated = True
            reward -= 10 # Morte Player
        
        if len(json_data['enemies']) > 0 and json_data['enemies'][0]['hp'] <= 0:
            terminated = True
            reward += 50 # Vittoria Boss

        # Aggiorna stati precedenti
        self.last_player_hp = json_data['hp']
        self.last_soul = json_data['soul']
        if len(json_data['enemies']) > 0:
            self.last_boss_hp = json_data['enemies'][0]['hp']

        return obs, reward, terminated, False, {}

    def _receive_json(self):
        buffer = ""
        while True:
            try:
                chunk = self.sock.recv(4096).decode('utf-8')
                if not chunk: return None
                buffer += chunk
                if "\n" in buffer:
                    message, _ = buffer.split('\n', 1)
                    return json.loads(message)
            except:
                return None

    def _process_observation(self, data):
        p_x, p_y = data.get('p_x', 0), data.get('p_y', 0)
        e_x, e_y, e_hp = p_x, p_y, 0 
        
        if data.get('enemies'):
            enemy = data['enemies'][0]
            e_x, e_y, e_hp = enemy['x'], enemy['y'], enemy['hp']

        # Normalizzazione
        obs = np.array([
            (e_x - p_x) / 20.0, # Dist X Relativa
            (e_y - p_y) / 10.0, # Dist Y Relativa
            data['hp'] / float(data['max_hp']),
            data['soul'] / 100.0,
            data['vel_x'] / 15.0,
            data['vel_y'] / 20.0,
            0.0, # Dash CD placeholder
            e_hp / 500.0,
            1.0 if data['ground'] else 0.0,
            1.0 if data['wall'] else 0.0
        ], dtype=np.float32)
        return obs

    def _calculate_reward(self, data, action):
        reward = 0.0
        
        # Penalità Danni subiti
        if data.get('hurt', False):
            reward -= 2.0 # Aumentato: evitare danni è prioritario

        # Bonus Danni inflitti
        current_boss_hp = data['enemies'][0]['hp'] if data['enemies'] else 0
        if current_boss_hp < self.last_boss_hp:
            diff = self.last_boss_hp - current_boss_hp
            reward += (diff * 0.1) 
            
            # Bonus Extra per Spell andato a segno (più danni in un colpo)
            if diff > 10: 
                reward += 1.0

        # Penalità Time-Wasting
        reward -= 0.005

        # Penalità Spreco Soul
        # Le azioni 8, 9, 10 sono CAST. Se provi a castare senza soul (33), sei stupido.
        if action in [8, 9, 10]:
            if self.last_soul < 33:
                reward -= 0.1 # Punizione leggera per insegnare la gestione risorse
        
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.sock.sendall("IDLE\n".encode('utf-8'))
            initial_json = self._receive_json()
        except:
            self._connect()
            return self.reset()
            
        if initial_json is None: return np.zeros(10, dtype=np.float32), {}

        self.last_player_hp = initial_json['hp']
        self.last_soul = initial_json['soul']
        self.last_boss_hp = initial_json['enemies'][0]['hp'] if initial_json['enemies'] else 100
        
        return self._process_observation(initial_json), {}

    def close(self):
        if self.sock: self.sock.close()