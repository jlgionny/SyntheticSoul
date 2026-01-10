import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
import math

class HollowKnightEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, host='127.0.0.1', port=8888):
        super(HollowKnightEnv, self).__init__()

        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        
        # 9 AZIONI
        self.action_space = spaces.Discrete(9)
        self.action_map = {
            0: "IDLE", 1: "MOVE_LEFT", 2: "MOVE_RIGHT", 3: "JUMP",
            4: "DASH", 5: "ATTACK", 6: "ATTACK_UP", 7: "ATTACK_DOWN",
            8: "CAST_NEUTRAL"
        }

        # OSSERVAZIONE ESTESA: 15 (Base) + 6 (2 Pericoli più vicini) = 21
        # Struttura Pericolo: [RelX, RelY, VX, VY] ma semplifichiamo a 3 valori per pericolo [RelX, RelY, Dist]
        # Oppure [RelX, RelY, VX, VY] per capire se sta arrivando addosso.
        # Facciamo 25 valori totali per stare larghi e includere velocità.
        # 15 Base + 5 (Danger 1) + 5 (Danger 2) = 25 float
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

    def connect(self):
        if self.connected: return
        print(f"[ENV] Connessione a {self.host}:{self.port}...")
        while not self.connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.host, self.port))
                self.connected = True
                print("[ENV] Connesso!")
            except:
                time.sleep(2)

    def _get_state_from_socket(self):
        if not self.connected: return None
        try:
            buffer = b""
            while b"\n" not in buffer:
                chunk = self.sock.recv(4096)
                if not chunk: raise ConnectionResetError
                buffer += chunk
            lines = buffer.decode('utf-8').strip().split('\n')
            return json.loads(lines[-1])
        except:
            self.connected = False
            return None

    def _process_observation(self, state_json):
        if state_json is None: return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Dati Eroe (Base)
        hx = state_json.get("x", 0.0)
        hy = state_json.get("y", 0.0)
        obs = [
            hx, hy,
            state_json.get("vx", 0.0), state_json.get("vy", 0.0),
            state_json.get("health", 0), state_json.get("maxHealth", 1),
            state_json.get("soul", 0),
            1.0 if state_json.get("onGround", False) else 0.0,
            1.0 if state_json.get("touchingWall", False) else 0.0,
            1.0 if state_json.get("damageTaken", False) else 0.0
        ]

        # Nemico più vicino
        enemies = state_json.get("enemies", [])
        if enemies:
            enemies.sort(key=lambda e: e.get("Distance", 9999))
            e = enemies[0]
            # Posizione Relativa è meglio per l'AI
            obs.extend([e.get("X", 0.0) - hx, e.get("Y", 0.0) - hy, e.get("Distance", 0.0), e.get("HP", 0), 1.0])
        else:
            obs.extend([0, 0, 0, 0, 0])

        # PERICOLI (Proiettili/Attacchi)
        dangers = state_json.get("dangers", [])
        # Ne prendiamo fino a 2
        dangers.sort(key=lambda d: d.get("Distance", 9999))
        
        for i in range(2):
            if i < len(dangers):
                d = dangers[i]
                # Salviamo posizione relativa (DX, DY) e velocità (VX, VY) e distanza
                obs.extend([
                    d.get("X", 0.0) - hx, 
                    d.get("Y", 0.0) - hy, 
                    d.get("VX", 0.0), 
                    d.get("VY", 0.0),
                    d.get("Distance", 0.0)
                ])
            else:
                # Padding se non ci sono pericoli
                obs.extend([0, 0, 0, 0, 0])

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, state_json):
        if state_json is None: return 0.0
        reward = 0.0
        
        # Punizione danno (Cruciale per imparare a schivare)
        if state_json.get("damageTaken", False):
            reward -= 20.0 # Aumentata la punizione
            
        reward += 0.05 # Sopravvivenza
        return reward

    def step(self, action_idx):
        if not self.connected: self.connect()
        try:
            cmd = self.action_map.get(action_idx, "IDLE")
            self.sock.sendall(f"{cmd}\n".encode('utf-8'))
        except: self.connected = False

        state_json = self._get_state_from_socket()
        obs = self._process_observation(state_json)
        reward = self._calculate_reward(state_json)
        
        terminated = False
        if state_json and state_json.get("health", 0) <= 0:
            terminated = True
            reward -= 100

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.connected: self.connect()
        try: self.sock.sendall(b"IDLE\n")
        except: self.connected = False
        
        state_json = self._get_state_from_socket()
        return self._process_observation(state_json), {}

    def close(self):
        if self.sock: self.sock.close()