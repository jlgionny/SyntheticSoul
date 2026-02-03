import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class HollowKnightEnv:
    """
    Environment wrapper per Hollow Knight - FASE 1: SURVIVAL.
    Include Reward Scaling per stabilità numerica.
    """

    # Mappatura Azioni (8 azioni)
    ACTIONS = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "UP",
        3: "DOWN",
        4: "JUMP",
        5: "ATTACK",
        6: "DASH",
        7: "SPELL",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        use_reward_shaping: bool = True,
        # NUOVO: Fattore di scala. I reward vengono divisi per questo numero.
        # Esempio: +50 diventa +5.0. Aiuta la rete neurale a convergere.
        reward_scale: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self.use_reward_shaping = use_reward_shaping
        self.reward_scale = reward_scale

        # Variabili per delta
        self.prev_boss_hp = None
        self.prev_mantis_killed = 0

        print("[Env] Initialized - PHASE 1: SURVIVAL MODE")
        print(f"      Focus: Dodge > Attack. Reward Scale: 1/{self.reward_scale}")

        self._connect()

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.socket_file = self.socket.makefile("r", encoding="utf-8")
            self.connected = True
            print(f"[Env] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Env] Connection failed: {e}")
            self.connected = False

    def _send_action(self, action_name: str):
        try:
            if not self.socket:
                return
            self.socket.sendall(f"{action_name}\n".encode("utf-8"))
        except Exception:
            self.connected = False
            self._connect()

    def _receive_state(self) -> Optional[Dict]:
        try:
            if not self.socket_file:
                return None
            line = self.socket_file.readline()
            if not line:
                return None
            return json.loads(line.strip())
        except Exception:
            return None

    def _compute_reward(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        Calcolo Reward FASE 1 (Survival).
        """
        reward = 0.0
        info = {}

        # --- ESTRAZIONE DATI ---
        boss_hp = state.get("bossHealth", 100.0)
        dist_to_boss = state.get("distanceToBoss", 20.0)
        damage_taken = state.get("damageTaken", 0)
        hazard_type = state.get("lastHazardType", 0)  # 1=Nemico, 2=Spuntoni
        mantis_killed = state.get("mantisLordsKilled", 0)

        # 1. LIVING REWARD (Incentivo a esistere)
        # +0.05 per frame -> +150 punti su 3000 step
        reward += 0.05

        # 2. DANNO SUBITO (Punizione Severa)
        if damage_taken > 0:
            if hazard_type == 2:  # Spuntoni/Env
                reward -= 5.0
                info["damage_source"] = "env"
            else:
                reward -= 3.0
                info["damage_source"] = "boss"

        # 3. POSIZIONAMENTO
        if dist_to_boss < 2.0:
            reward -= 0.05  # Troppo vicino
        elif 4.0 <= dist_to_boss <= 12.0:
            reward += 0.02  # Distanza sicura
        elif dist_to_boss > 20.0:
            reward -= 0.05  # Camping

        # 4. ATTACCO (Simbolico per ora)
        if self.prev_boss_hp is not None:
            if (self.prev_boss_hp - boss_hp) > 0:
                reward += 1.0

        # 5. FINE EPISODIO
        if done:
            if state.get("isDead", False):
                reward -= 5.0
            elif state.get("bossDefeated", False) or mantis_killed == 3:
                reward += 50.0

        # Aggiornamento storico
        self.prev_boss_hp = boss_hp

        if mantis_killed > self.prev_mantis_killed:
            reward += 10.0
            self.prev_mantis_killed = mantis_killed

        return reward, info

    def reset(self) -> Dict:
        state = self._receive_state()
        attempts = 0
        while state is None and attempts < 20:
            time.sleep(0.5)
            state = self._receive_state()
            attempts += 1

        if state:
            self.prev_boss_hp = state.get("bossHealth", 100.0)
            self.prev_mantis_killed = 0

        return state if state else {}

    def step(self, action: int) -> Tuple:
        action_name = self.ACTIONS.get(action, "MOVE_LEFT")
        self._send_action(action_name)
        state = self._receive_state()

        if state is None:
            return {}, 0, True, {"error": "Connection lost"}

        done = state.get("isDead", False) or state.get("bossDefeated", False)

        if self.use_reward_shaping:
            raw_reward, info = self._compute_reward(state, done)

            # --- APPLICAZIONE SCALING ---
            # Scaliamo il reward per renderlo più digeribile alla rete
            # Manteniamo però il segno e le proporzioni
            scaled_reward = raw_reward / self.reward_scale

            info["action_name"] = action_name
            info["raw_reward"] = raw_reward  # Per debug

            return state, scaled_reward, done, info
        else:
            return state, 0.0, done, {}

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.connected = False
