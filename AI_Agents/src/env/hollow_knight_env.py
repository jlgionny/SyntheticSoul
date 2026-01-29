import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class HollowKnightEnv:
    """
    Environment wrapper aggiornato (Anti-Camping & Aggressive).
    """

    # DEFINIZIONE AZIONI (Indenta con 4 spazi)
    ACTIONS = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "UP",
        3: "DOWN",
        4: "JUMP",
        5: "ATTACK",
        6: "DASH",
        7: "SPELL",
        8: "IDLE",  # <--- Nuova azione aggiunta correttamente
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        use_reward_shaping: bool = False,
    ):
        """Initialize environment."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self.use_reward_shaping = use_reward_shaping

        # Tracking per calcolo differenze reward
        if use_reward_shaping:
            self.prev_boss_hp = None
            self.prev_player_hp = None
            self.prev_mantis_killed = 0

        self._connect()

    def _connect(self):
        """Establish TCP connection."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.socket_file = self.socket.makefile("r", encoding="utf-8")
            self.connected = True
            print(f"[Env] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Env] Connection failed: {e}")

    def _send_action(self, action_name: str):
        """Send action to C# mod."""
        try:
            if not self.socket:
                return
            message = f"{action_name}\n"
            self.socket.sendall(message.encode("utf-8"))
        except BrokenPipeError:
            print("[Env] Broken pipe. Reconnecting...")
            self.connected = False
            self._connect()
        except Exception as e:
            print(f"[Env] Error sending action: {e}")

    def _receive_state(self) -> Optional[Dict]:
        """Receive state JSON from C# mod."""
        try:
            if not self.socket_file:
                return None
            line = self.socket_file.readline()
            if not line:
                self.connected = False
                return None
            line = line.strip()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Env] JSON decode error: {e}")
            return None
        except socket.timeout:
            print(f"[Env] Socket timeout ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"[Env] Error receiving state: {e}")
            return None

    def _compute_reward(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        Calcola il reward con logica corretta per evitare il camping sui muri.
        """
        reward = 0.0
        info = {"damage_taken": 0}

        boss_hp = state.get("bossHealth", 100.0)
        player_hp = state.get("playerHealth", 5)
        mantis_killed = state.get("mantisLordsKilled", 0)

        # --- DATI POSIZIONALI ---
        terrain = state.get("terrainInfo", [1.0] * 5)
        space_ahead = terrain[2] if len(terrain) > 2 else 0.0
        space_behind = terrain[3] if len(terrain) > 3 else 0.0

        dist_to_boss = state.get("distanceToBoss", 20.0)
        is_facing_boss = state.get("isFacingBoss", False)
        hazards = state.get("nearbyHazards", [])

        # 1. ANALISI POSIZIONE
        is_touching_wall = space_ahead < 1.0 or space_behind < 1.0
        is_centered = space_ahead > 5.0 and space_behind > 5.0

        # 2. CALCOLO REWARD PROSSIMITÀ
        proximity_reward = 0.0

        if dist_to_boss < 5.0:
            proximity_reward = 0.2
            if is_facing_boss:
                proximity_reward += 0.1
        elif dist_to_boss < 10.0:
            proximity_reward = 0.1
            if is_facing_boss:
                proximity_reward += 0.05
        elif dist_to_boss < 20.0:
            proximity_reward = 0.02
        else:
            proximity_reward = -0.1

        # 3. PENALITÀ AMBIENTALI
        if is_touching_wall:
            reward -= 0.5
            if proximity_reward > 0:
                proximity_reward = 0.0

        reward += proximity_reward

        if is_centered:
            reward += 0.05

        if hazards:
            min_hazard_dist = 100.0
            for h in hazards:
                d = h.get("distance", 100.0)
                if d < min_hazard_dist:
                    min_hazard_dist = d
            if min_hazard_dist < 1.0:
                reward -= 0.5

        # 4. COMBATTIMENTO
        if self.prev_boss_hp is not None:
            boss_damage = self.prev_boss_hp - boss_hp
            if boss_damage > 0:
                reward += 2.0

        if self.prev_player_hp is not None:
            player_damage = self.prev_player_hp - player_hp
            if player_damage > 0:
                reward -= 0.5
                info["damage_taken"] = player_damage

        if mantis_killed > self.prev_mantis_killed:
            reward += 20.0
            self.prev_mantis_killed = mantis_killed

        if done:
            if state.get("isDead", False):
                reward -= 5.0
            elif state.get("bossDefeated", False) or mantis_killed == 3:
                reward += 100.0

        self.prev_boss_hp = boss_hp
        self.prev_player_hp = player_hp
        self.prev_mantis_killed = mantis_killed

        reward = max(-5.0, min(5.0, reward))

        return reward, info

    def reset(self) -> Dict:
        """Reset environment."""
        print("[Env] Reset - waiting for new episode...")
        self._send_action("IDLE")

        state = self._receive_state()
        attempts = 0
        max_attempts = 20

        while state is None and attempts < max_attempts:
            time.sleep(0.5)
            state = self._receive_state()
            attempts += 1

        if state is None:
            print("[Env] WARNING: No state after reset")
            return {}

        print("[Env] ✓ Reset complete")

        if self.use_reward_shaping:
            self.prev_boss_hp = state.get("bossHealth", 100.0)
            self.prev_player_hp = state.get("playerHealth", 5)
            self.prev_mantis_killed = 0

        return state

    def step(self, action: int) -> Tuple:
        """Execute action."""
        action_name = self.ACTIONS.get(action, "IDLE")
        self._send_action(action_name)
        state = self._receive_state()

        if state is None:
            done = True
            state = {}
            if self.use_reward_shaping:
                return state, -10.0, done, {"error": "Connection lost"}
            return state, done, {"error": "Connection lost"}

        done = state.get("isDead", False) or state.get("bossDefeated", False)
        info = {"action_name": action_name}

        if self.use_reward_shaping:
            reward, reward_info = self._compute_reward(state, done)
            info.update(reward_info)
            return state, reward, done, info
        else:
            return state, done, info

    def close(self):
        """Close connection."""
        if self.socket_file:
            try:
                self.socket_file.close()
            except Exception:
                pass
        if self.socket:
            try:
                self.socket.close()
                print("[Env] Connection closed")
            except Exception:
                pass
        self.connected = False
