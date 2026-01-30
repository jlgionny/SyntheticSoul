import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class HollowKnightEnv:
    """
    Environment wrapper per Hollow Knight (Mantis Lords).
    Gestisce la connessione TCP e il calcolo dei Reward basato sui dati della Mod.

    NOTA: Richiede la Mod aggiornata che invia 'lastHazardType'.
    """

    # Mappatura Azioni (Non modificare se non cambi anche l'Agent)
    ACTIONS = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "UP",
        3: "DOWN",
        4: "JUMP",
        5: "ATTACK",
        6: "DASH",
        7: "SPELL",
        8: "IDLE",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        use_reward_shaping: bool = False,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self.use_reward_shaping = use_reward_shaping

        # Variabili per calcolare le differenze (Delta) tra uno step e l'altro
        if use_reward_shaping:
            self.prev_boss_hp = None
            self.prev_player_hp = None
            self.prev_mantis_killed = 0

        # Avvia connessione
        self._connect()

    def _connect(self):
        """Stabilisce la connessione TCP con la Mod di Unity."""
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
        """Invia la stringa dell'azione al socket."""
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
            self.connected = False

    def _receive_state(self) -> Optional[Dict]:
        """Legge una riga dal socket e la converte da JSON a Dict."""
        try:
            if not self.socket_file:
                return None
            line = self.socket_file.readline()
            if not line:
                self.connected = False
                return None
            return json.loads(line.strip())
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
        Calcola il reward basato sullo stato corrente.
        Usa 'lastHazardType' dalla Mod per distinguere Spuntoni vs Boss.
        """
        reward = 0.0
        info = {"damage_taken": 0, "spike_damage": False, "damage_source": "none"}

        # --- ESTRAZIONE DATI ---
        boss_hp = state.get("bossHealth", 100.0)
        player_hp = state.get("playerHealth", 9)
        mantis_killed = state.get("mantisLordsKilled", 0)
        dist_to_boss = state.get("distanceToBoss", 20.0)
        is_facing_boss = state.get("isFacingBoss", False)

        # Dati specifici per il danno (Richiede Mod C# Aggiornata)
        damage_taken_in_step = state.get("damageTaken", 0)
        hazard_type = state.get("lastHazardType", 0)  # 0=None, 1=Enemy, 2=Env/Spike

        # --- 1. ATTACCO (Hit Boss) ---
        # Premia se abbiamo tolto vita al boss rispetto allo step precedente
        if self.prev_boss_hp is not None:
            boss_damage = self.prev_boss_hp - boss_hp
            if boss_damage > 0:
                reward += 1.0  # +1.0 Base Reward per colpo

        # --- 2. POSIZIONAMENTO (Spacing & Facing) ---
        proximity_reward = 0.0

        # Bonus costante se guardiamo il boss (fondamentale per colpire)
        if is_facing_boss:
            proximity_reward += 0.02

        # Gestione Zone di Distanza
        if dist_to_boss < 1.5:
            # TROPPO VICINO (Crowding) -> Rischio collisione
            proximity_reward -= 0.1
        elif 1.5 <= dist_to_boss <= 3.5:
            # SWEET SPOT (Range Aculeo) -> Ottimo
            proximity_reward += 0.1
            if is_facing_boss:
                proximity_reward += 0.05  # Bonus extra se pronto a colpire
        elif dist_to_boss > 15.0:
            # CAMPING (Troppo lontano) -> Inutile
            proximity_reward -= 0.1

        reward += proximity_reward

        # --- 3. GESTIONE DANNI (Fonte Verificata da Unity) ---
        # Se la mod ci dice che abbiamo preso danno in questo frame
        if damage_taken_in_step > 0:
            info["damage_taken"] = damage_taken_in_step

            # HAZARD TYPE 2 = Danno Ambientale (Spuntoni, Acido, Lave)
            if hazard_type == 2:
                info["spike_damage"] = True
                info["damage_source"] = "spikes"
                reward -= 3.5  # Penalità Severa (Morte istantanea logica)

            # HAZARD TYPE 1 (o altro) = Nemico, Proiettile, Boss
            else:
                info["damage_source"] = "boss"
                reward -= 2.0  # Penalità Standard (Combattimento)

        # --- 4. OBIETTIVI (Kill & Vittoria) ---
        if mantis_killed > self.prev_mantis_killed:
            reward += 10.0  # Bonus per aver ucciso una delle mantidi
            self.prev_mantis_killed = mantis_killed

        if done:
            if state.get("isDead", False):
                reward -= 5.0  # Penalità finale morte
            elif state.get("bossDefeated", False) or mantis_killed == 3:
                reward += 50.0  # Grande bonus vittoria

        # Aggiornamento stato precedente
        self.prev_boss_hp = boss_hp
        self.prev_player_hp = player_hp

        # Clipping finale per stabilità numerica (-5.0 a +5.0 esclude solo i reward vittoria estremi)
        reward = max(-5.0, min(5.0, reward))

        return reward, info

    def reset(self) -> Dict:
        """Resetta l'ambiente e attende un nuovo stato valido."""
        print("[Env] Reset - waiting for new episode...")

        # Invia IDLE per sbloccare eventuali socket appesi
        self._send_action("IDLE")

        state = self._receive_state()
        attempts = 0

        # Riprova finché non ottiene uno stato valido
        while state is None and attempts < 20:
            time.sleep(0.5)
            state = self._receive_state()
            attempts += 1

        if state is None:
            print("[Env] WARNING: No state after reset")
            return {}

        print("[Env] ✓ Reset complete")

        # Reset variabili reward shaping
        if self.use_reward_shaping:
            self.prev_boss_hp = state.get("bossHealth", 100.0)
            self.prev_player_hp = state.get("playerHealth", 5)
            self.prev_mantis_killed = 0

        return state

    def step(self, action: int) -> Tuple:
        """
        Esegue un'azione e restituisce (stato, reward, done, info).
        """
        action_name = self.ACTIONS.get(action, "IDLE")
        self._send_action(action_name)

        state = self._receive_state()

        # Gestione disconnessione o errore
        if state is None:
            return {}, 0, True, {"error": "Connection lost"}

        done = state.get("isDead", False) or state.get("bossDefeated", False)
        info = {"action_name": action_name}

        if self.use_reward_shaping:
            reward, reward_info = self._compute_reward(state, done)
            info.update(reward_info)
            return state, reward, done, info
        else:
            # Modalità senza reward shaping (solo test puro)
            return state, 0.0, done, info

    def close(self):
        """Chiude le risorse di rete."""
        if self.socket_file:
            try:
                self.socket_file.close()
            except Exception:
                pass
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.connected = False
