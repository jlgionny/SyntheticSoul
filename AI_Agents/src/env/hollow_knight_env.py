import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class HollowKnightEnv:
    """
    Environment wrapper aggiornato (Anti-Camping & Aggressive).
    """

    # DEFINIZIONE AZIONI BASE
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
        Reward shaping per combat contro Mantis Lords.

        Filosofia del trade:
        - Knight ha 9 HP, Boss ha ~300+ HP totali (3 Mantis Lords)
        - Ogni colpo del Knight fa ~13 danni, quindi servono ~23 colpi per vincere
        - Il Knight può subire max 9 colpi (1 danno ciascuno) prima di morire
        - Quindi il ratio ideale è: 23 colpi dati / 9 colpi subiti = ~2.5:1
        - Il reward deve incentivare questo trade favorevole

        Danno da spuntoni:
        - Gli spuntoni fanno 2 danni (doppio!)
        - NON danno soul (non ricaricano heal)
        - Sono completamente inutili e vanno evitati a tutti i costi
        """
        reward = 0.0
        info = {"damage_taken": 0, "spike_damage": False, "damage_source": "none"}

        boss_hp = state.get("bossHealth", 100.0)
        player_hp = state.get("playerHealth", 9)
        mantis_killed = state.get("mantisLordsKilled", 0)

        dist_to_boss = state.get("distanceToBoss", 20.0)
        is_facing_boss = state.get("isFacingBoss", False)

        # ============================================
        # 1. REWARD PER DANNO AL BOSS
        # ============================================
        # Ogni colpo al boss è prezioso - il Knight deve colpire ~23 volte
        if self.prev_boss_hp is not None:
            boss_damage = self.prev_boss_hp - boss_hp
            if boss_damage > 0:
                # Reward proporzionale: se il boss ha ~300 HP e noi 9,
                # ogni nostro colpo vale ~3x di più in termini di "progresso"
                reward += 2.5

        # ============================================
        # 2. REWARD PER DISTANZA OTTIMALE DAL BOSS
        # ============================================
        # Range nail in Hollow Knight: ~2.5 unità
        # Distanza ideale: 2.5-4 unità (può colpire, può schivare)
        # I Mantis Lords hanno attacchi a media distanza (dash, boomerang)

        proximity_reward = 0.0

        if dist_to_boss < 2.5:
            # RANGE DI ATTACCO PERFETTO - può colpire!
            # Ma è anche pericoloso, quindi reward moderato
            proximity_reward = 0.15
            if is_facing_boss:
                proximity_reward += 0.1  # Bonus se guarda il boss
        elif dist_to_boss < 4.5:
            # RANGE OTTIMALE - può colpire e ha tempo di reagire
            proximity_reward = 0.2
            if is_facing_boss:
                proximity_reward += 0.1
        elif dist_to_boss < 8.0:
            # Range medio - deve avvicinarsi
            proximity_reward = 0.05
            if is_facing_boss:
                proximity_reward += 0.03
        elif dist_to_boss < 15.0:
            # Lontano - piccola penalità
            proximity_reward = -0.05
        else:
            # Troppo lontano - sta scappando
            proximity_reward = -0.15

        reward += proximity_reward

        # ============================================
        # 3. PENALITÀ PER DANNO SUBITO
        # ============================================
        # Il Knight ha solo 9 HP - ogni danno conta molto!
        # Ma il danno da boss/proiettili è "accettabile" se stiamo facendo danni
        # Il danno da spuntoni è SEMPRE negativo (niente soul, niente progresso)

        if self.prev_player_hp is not None:
            player_damage = self.prev_player_hp - player_hp
            if player_damage > 0:
                info["damage_taken"] = player_damage

                # Determina la fonte del danno
                # In Hollow Knight, gli spuntoni fanno 2 danni, il boss fa 1
                if player_damage >= 2:
                    # Danno da SPUNTONI (2 danni) - MOLTO GRAVE
                    # - Non dà soul
                    # - È evitabile al 100%
                    # - È un errore di posizionamento
                    info["spike_damage"] = True
                    info["damage_source"] = "spikes"
                    # Penalità MOLTO alta: 2 HP persi = 22% della vita
                    # + è danno "stupido" che non dà nulla in cambio
                    reward -= 3.0
                else:
                    # Danno da BOSS/PROIETTILI (1 danno)
                    # - Dà soul (può healare dopo)
                    # - Fa parte del combat normale
                    # - Accettabile se stiamo facendo danni
                    info["damage_source"] = "boss"
                    # Penalità moderata: 1 HP = 11% della vita
                    # Ma se stiamo facendo trade favorevoli, è ok
                    reward -= 0.8

        # ============================================
        # 4. MANTIS LORDS KILLED
        # ============================================
        # Uccidere un Mantis Lord è un grande traguardo
        if mantis_killed > self.prev_mantis_killed:
            # Bonus scalato: più ne uccidi, più difficile diventa
            # (nella fase 2 ci sono 2 Mantis contemporaneamente)
            kill_bonus = 25.0 + (mantis_killed * 5.0)
            reward += kill_bonus
            self.prev_mantis_killed = mantis_killed

        # ============================================
        # 5. EPISODIO TERMINATO
        # ============================================
        if done:
            if state.get("isDead", False):
                # Morte - penalità scalata in base ai progressi
                # Se ha ucciso 2 Mantis ed è morto, non è così grave
                death_penalty = -8.0 + (mantis_killed * 2.0)
                reward += death_penalty
            elif state.get("bossDefeated", False) or mantis_killed == 3:
                # VITTORIA!
                # Bonus extra se ha vinto con tanta vita
                hp_bonus = player_hp * 2.0  # Fino a +18 se vita piena
                reward += 100.0 + hp_bonus

        # Update tracking
        self.prev_boss_hp = boss_hp
        self.prev_player_hp = player_hp

        # Clip reward per stabilità training (range più ampio per eventi importanti)
        reward = max(-10.0, min(10.0, reward))

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
