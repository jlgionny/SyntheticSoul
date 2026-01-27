"""
Hollow Knight Environment - Universal wrapper for DQN and PPO agents.
Communicates with C# mod via TCP socket.
"""

import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class HollowKnightEnv:
    """Environment wrapper with optional reward shaping for PPO."""

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
        use_reward_shaping: bool = False,
    ):
        """
        Initialize environment.

        Args:
            host: Server hostname
            port: Server port
            timeout: Socket timeout in seconds
            use_reward_shaping: If True, compute shaped rewards (for PPO).
                               If False, return raw state dict (for DQN).
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self.use_reward_shaping = use_reward_shaping

        # Reward shaping state tracking
        if use_reward_shaping:
            self._init_reward_tracking()

        self._connect()

    def _init_reward_tracking(self):
        """Initialize reward shaping variables."""
        self.prev_state = None
        self.prev_boss_health = 100.0
        self.prev_player_health = 5
        self.prev_distance = 100.0
        self.wall_stuck_counter = 0
        self.idle_counter = 0
        self.steps_far = 0
        self.total_damage_dealt = 0
        self.total_boss_hits = 0

    def _connect(self):
        """Establish TCP connection to C# mod."""
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
        """Send action command to C# mod."""
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

    def _compute_reward(
        self, state: Dict, action_name: str, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute shaped reward for PPO training.
        Optimized for Mantis Lords boss fight.
        """
        reward = 0.0
        info = {}

        # Extract values
        player_hp = state.get("playerHealth", 5)
        boss_hp = state.get("bossHealth", 100.0)
        distance = state.get("distanceToBoss", 100.0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        terrain = state.get("terrainInfo", [1.0] * 5)
        vel_x = state.get("playerVelocityX", 0.0)
        vel_y = state.get("playerVelocityY", 0.0)
        is_grounded = state.get("isGrounded", False)
        facing_boss = state.get("isFacingBoss", False)

        # 1. BOSS DAMAGE (highest priority)
        if self.prev_boss_health is not None:
            damage_dealt = self.prev_boss_health - boss_hp
            if damage_dealt > 0:
                damage_reward = damage_dealt * 40.0
                reward += damage_reward
                self.total_damage_dealt += damage_dealt
                self.total_boss_hits += 1
                info["boss_hit"] = True
                print(
                    f"💥 Boss Hit #{self.total_boss_hits}: {damage_dealt} HP (+{damage_reward:.1f})"
                )

        # 2. MANTIS LORDS KILL
        if self.prev_state is not None:
            prev_mantis = self.prev_state.get("mantisLordsKilled", 0)
            if mantis_killed > prev_mantis:
                kill_reward = 400.0
                reward += kill_reward
                info["mantis_killed"] = True
                print(f"🏆 Mantis Lord #{mantis_killed} defeated! (+{kill_reward:.1f})")

        # 3. ATTACK ACTION REWARD (combat encouragement)
        if action_name == "ATTACK":
            if 2.0 <= distance <= 8.0:
                attack_reward = 8.0
                reward += attack_reward
            elif distance < 2.0:
                attack_reward = 6.0
                reward += attack_reward

        # 4. DISTANCE SHAPING (approach bonus)
        if self.prev_distance is not None and distance < 90.0:
            distance_change = self.prev_distance - distance
            if distance_change > 0.3:
                if distance > 3.0:
                    reward += distance_change * 0.4
            elif distance_change < -0.5 and distance > 15.0:
                reward -= abs(distance_change) * 1.5

        # 5. WALL COLLISION PENALTY (prevent wall-banging)
        if len(terrain) >= 3:
            wall_dist = terrain[2]
            is_moving = action_name in ["MOVE_LEFT", "MOVE_RIGHT"]
            is_stuck = abs(vel_x) < 0.1 and abs(vel_y) < 0.1

            if is_moving and is_stuck and wall_dist < 0.15 and is_grounded:
                self.wall_stuck_counter += 1
                if self.wall_stuck_counter > 5:
                    wall_penalty = -2.5
                    reward += wall_penalty
            else:
                self.wall_stuck_counter = max(0, self.wall_stuck_counter - 1)

        # 6. IDLE NEAR BOSS PENALTY
        is_idle = abs(vel_x) < 0.15 and abs(vel_y) < 0.15
        if distance < 12.0 and is_idle:
            self.idle_counter += 1
            if self.idle_counter > 40:
                idle_penalty = (self.idle_counter - 40) * 0.08
                reward -= idle_penalty
        else:
            self.idle_counter = 0

        # 7. FAR FROM BOSS PENALTY
        if distance > 18.0:
            self.steps_far += 1
            if self.steps_far > 60:
                far_penalty = (self.steps_far - 60) * 0.25
                reward -= far_penalty
        else:
            self.steps_far = 0

        # 8. HEALTH LOSS PENALTY
        if self.prev_player_health is not None:
            hp_loss = self.prev_player_health - player_hp
            if hp_loss > 0:
                hp_penalty = hp_loss * 12.0
                reward -= hp_penalty
                info["damage_taken"] = hp_loss
                print(f"💔 Damage taken: -{hp_loss} HP (-{hp_penalty:.1f})")

        # 9. FACING BOSS BONUS (small orientation reward)
        if facing_boss and distance < 12.0:
            reward += 0.15

        # 10. TIME PENALTY (encourage efficiency)
        reward -= 0.0008

        # 11. TERMINAL REWARDS
        if done:
            if state.get("isDead", False):
                death_penalty = 40.0
                reward -= death_penalty
                info["death"] = True
                print(f"💀 Death penalty: -{death_penalty:.1f}")
            elif state.get("bossDefeated", False) or mantis_killed == 3:
                health_bonus = player_hp * 15.0
                victory_reward = 800.0 + health_bonus
                reward += victory_reward
                info["victory"] = True
                print(f"🎉 VICTORY! (+{victory_reward:.1f})")

        # Update tracking
        self.prev_boss_health = boss_hp
        self.prev_player_health = player_hp
        self.prev_distance = distance
        self.prev_state = state

        info["total_damage_dealt"] = self.total_damage_dealt
        info["total_boss_hits"] = self.total_boss_hits

        return reward, info

    def reset(self) -> Dict:
        """Reset environment for new episode."""
        print("[Env] Reset - waiting for new episode...")
        self._send_action("IDLE")

        state = self._receive_state()
        attempts = 0
        max_attempts = 20

        while state is None and attempts < max_attempts:
            time.sleep(0.5)
            state = self._receive_state()
            attempts += 1
            if attempts % 4 == 0:
                print(f"[Env] Waiting... ({attempts * 0.5:.1f}s)")

        if state is None:
            print("[Env] WARNING: No state after reset")
            return {}

        print(f"[Env] ✓ Reset complete ({attempts * 0.5:.1f}s)")

        # Reset reward tracking
        if self.use_reward_shaping:
            self._init_reward_tracking()
            self.prev_state = state
            self.prev_boss_health = state.get("bossHealth", 100.0)
            self.prev_player_health = state.get("playerHealth", 5)
            self.prev_distance = state.get("distanceToBoss", 100.0)

        return state

    def step(self, action: int) -> Tuple:
        """
        Execute action and return next state.

        Returns:
            If use_reward_shaping=False (DQN mode):
                (state_dict, done, info)
            If use_reward_shaping=True (PPO mode):
                (state_dict, reward, done, info)
        """
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
        info = {"action_name": action_name, "timestamp": state.get("timestamp", 0)}

        if self.use_reward_shaping:
            reward, reward_info = self._compute_reward(state, action_name, done)
            info.update(reward_info)
            return state, reward, done, info
        else:
            return state, done, info

    def close(self):
        """Close socket connection."""
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
