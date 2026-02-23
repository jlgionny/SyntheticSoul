"""
═══════════════════════════════════════════════════════════════════════
  Hollow Knight Environment — PPO VARIANT (Dense Rewards)
  Mantis Lords Boss Fight

  KEY DIFFERENCE FROM DQN ENV:
  PPO benefits from DENSE, continuous reward signals because it uses
  on-policy gradient estimation (GAE). Sparse rewards create high
  variance in advantage estimates, stalling learning.

  REWARD PHILOSOPHY:
  ● Small continuous rewards every step (survival ticks, distance
    tracking, positioning quality)
  ● Moderate shaping signals (dodge success, approach/retreat
    quality, wind-up awareness)
  ● Scaled terminal bonuses (kills, victory, death) — still present
    but less dominant relative to the dense signal stream
  ● Gradient-friendly: rewards are smooth and differentiable in
    expectation, reducing GAE variance

  OBSERVATION & ACTION SPACE: Identical to env_dqn.py
═══════════════════════════════════════════════════════════════════════
"""

import socket
import json
import time
from typing import Dict, Tuple, Optional, Any
import numpy as np


class HollowKnightEnvPPO:
    """
    Environment wrapper for Hollow Knight — Mantis Lords (PPO variant).

    Dense reward shaping for on-policy learning. PPO collects rollouts
    and computes advantages via GAE(λ), so every timestep should carry
    a meaningful gradient signal.

    TRAINING PHASES (identical to DQN env):
      1 - SURVIVAL:      Learn to dodge, move, not die
      2 - FIRST HITS:    Learn to punish during recovery windows
      3 - AGGRESSION:    Maximize DPS, kill first mantis
      4 - DUAL MANTIS:   Handle two mantises simultaneously
      5 - MASTERY:       Full victory, optimize time & no-hit
    """

    # ═══ ACTION SPACE — Identical across PPO/DQN ═══
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
        phase: int = 1,
        reward_scale: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self.phase = phase
        self.reward_scale = reward_scale

        # Previous-state tracking for delta computation
        self.prev_boss_hp = None
        self.prev_mantis_killed = 0
        self.prev_player_hp = None

        # ═══ PPO-SPECIFIC: extra tracking for dense shaping ═══
        self.prev_distance_to_boss = None   # Track approach/retreat
        self.prev_player_x = None           # Track movement quality
        self.consecutive_idle_steps = 0     # Penalize standing still
        self.steps_without_damage = 0       # Reward sustained dodging

        # Action tracking
        self.last_action = None
        self.steps_since_attack = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.episode_steps = 0

        phase_names = {
            1: "SURVIVAL", 2: "FIRST HITS", 3: "AGGRESSION",
            4: "DUAL MANTIS", 5: "MASTERY"
        }
        print(f"[EnvPPO] Initialized — PHASE {phase}: {phase_names.get(phase, '?')}")
        print(f"         Host: {host}:{port} | Reward Scale: 1/{self.reward_scale}")
        print(f"         Mode: DENSE rewards (PPO-optimized)")

        self._connect()

    # ═══════════════════════════════════════════════════════════════
    # NETWORK — Identical to DQN env
    # ═══════════════════════════════════════════════════════════════

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.socket_file = self.socket.makefile("r", encoding="utf-8")
            self.connected = True
            print(f"[EnvPPO] Connected.")
        except Exception as e:
            print(f"[EnvPPO] Connection failed: {e}")
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

    # ═══════════════════════════════════════════════════════════════
    # REWARD FUNCTIONS — PPO VARIANT (DENSE SHAPING)
    #
    # KEY DIFFERENCES FROM DQN:
    #   1. Every step yields non-zero reward (survival tick, position)
    #   2. Distance-to-boss tracking: reward approaching during safe
    #      windows, reward retreating during attacks
    #   3. Dodge streaks: small cumulative bonus for consecutive
    #      no-damage steps during active boss attacks
    #   4. Smoothed terminal bonuses: kill/death rewards are lower
    #      magnitude since dense signal already captures progress
    #   5. Movement quality: penalize idle, reward purposeful motion
    # ═══════════════════════════════════════════════════════════════

    def _compute_reward(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """Dispatch to phase-specific reward function."""
        if self.phase == 1:
            return self._reward_phase1_survive(state, done)
        elif self.phase == 2:
            return self._reward_phase2_first_hits(state, done)
        elif self.phase == 3:
            return self._reward_phase3_aggression(state, done)
        elif self.phase == 4:
            return self._reward_phase4_dual_mantis(state, done)
        elif self.phase == 5:
            return self._reward_phase5_mastery(state, done)
        else:
            return self._reward_phase3_aggression(state, done)

    # ─── Shared dense shaping utilities (PPO-only) ───

    def _distance_shaping(self, state: Dict, safe_to_approach: bool) -> float:
        """
        PPO-SPECIFIC: Continuous reward for spatial awareness.
        Reward approaching boss during safe windows (recovery),
        reward retreating during dangerous windows (wind-up/attack).
        """
        dist = state.get("distanceToBoss", 50.0)
        reward = 0.0

        if self.prev_distance_to_boss is not None:
            delta_dist = self.prev_distance_to_boss - dist  # positive = approaching

            if safe_to_approach:
                # Reward approaching during punish windows
                reward += np.clip(delta_dist * 0.02, -0.05, 0.05)
            else:
                # Reward maintaining/increasing distance during danger
                reward += np.clip(-delta_dist * 0.015, -0.03, 0.05)

        self.prev_distance_to_boss = dist
        return reward

    def _movement_quality(self, state: Dict) -> float:
        """
        PPO-SPECIFIC: Small reward for purposeful movement, penalize
        standing still. Prevents PPO from converging on idle policies.
        """
        player_x = state.get("playerPositionX", 0.0)
        reward = 0.0

        if self.prev_player_x is not None:
            movement = abs(player_x - self.prev_player_x)
            if movement < 0.01:
                self.consecutive_idle_steps += 1
                if self.consecutive_idle_steps > 30:
                    reward -= 0.005  # Gentle idle penalty after ~0.5s
            else:
                self.consecutive_idle_steps = 0
                reward += 0.002  # Tiny movement reward

        self.prev_player_x = player_x
        return reward

    def _dodge_streak_bonus(self, damage_taken: int, boss_attacking: bool) -> float:
        """
        PPO-SPECIFIC: Cumulative bonus for consecutive dodges.
        Grows slightly over time to reward sustained awareness.
        """
        if damage_taken > 0:
            self.steps_without_damage = 0
            return 0.0

        if boss_attacking:
            self.steps_without_damage += 1
            # Logarithmic growth: diminishing returns but never zero
            if self.steps_without_damage > 5:
                return 0.01 * min(np.log(self.steps_without_damage), 3.0)
        return 0.0

    # ─── Phase-specific reward functions ───

    def _reward_phase1_survive(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 1: SURVIVAL (PPO Dense)
        Dense: continuous survival tick + distance shaping + dodge streaks
        """
        reward = 0.0
        info = {}

        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        boss_attacking = state.get("primaryMantisActive", False)
        boss_windup = state.get("primaryMantisWindUp", False)

        # ═══ PPO DENSE: Continuous survival tick ═══
        # Higher than DQN's sparse version; PPO needs per-step signal
        reward += 0.02

        # ═══ PPO DENSE: Distance shaping ═══
        # In survival phase, staying at medium distance is good
        safe = not boss_attacking and not boss_windup
        reward += self._distance_shaping(state, safe_to_approach=safe)

        # ═══ PPO DENSE: Movement quality ═══
        reward += self._movement_quality(state)

        # ═══ PPO DENSE: Dodge streak bonus ═══
        reward += self._dodge_streak_bonus(damage_taken, boss_attacking)

        # Damage taken penalty (moderate — PPO learns from dense signal)
        if damage_taken > 0:
            reward -= 3.0  # Lower than DQN's -4.0; dense signal compensates
            if hazard_type == 2:  # Spikes
                reward -= 3.0
                info["damage_source"] = "SPIKES"

        # Dodge success during active boss attack
        if boss_attacking and damage_taken == 0:
            reward += 0.15
            info["event"] = "DODGE_SUCCESS"

        # Wind-up reaction bonus
        if boss_windup and self.last_action in [4, 6]:  # JUMP or DASH
            reward += 0.08  # Slightly higher than original for PPO signal

        # Death
        if done and is_dead:
            reward -= 2.0  # Lower than DQN — dense signal already penalized

        # ═══ PPO DENSE: Survival milestones (smoother) ═══
        # Every 100 steps instead of 200, smaller bonus (smoother gradient)
        if self.episode_steps > 0 and self.episode_steps % 100 == 0:
            reward += 0.3
            info["milestone"] = f"SURVIVED_{self.episode_steps}_STEPS"

        return reward, info

    def _reward_phase2_first_hits(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 2: FIRST HITS (PPO Dense)
        Dense: approach during recovery + retreat during attack + damage dealt
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        boss_recovering = state.get("primaryMantisRecovering", False)
        boss_windup = state.get("primaryMantisWindUp", False)
        boss_attacking = state.get("primaryMantisActive", False)

        # ═══ PPO DENSE: Per-step living reward ═══
        reward += 0.01

        # ═══ PPO DENSE: Distance shaping — approach during recovery ═══
        safe = boss_recovering or (not boss_attacking and not boss_windup)
        reward += self._distance_shaping(state, safe_to_approach=safe)

        # ═══ PPO DENSE: Dodge streaks ═══
        reward += self._dodge_streak_bonus(damage_taken, boss_attacking)

        # Damage dealt
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.12  # Slightly lower per-hit than DQN
                info["damage_dealt"] = damage_dealt
                self.total_damage_dealt += damage_dealt

                # Smart hit bonus: attacking during recovery
                if boss_recovering:
                    reward += 0.25
                    info["smart_hit"] = True

        # Damage taken
        if damage_taken > 0:
            reward -= 2.5  # Lower than DQN's -3.0; dense signal compensates
            self.total_damage_taken += damage_taken
            if hazard_type == 2:
                reward -= 2.5

        # Penalize attacking during wind-up (bad trade)
        if boss_windup and self.last_action == 5:
            reward -= 0.1

        # ═══ PPO DENSE: Attack timing awareness ═══
        # Small reward for attacking when close + boss recovering
        dist = state.get("distanceToBoss", 50.0)
        if self.last_action == 5 and boss_recovering and dist < 10.0:
            reward += 0.1  # Reinforces punish-window aggression
            info["optimal_attack_position"] = True

        if done and is_dead:
            reward -= 2.0

        self.prev_boss_hp = boss_hp
        return reward, info

    def _reward_phase3_aggression(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 3: AGGRESSION (PPO Dense)
        Dense: HP progress tracking + approach shaping + kill bonus (lower
        magnitude since dense signal accumulates over the episode).
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        boss_recovering = state.get("primaryMantisRecovering", False)
        boss_attacking = state.get("primaryMantisActive", False)

        # ═══ PPO DENSE: Continuous tick ═══
        reward += 0.005

        # ═══ PPO DENSE: Distance shaping ═══
        safe = boss_recovering or not boss_attacking
        reward += self._distance_shaping(state, safe_to_approach=safe)

        # Damage dealt (per-hit reward)
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.18
                info["damage_dealt"] = damage_dealt

            # ═══ PPO DENSE: HP threshold bonuses (smoothed) ═══
            # Lower magnitude than DQN since PPO accumulates dense signal
            for threshold, bonus in [(200, 2.0), (100, 3.5), (50, 5.0)]:
                if self.prev_boss_hp > threshold >= boss_hp:
                    reward += bonus
                    info[f"threshold_{threshold}"] = True

        # ═══ PPO DENSE: HP progress tracking ═══
        # Continuous small reward proportional to boss HP lost so far
        # This gives PPO a gradient even between discrete damage events
        if boss_hp < (self.prev_boss_hp or 0):
            hp_progress = 1.0 - (boss_hp / max(state.get("bossMaxHealth", 200.0), 1.0))
            reward += hp_progress * 0.02  # Scales with fight progress

        # Damage taken (reduced for aggression)
        if damage_taken > 0:
            reward -= 1.2
            if hazard_type == 2:
                reward -= 2.5

        # Kill bonus — lower than DQN since dense signal accumulates
        if mantis_killed > self.prev_mantis_killed:
            reward += 60.0  # DQN uses 100.0 — PPO doesn't need as large
            info["event"] = "MANTIS_KILLED"

        if done:
            if state.get("bossDefeated", False):
                reward += 30.0  # DQN uses 50.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 3.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    def _reward_phase4_dual_mantis(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 4: DUAL MANTIS (PPO Dense)
        Dense: spatial awareness shaping + dual-dodge tracking
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        active_count = state.get("activeMantisCount", 0)
        boss_attacking = state.get("primaryMantisActive", False)
        secondary_active = state.get("secondaryMantisActive", False)

        # ═══ PPO DENSE: Continuous tick ═══
        reward += 0.005

        # ═══ PPO DENSE: Distance shaping ═══
        safe = not boss_attacking and not secondary_active
        reward += self._distance_shaping(state, safe_to_approach=safe)

        # ═══ PPO DENSE: Dodge streaks (extra important in phase 4) ═══
        reward += self._dodge_streak_bonus(damage_taken, boss_attacking or secondary_active)

        # Damage dealt
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.18
                info["damage_dealt"] = damage_dealt

        # Damage taken (high — two mantises are dangerous)
        if damage_taken > 0:
            reward -= 3.0
            if hazard_type == 2:
                reward -= 2.5

        # Kill bonus
        if mantis_killed > self.prev_mantis_killed:
            kills_diff = mantis_killed - self.prev_mantis_killed
            reward += 60.0 * kills_diff
            info["event"] = f"MANTIS_KILLED_x{mantis_killed}"

        # ═══ PPO DENSE: Dual awareness shaping ═══
        # Continuous reward for reactive movement when both mantises active
        if active_count >= 2 and secondary_active:
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.12
                info["dual_dodge"] = True
            # Extra: reward being in center of arena for better escape angles
            player_x = state.get("playerPositionX", 0.0)
            arena_center = state.get("arenaCenterX", 0.0)
            if arena_center != 0.0:
                center_dist = abs(player_x - arena_center) / 50.0
                reward += max(0, 0.02 - center_dist * 0.01)  # Tiny center bonus

        if done:
            if state.get("bossDefeated", False):
                reward += 30.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 3.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    def _reward_phase5_mastery(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 5: MASTERY (PPO Dense)
        Dense: speed bonus + no-hit tracking + efficiency shaping
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        boss_recovering = state.get("primaryMantisRecovering", False)

        # ═══ PPO DENSE: Continuous tick with step penalty ═══
        # Tiny negative per step to incentivize speed
        reward += 0.002
        reward -= 0.001 * (self.episode_steps / 1000.0)  # Grows over time

        # ═══ PPO DENSE: Aggressive distance shaping ═══
        safe = boss_recovering
        reward += self._distance_shaping(state, safe_to_approach=safe) * 1.5

        # Damage dealt (highest multiplier — speed is king)
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.22
                info["damage_dealt"] = damage_dealt

        # Damage taken (VERY high penalty — no-hit objective)
        if damage_taken > 0:
            reward -= 4.0
            if hazard_type == 2:
                reward -= 3.5

        # ═══ PPO DENSE: No-hit streak bonus (continuous) ═══
        # The longer you go without taking damage, the more reward per step
        if damage_taken == 0 and self.episode_steps > 100:
            no_hit_bonus = min(0.03, self.episode_steps * 0.00002)
            reward += no_hit_bonus

        # Kill
        if mantis_killed > self.prev_mantis_killed:
            reward += 30.0

        if done:
            if state.get("bossDefeated", False):
                reward += 30.0
                # ═══ PPO DENSE: Smoother speed bonus ═══
                time_bonus = max(0, (3000 - self.episode_steps) / 150.0)
                reward += time_bonus
                # No-hit bonus
                if self.total_damage_taken == 0:
                    reward += 20.0
                    info["outcome"] = "PERFECT_VICTORY"
                else:
                    info["outcome"] = "VICTORY"
                info["time_bonus"] = time_bonus
            elif is_dead:
                reward -= 5.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    # ═══════════════════════════════════════════════════════════════
    # ENV API — Identical interface to DQN env
    # ═══════════════════════════════════════════════════════════════

    def reset(self) -> Dict:
        state = self._receive_state()
        attempts = 0
        while state is None and attempts < 10:
            time.sleep(0.1)
            state = self._receive_state()
            attempts += 1

        if state:
            self.prev_boss_hp = state.get("bossHealth", 100.0)
            self.prev_mantis_killed = 0
            self.prev_player_hp = state.get("playerHealth", 9)

        self.last_action = None
        self.steps_since_attack = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.episode_steps = 0

        # ═══ PPO-SPECIFIC: Reset dense tracking state ═══
        self.prev_distance_to_boss = None
        self.prev_player_x = None
        self.consecutive_idle_steps = 0
        self.steps_without_damage = 0

        return state if state else {}

    def step(self, action: int) -> Tuple:
        action_name = self.ACTIONS.get(action, "MOVE_LEFT")
        self._send_action(action_name)

        self.last_action = action
        self.episode_steps += 1

        # Track attack timing
        if action == 5:
            self.steps_since_attack = 0
        else:
            self.steps_since_attack += 1

        state = self._receive_state()

        if state is None:
            return {}, 0, True, {"error": "Connection lost"}

        done = state.get("isDead", False) or state.get("bossDefeated", False)

        raw_reward, info = self._compute_reward(state, done)
        scaled_reward = raw_reward / self.reward_scale
        info["raw_reward"] = raw_reward
        info["action"] = action_name
        info["phase"] = self.phase
        info["reward_mode"] = "dense_ppo"

        return state, scaled_reward, done, info

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.connected = False
