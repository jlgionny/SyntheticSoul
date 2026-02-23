"""
═══════════════════════════════════════════════════════════════════════
  Hollow Knight Environment — DQN VARIANT (Sparse/Discrete Rewards)
  Mantis Lords Boss Fight

  KEY DIFFERENCE FROM PPO ENV:
  DQN learns from individual (s, a, r, s') transitions stored in a
  replay buffer. It bootstraps Q-values via the Bellman equation,
  which propagates sparse terminal rewards backward through the
  buffer over many training steps.

  REWARD PHILOSOPHY:
  ● Minimal per-step signal (avoid reward noise in Q-estimates)
  ● Large, discrete event-based rewards (damage dealt, damage taken,
    kills, death, victory) — these create clear Q-value peaks that
    DQN can lock onto
  ● Binary-style bonuses (got hit? big penalty. Dealt damage? big
    reward) rather than continuous shaping
  ● Sparse terminal bonuses dominate: DQN's replay + target network
    will propagate these backwards efficiently
  ● No distance-based shaping — DQN can overfit continuous shaping
    signals and develop "orbiting" behaviors

  OBSERVATION & ACTION SPACE: Identical to env_ppo.py
═══════════════════════════════════════════════════════════════════════
"""

import socket
import json
import time
from typing import Dict, Tuple, Optional, Any
import numpy as np


class HollowKnightEnvDQN:
    """
    Environment wrapper for Hollow Knight — Mantis Lords (DQN variant).

    Sparse, event-driven reward structure optimized for off-policy
    Q-learning. Replay buffer + target network handle temporal credit
    assignment, so we rely on large discrete rewards at key events.

    TRAINING PHASES (identical to PPO env):
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
        print(f"[EnvDQN] Initialized — PHASE {phase}: {phase_names.get(phase, '?')}")
        print(f"         Host: {host}:{port} | Reward Scale: 1/{self.reward_scale}")
        print(f"         Mode: SPARSE rewards (DQN-optimized)")

        self._connect()

    # ═══════════════════════════════════════════════════════════════
    # NETWORK — Identical to PPO env
    # ═══════════════════════════════════════════════════════════════

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.socket_file = self.socket.makefile("r", encoding="utf-8")
            self.connected = True
            print(f"[EnvDQN] Connected.")
        except Exception as e:
            print(f"[EnvDQN] Connection failed: {e}")
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
    # REWARD FUNCTIONS — DQN VARIANT (SPARSE/DISCRETE)
    #
    # KEY DIFFERENCES FROM PPO:
    #   1. Near-zero per-step reward (avoid Q-value noise)
    #   2. Large discrete events: damage dealt/taken are binary-like
    #      spikes, not continuous shaping
    #   3. No distance tracking or movement quality — DQN overfits
    #      these and develops circular movement patterns
    #   4. Higher magnitude terminal rewards: kill (+100), death (-5),
    #      victory (+50) — replay buffer propagates these
    #   5. Threshold bonuses are sharper (step functions, not ramps)
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

    def _reward_phase1_survive(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 1: SURVIVAL (DQN Sparse)
        Sparse: minimal per-step, large penalty on damage/death,
        milestone bonuses at long survival intervals.
        """
        reward = 0.0
        info = {}

        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)

        # ═══ DQN SPARSE: Minimal living reward ═══
        # Near-zero to avoid Q-value inflation; DQN doesn't need
        # dense signal — replay buffer handles credit assignment
        reward += 0.01

        # ═══ DQN SPARSE: Large discrete penalty on damage ═══
        # Single sharp penalty — DQN learns "avoid this state" directly
        if damage_taken > 0:
            reward -= 4.0  # Higher than PPO's -3.0
            if hazard_type == 2:  # Spikes
                reward -= 4.0  # Harsh spike penalty
                info["damage_source"] = "SPIKES"

        # Dodge success (kept but smaller than PPO — event-based)
        if state.get("primaryMantisActive", False) and damage_taken == 0:
            reward += 0.15
            info["event"] = "DODGE_SUCCESS"

        # Wind-up reaction
        if state.get("primaryMantisWindUp", False):
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.05

        # ═══ DQN SPARSE: Large terminal death penalty ═══
        if done and is_dead:
            reward -= 3.0

        # ═══ DQN SPARSE: Infrequent milestone bonuses ═══
        # Every 200 steps (less frequent than PPO's 100) with larger bonus
        if self.episode_steps > 0 and self.episode_steps % 200 == 0:
            reward += 0.5
            info["milestone"] = f"SURVIVED_{self.episode_steps}_STEPS"

        return reward, info

    def _reward_phase2_first_hits(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 2: FIRST HITS (DQN Sparse)
        Sparse: large reward per damage event, large penalty per hit taken.
        No approach/retreat shaping.
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)

        # ═══ DQN SPARSE: Minimal per-step ═══
        reward += 0.005

        # ═══ DQN SPARSE: Large per-event damage dealt reward ═══
        # Higher multiplier than PPO — each hit is a clear Q-value spike
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.15  # Higher than PPO's 0.12
                info["damage_dealt"] = damage_dealt
                self.total_damage_dealt += damage_dealt

                # Recovery punish bonus (large discrete)
                if state.get("primaryMantisRecovering", False):
                    reward += 0.3  # Sharp bonus for smart timing
                    info["smart_hit"] = True

        # ═══ DQN SPARSE: Large discrete damage taken penalty ═══
        if damage_taken > 0:
            reward -= 3.0
            self.total_damage_taken += damage_taken
            if hazard_type == 2:
                reward -= 3.0

        # Penalize attacking during wind-up
        if state.get("primaryMantisWindUp", False) and self.last_action == 5:
            reward -= 0.1

        # ═══ DQN SPARSE: No distance/position shaping ═══
        # (PPO env has approach/retreat rewards here — DQN skips these
        # to avoid Q-value noise and orbiting behavior)

        if done and is_dead:
            reward -= 3.0

        self.prev_boss_hp = boss_hp
        return reward, info

    def _reward_phase3_aggression(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 3: AGGRESSION (DQN Sparse)
        Sparse: large damage multiplier + sharp HP threshold bonuses +
        very large kill reward. This is the core Q-learning signal.
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)

        # ═══ DQN SPARSE: Minimal living tick ═══
        reward += 0.001

        # ═══ DQN SPARSE: High damage dealt reward ═══
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.2  # Higher than PPO's 0.18
                info["damage_dealt"] = damage_dealt

            # ═══ DQN SPARSE: Sharp HP threshold step bonuses ═══
            # Larger than PPO — these are the key discrete signals
            # that DQN's replay buffer propagates backward
            for threshold, bonus in [(200, 3.0), (100, 5.0), (50, 8.0)]:
                if self.prev_boss_hp > threshold >= boss_hp:
                    reward += bonus
                    info[f"threshold_{threshold}"] = True

        # ═══ DQN SPARSE: No HP progress tracking ═══
        # (PPO env has continuous hp_progress here — DQN relies on
        # discrete threshold bonuses instead)

        # Damage taken (reduced to encourage aggression)
        if damage_taken > 0:
            reward -= 1.5
            if hazard_type == 2:
                reward -= 3.0

        # ═══ DQN SPARSE: Very large kill reward ═══
        # The primary Q-learning target — large enough to dominate
        # the Q-value landscape and pull trajectories toward kills
        if mantis_killed > self.prev_mantis_killed:
            reward += 100.0  # PPO uses 60.0 — DQN needs larger
            info["event"] = "MANTIS_KILLED"

        if done:
            if state.get("bossDefeated", False):
                reward += 50.0  # PPO uses 30.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 5.0  # PPO uses -3.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    def _reward_phase4_dual_mantis(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 4: DUAL MANTIS (DQN Sparse)
        Sparse: large event rewards, no center-positioning shaping.
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        active_count = state.get("activeMantisCount", 0)

        # ═══ DQN SPARSE: Minimal tick ═══
        reward += 0.001

        # Damage dealt
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.2
                info["damage_dealt"] = damage_dealt

        # ═══ DQN SPARSE: Higher damage penalty (two sources) ═══
        if damage_taken > 0:
            reward -= 3.5  # Higher than PPO's -3.0
            if hazard_type == 2:
                reward -= 3.0

        # ═══ DQN SPARSE: Large kill bonus (progressive) ═══
        if mantis_killed > self.prev_mantis_killed:
            kills_diff = mantis_killed - self.prev_mantis_killed
            reward += 100.0 * kills_diff  # PPO uses 60.0
            info["event"] = f"MANTIS_KILLED_x{mantis_killed}"

        # Dual awareness — discrete event-based dodge bonus
        if active_count >= 2 and state.get("secondaryMantisActive", False):
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.08
                info["dual_dodge"] = True
            # ═══ DQN: No center-positioning bonus ═══
            # (PPO env has arena-center shaping here — DQN skips it)

        if done:
            if state.get("bossDefeated", False):
                reward += 50.0  # PPO uses 30.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 5.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    def _reward_phase5_mastery(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 5: MASTERY (DQN Sparse)
        Sparse: high per-hit reward + very high terminal bonuses.
        No-hit and speed bonuses are terminal (not continuous).
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)

        # ═══ DQN SPARSE: Minimal tick ═══
        reward += 0.001

        # ═══ DQN SPARSE: No step penalty (unlike PPO) ═══
        # DQN handles speed optimization through terminal time bonus

        # Damage dealt (high multiplier for speed)
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                reward += damage_dealt * 0.25  # Same as PPO here
                info["damage_dealt"] = damage_dealt

        # ═══ DQN SPARSE: Very high damage penalty (no-hit goal) ═══
        if damage_taken > 0:
            reward -= 5.0
            if hazard_type == 2:
                reward -= 4.0

        # ═══ DQN SPARSE: No continuous no-hit streak ═══
        # (PPO env tracks continuous no-hit bonus — DQN gives it
        # all at terminal to avoid Q-value noise)

        # Kill
        if mantis_killed > self.prev_mantis_killed:
            reward += 40.0

        # ═══ DQN SPARSE: Large terminal bonuses ═══
        if done:
            if state.get("bossDefeated", False):
                reward += 50.0  # Higher than PPO's 30.0
                # Speed bonus (terminal, not per-step)
                time_bonus = max(0, (3000 - self.episode_steps) / 100.0)
                reward += time_bonus
                # No-hit bonus (all at terminal)
                if self.total_damage_taken == 0:
                    reward += 30.0  # Higher than PPO's 20.0
                    info["outcome"] = "PERFECT_VICTORY"
                else:
                    info["outcome"] = "VICTORY"
                info["time_bonus"] = time_bonus
            elif is_dead:
                reward -= 8.0  # Harsher than PPO's -5.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    # ═══════════════════════════════════════════════════════════════
    # ENV API — Identical interface to PPO env
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

        return state if state else {}

    def step(self, action: int) -> Tuple:
        action_name = self.ACTIONS.get(action, "MOVE_LEFT")
        self._send_action(action_name)

        self.last_action = action
        self.episode_steps += 1

        state = self._receive_state()

        if state is None:
            return {}, 0, True, {"error": "Connection lost"}

        done = state.get("isDead", False) or state.get("bossDefeated", False)

        raw_reward, info = self._compute_reward(state, done)
        scaled_reward = raw_reward / self.reward_scale
        info["raw_reward"] = raw_reward
        info["action"] = action_name
        info["phase"] = self.phase
        info["reward_mode"] = "sparse_dqn"

        return state, scaled_reward, done, info

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.connected = False
