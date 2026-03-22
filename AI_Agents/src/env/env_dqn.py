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

  FIXES APPLIED:
  ● Phase 3: Progressive kill rewards (1st kill reduced, 2nd/3rd boosted)
  ● Phase 3: HP threshold bonuses during dual phase for intermediate signal
  ● Phase 3: Damage multiplier boost during dual phase
  ● Phase 3: Death penalty scaled by progress (less punitive if fighting duals)
  ● Phase 4: Same progressive structure + speed/no-hit terminal bonuses
═══════════════════════════════════════════════════════════════════════
"""

import socket
import json
from threading import active_count
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
      1 - SURVIVE:       Learn to dodge, move, not die
      2 - FIRST BLOOD:   Deal damage and kill the first mantis
      3 - DUAL MANTIS:   Handle two mantises simultaneously
      4 - MASTERY:       Full victory, optimize time & no-hit
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

        phase_names = {1: "SURVIVE", 2: "FIRST BLOOD", 3: "DUAL MANTIS", 4: "MASTERY"}
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
            return self._reward_phase2_first_blood(state, done)
        elif self.phase == 3:
            return self._reward_phase3_dual_mantis(state, done)
        elif self.phase == 4:
            return self._reward_phase4_mastery(state, done)
        else:
            return self._reward_phase2_first_blood(state, done)

    def _reward_phase1_survive(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 1: SURVIVE (DQN Sparse)
        Sparse: minimal per-step, large penalty on damage/death,
        milestone bonuses at long survival intervals.
        """
        reward = 0.0
        info = {}

        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)

        # ═══ DQN SPARSE: Minimal living reward ═══
        reward += 0.01

        # ═══ DQN SPARSE: Large discrete penalty on damage ═══
        if damage_taken > 0:
            reward -= 4.0
            if hazard_type == 2:  # Spikes
                reward -= 4.0
                info["damage_source"] = "SPIKES"

        # Dodge success
        if state.get("primaryMantisActive", False) and damage_taken == 0:
            reward += 0.15
            info["event"] = "DODGE_SUCCESS"

        # Wind-up reaction
        if state.get("primaryMantisWindUp", False):
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.05

        # ═══ DQN SPARSE: Large terminal death penalty ═══
        if done and is_dead:
            reward -= 15.0

        # ═══ DQN SPARSE: Infrequent milestone bonuses ═══
        if self.episode_steps > 0 and self.episode_steps % 200 == 0:
            reward += 0.5
            info["milestone"] = f"SURVIVED_{self.episode_steps}_STEPS"

        return reward, info

    def _reward_phase2_first_blood(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 2: FIRST BLOOD (DQN Sparse)
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
                reward += damage_dealt * 0.2
                info["damage_dealt"] = damage_dealt

            # ═══ DQN SPARSE: Sharp HP threshold step bonuses ═══
            for threshold, bonus in [(200, 3.0), (100, 5.0), (50, 8.0)]:
                if self.prev_boss_hp > threshold >= boss_hp:
                    reward += bonus
                    info[f"threshold_{threshold}"] = True

        # Damage taken (reduced to encourage aggression)
        if damage_taken > 0:
            reward -= 1.5
            if hazard_type == 2:
                reward -= 3.0

        # ═══ DQN SPARSE: Very large kill reward ═══
        if mantis_killed > self.prev_mantis_killed:
            reward += 100.0
            info["event"] = "MANTIS_KILLED"

        if done:
            if state.get("bossDefeated", False):
                reward += 50.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 10.0

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: DUAL MANTIS — FIXED
    #
    # PROBLEMS FIXED:
    #   1. Kill reward was flat 100.0 per kill — agent was "happy"
    #      just killing the first mantis. Now progressive: 1st kill
    #      reduced to 50, 2nd boosted to 120, 3rd to 150.
    #   2. No intermediate signal during dual phase — agent got zero
    #      feedback for partial HP progress on the two mantises.
    #      Added HP threshold bonuses (600/500/400/300/200/100).
    #   3. Damage multiplier was flat 0.2 — now 0.35 during dual
    #      phase to reward every hit more strongly.
    #   4. Death penalty was flat -15.0 — now scaled by progress so
    #      dying in dual phase is less punitive (encourages aggression).
    # ═══════════════════════════════════════════════════════════════

    def _reward_phase3_dual_mantis(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 3: DUAL MANTIS (DQN Sparse) — FIXED
        Progressive kill rewards + HP threshold bonuses during dual phase.
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

        # ═══ Damage dealt — higher multiplier during dual phase ═══
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                # FIX: Boost damage reward during dual phase to reward aggression
                multiplier = 0.35 if mantis_killed >= 1 else 0.2
                reward += damage_dealt * multiplier
                info["damage_dealt"] = damage_dealt

            # ═══ FIX: HP threshold bonuses during dual phase ═══
            # Critical intermediate signal — the agent sees incremental
            # progress toward the 2nd and 3rd kills instead of nothing
            if mantis_killed >= 1:
                for threshold, bonus in [
                    (600, 2.0),
                    (500, 4.0),
                    (400, 6.0),
                    (300, 8.0),
                    (200, 10.0),
                    (100, 12.0),
                ]:
                    if self.prev_boss_hp > threshold >= boss_hp:
                        reward += bonus
                        info[f"dual_threshold_{threshold}"] = True

        # ═══ Damage taken — more tolerant during dual phase ═══
        if damage_taken > 0:
            if mantis_killed >= 1:
                reward -= 2.0  # FIX: Was -3.5, reduced to encourage aggression vs duals
            else:
                reward -= 3.5
            if hazard_type == 2:
                reward -= 3.0

        # ═══ FIX: Progressive kill rewards ═══
        # 1st kill is already easy from phase 3 — reduce reward.
        # 2nd and 3rd kills are the actual learning targets — boost them.
        if mantis_killed > self.prev_mantis_killed:
            if mantis_killed == 1:
                reward += 50.0  # Was 100.0 — reduced, agent already knows this
                info["event"] = "MANTIS_KILLED_1"
            elif mantis_killed == 2:
                reward += 120.0  # Boosted — THIS is what phase 4 needs to learn
                info["event"] = "MANTIS_KILLED_2"
            elif mantis_killed == 3:
                reward += 150.0  # Highest kill reward — final kill
                info["event"] = "MANTIS_KILLED_3"

        # Dual awareness — discrete event-based dodge bonus
        if active_count >= 2 and state.get("secondaryMantisActive", False):
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.20
                info["dual_dodge"] = True

        # ═══ FIX: Death penalty scaled by progress ═══
        if done:
            if state.get("bossDefeated", False):
                reward += 50.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                if mantis_killed == 0:
                    reward -= 20.0  # Harsh — didn't even kill first
                elif mantis_killed == 1:
                    reward -= 8.0  # FIX: Was -15.0, now tolerant during dual learning
                else:
                    reward -= 3.0  # Nearly there — minimal punishment

        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        return reward, info

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: MASTERY — FIXED
    #
    # PROBLEMS FIXED:
    #   Same as phase 4, plus:
    #   1. Kill reward was flat 80.0 — now progressive (40/120/150)
    #   2. No HP thresholds at all — added for dual phase
    #   3. Death penalty was flat -30.0 — now scaled by progress
    #   4. Damage multiplier now boosted during dual phase
    # ═══════════════════════════════════════════════════════════════

    def _reward_phase4_mastery(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        PHASE 4: MASTERY (DQN Sparse) — FIXED
        Progressive kill rewards + HP thresholds + speed/no-hit terminal bonuses.
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

        # ═══ Damage dealt — higher multiplier during dual phase ═══
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                # FIX: Boost multiplier when fighting duals
                multiplier = 0.4 if mantis_killed >= 1 else 0.25
                reward += damage_dealt * multiplier
                info["damage_dealt"] = damage_dealt

            # ═══ FIX: HP threshold bonuses during dual phase ═══
            if mantis_killed >= 1:
                for threshold, bonus in [
                    (600, 3.0),
                    (500, 5.0),
                    (400, 8.0),
                    (300, 10.0),
                    (200, 12.0),
                    (100, 15.0),
                ]:
                    if self.prev_boss_hp > threshold >= boss_hp:
                        reward += bonus
                        info[f"dual_threshold_{threshold}"] = True

        # Dual dodge bonus
        if active_count >= 2:
            if self.last_action in [4, 6]:  # JUMP or DASH
                reward += 0.4
                info["dual_dodge"] = True

        # ═══ Damage taken — more tolerant during dual phase ═══
        if damage_taken > 0:
            if mantis_killed >= 1:
                reward -= 2.5  # FIX: Was -5.0, reduced during dual phase
            else:
                reward -= 5.0
            if hazard_type == 2:
                reward -= 4.0

        # ═══ FIX: Progressive kill rewards ═══
        if mantis_killed > self.prev_mantis_killed:
            if mantis_killed == 1:
                reward += 40.0  # Was 80.0 — reduced, already mastered
            elif mantis_killed == 2:
                reward += 120.0  # Boosted — key learning target
            elif mantis_killed == 3:
                reward += 150.0  # Highest — final kill

        # ═══ FIX: Terminal bonuses with progress-scaled death penalty ═══
        if done:
            if state.get("bossDefeated", False):
                reward += 120.0
                # Speed bonus (terminal, not per-step)
                time_bonus = max(0, (3000 - self.episode_steps) / 100.0)
                reward += time_bonus
                # No-hit bonus (all at terminal)
                if self.total_damage_taken == 0:
                    reward += 30.0
                    info["outcome"] = "PERFECT_VICTORY"
                else:
                    info["outcome"] = "VICTORY"
                info["time_bonus"] = time_bonus
            elif is_dead:
                # FIX: Was flat -30.0 — now scaled by progress
                if mantis_killed == 0:
                    reward -= 40.0  # Harsh — should at least kill first
                elif mantis_killed == 1:
                    reward -= 15.0  # Tolerant — died learning duals
                else:
                    reward -= 5.0  # Nearly there — minimal punishment

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
