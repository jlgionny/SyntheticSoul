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

        # ═══ PPO-SPECIFIC: extra tracking for dense shaping ═══
        self.prev_distance_to_boss = None  # Track approach/retreat
        self.prev_player_x = None  # Track movement quality
        self.consecutive_idle_steps = 0  # Penalize standing still
        self.steps_without_damage = 0  # Reward sustained dodging

        # Action tracking
        self.last_action = None
        self.steps_since_attack = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.episode_steps = 0

        phase_names = {1: "SURVIVE", 2: "FIRST BLOOD", 3: "DUAL MANTIS", 4: "MASTERY"}
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
    # ═══════════════════════════════════════════════════════════════

    def _compute_reward(self, state: Dict, done: bool) -> Tuple[float, Dict[str, Any]]:
        if self.phase == 1:
            return self._reward_phase1_survive(state, done)
        elif self.phase == 2:
            return self._reward_phase2_first_blood(state, done)
        elif self.phase in (3, 4):
            return self._reward_phase3_dual_mantis(state, done)
        else:
            return self._reward_phase2_first_blood(state, done)
            return self._reward_phase2_first_blood(state, done)

    # ─── Shared dense shaping utilities (PPO-only) ───

    def _distance_shaping(self, state: Dict, safe_to_approach: bool) -> float:
        dist = state.get("distanceToBoss", 50.0)
        reward = 0.0

        if self.prev_distance_to_boss is not None:
            delta_dist = self.prev_distance_to_boss - dist

            if safe_to_approach:
                reward += np.clip(delta_dist * 0.02, -0.05, 0.05)
            else:
                reward += np.clip(-delta_dist * 0.015, -0.03, 0.05)

        self.prev_distance_to_boss = dist
        return reward

    def _movement_quality(self, state: Dict) -> float:
        player_x = state.get("playerPositionX", 0.0)
        reward = 0.0

        if self.prev_player_x is not None:
            movement = abs(player_x - self.prev_player_x)
            if movement < 0.01:
                self.consecutive_idle_steps += 1
                if self.consecutive_idle_steps > 30:
                    reward -= 0.005
            else:
                self.consecutive_idle_steps = 0
                reward += 0.002

        self.prev_player_x = player_x
        return reward

    def _dodge_streak_bonus(self, damage_taken: int, boss_attacking: bool) -> float:
        if damage_taken > 0:
            self.steps_without_damage = 0
            return 0.0

        if boss_attacking:
            self.steps_without_damage += 1
            if self.steps_without_damage > 5:
                return 0.01 * min(np.log(self.steps_without_damage), 3.0)
        return 0.0

    # ─── Phase-specific reward functions ───

    def _reward_phase1_survive(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        info = {}
        damage_taken = state.get("damageTaken", 0)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        boss_attacking = state.get("primaryMantisActive", False)
        boss_windup = state.get("primaryMantisWindUp", False)

        reward += 0.02
        safe = not boss_attacking and not boss_windup
        reward += self._distance_shaping(state, safe_to_approach=safe)
        reward += self._movement_quality(state)
        reward += self._dodge_streak_bonus(damage_taken, boss_attacking)

        if damage_taken > 0:
            reward -= 3.0
            if hazard_type == 2:
                reward -= 3.0
                info["damage_source"] = "SPIKES"

        if boss_attacking and damage_taken == 0:
            reward += 0.15
            info["event"] = "DODGE_SUCCESS"

        if boss_windup and self.last_action in [4, 6]:
            reward += 0.08

        if done and is_dead:
            reward -= 2.0

        if self.episode_steps > 0 and self.episode_steps % 100 == 0:
            reward += 0.3
            info["milestone"] = f"SURVIVED_{self.episode_steps}_STEPS"

        return reward, info

    def _reward_phase2_first_blood(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        max_boss_hp = state.get("bossMaxHealth", 400.0)
        if max_boss_hp <= 0:
            max_boss_hp = 400.0

        current_player_hp = state.get("playerHealth", 9)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        boss_recovering = state.get("primaryMantisRecovering", False)
        boss_attacking = state.get("primaryMantisActive", False)
        boss_windup = state.get(
            "primaryMantisWindUp", False
        )  # MODIFICA: Telegraph detection

        # 1. EVITARE IL CATASTROPHIC FORGETTING
        safe = boss_recovering or (not boss_attacking and not boss_windup)
        reward += self._distance_shaping(state, safe_to_approach=safe)
        reward += self._movement_quality(state)

        # Gestione Danni Subiti
        hp_lost = 0
        if hasattr(self, "prev_player_hp") and self.prev_player_hp is not None:
            hp_lost = max(0, self.prev_player_hp - current_player_hp)
            if hp_lost > 0:
                reward -= 3.5 * hp_lost
                info["took_damage"] = True
                if hazard_type == 2:
                    reward -= 2.5

        reward += self._dodge_streak_bonus(hp_lost, boss_attacking)

        # 2. SEVERE ANTI-STALLING E TELEGRAPH AWARENESS
        if self.steps_since_attack > 60:
            reward -= 0.05  # MODIFICA: Penalità decuplicata per passività
            info["stall_penalty"] = True

        if self.consecutive_idle_steps > 30:
            reward -= 0.02

        # Penalizza l'agente se attacca durante il wind-up del boss (telegraph)
        if boss_windup and self.last_action == 5:
            reward -= 0.2
            info["bad_attack_timing"] = True

        # Premia se usa la finestra di wind-up per evadere
        if boss_windup and self.last_action in [4, 6]:
            reward += 0.1

        # 3. DPS E AGGRESSIONE (EXECUTION BONUS)
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:

                # MODIFICA: Il reward per i danni scala man mano che il boss perde vita
                missing_hp_ratio = (max_boss_hp - boss_hp) / max_boss_hp
                damage_reward = (damage_dealt * 0.18) * (1.0 + (0.5 * missing_hp_ratio))

                reward += damage_reward
                info["damage_dealt"] = damage_dealt

                if boss_recovering:
                    reward += 0.35

            for threshold, bonus in [(200, 2.0), (100, 3.5), (50, 5.0)]:
                if self.prev_boss_hp > threshold >= boss_hp:
                    reward += bonus
                    info[f"threshold_{threshold}"] = True

        if boss_hp < (self.prev_boss_hp or 0):
            hp_progress = 1.0 - (boss_hp / max_boss_hp)
            reward += hp_progress * 0.05

        # 4. TERMINAL REWARDS
        if mantis_killed > self.prev_mantis_killed:
            reward += 60.0
            info["event"] = "MANTIS_KILLED"

        if done:
            if state.get("bossDefeated", False):
                reward += 30.0
                info["outcome"] = "VICTORY"
            elif is_dead:
                reward -= 3.0

        # Update per frame successivo
        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        self.prev_player_hp = current_player_hp

        return reward, info

    def _reward_phase3_dual_mantis(
        self, state: Dict, done: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        ═══════════════════════════════════════════════════════════════
        FASE 3 v2 — DUAL MANTIS (Riscritta post-analisi plateau)
        FASE 3 v2 — DUAL MANTIS (Riscritta post-analisi plateau)

        PROBLEMI RISOLTI:
        ● L'agente si fermava a 1 kill perché il reward 1v1 vs 2v1
          era sbilanciato: 60 punti per la prima kill dominavano tutto.
        ● Penalità danno troppo alta nel 2v1 → l'agente imparava a
          morire veloce piuttosto che sopravvivere.
        ● Nessun incentivo a sopravvivere nel 2v1 (tick +0.005 era
          identico al 1v1).
        ● Usava damageTaken cumulativo invece del delta HP.
        ● Mancava telegraph awareness (wind-up) per il 2v1.

        MODIFICHE:
        1. Tick sopravvivenza 5× nel 2v1 (+0.025 vs +0.005)
        2. Kill reward ribilanciato: 1st kill = 35, 2nd kill = 85
        3. Danno inflitto scala 1.5× dopo la prima kill
        4. Penalità danno ridotta nel 2v1 (-3.0 vs -4.0)
        5. Delta HP per danni subiti (come fasi precedenti)
        5. Delta HP per danni subiti (come fasi precedenti)
        6. Telegraph awareness + anti-stall
        7. Threshold bonus per HP nel 2v1
        8. Vittoria bonus aumentato a 50
        ═══════════════════════════════════════════════════════════════
        """
        reward = 0.0
        info = {}

        boss_hp = state.get("bossHealth", 0.0)
        max_boss_hp = state.get("bossMaxHealth", 700.0)
        if max_boss_hp <= 0:
            max_boss_hp = 700.0

        current_player_hp = state.get("playerHealth", 9)
        is_dead = state.get("isDead", False)
        hazard_type = state.get("lastHazardType", 0)
        mantis_killed = state.get("mantisLordsKilled", 0)
        active_count = state.get("activeMantisCount", 0)
        boss_attacking = state.get("primaryMantisActive", False)
        secondary_active = state.get("secondaryMantisActive", False)
        boss_recovering = state.get("primaryMantisRecovering", False)
        boss_windup = state.get("primaryMantisWindUp", False)

        # ─── 1. SOPRAVVIVENZA — molto più alta nel 2v1 ───
        in_dual_phase = active_count >= 2 or mantis_killed >= 1
        if in_dual_phase:
            reward += 0.025  # 5× il tick base: sopravvivere al 2v1 è prezioso
            info["dual_phase"] = True
        else:
            reward += 0.005

        # ─── 2. DISTANCE + MOVEMENT + DODGE ───
        safe = boss_recovering or (not boss_attacking and not secondary_active)
        reward += self._distance_shaping(state, safe_to_approach=safe)
        reward += self._movement_quality(state)

        # Delta HP per danni subiti (fix dal vecchio damageTaken cumulativo)
        hp_lost = 0
        if self.prev_player_hp is not None:
            hp_lost = max(0, self.prev_player_hp - current_player_hp)

        reward += self._dodge_streak_bonus(hp_lost, boss_attacking or secondary_active)

        # Dodge streak bonus amplificato nel 2v1
        if in_dual_phase and hp_lost == 0 and (boss_attacking or secondary_active):
            reward += 0.015  # Extra per schivare colpi nel 2v1

        # ─── 3. DANNI INFLITTI — scala dopo la prima kill ───
        if self.prev_boss_hp is not None:
            damage_dealt = self.prev_boss_hp - boss_hp
            if damage_dealt > 0 and damage_dealt < 500:
                # Moltiplicatore post-kill: 1.5× dopo la prima kill
                if mantis_killed >= 1:
                    damage_reward = damage_dealt * 0.27  # 0.18 * 1.5
                    info["post_kill_damage"] = True
                else:
                    damage_reward = damage_dealt * 0.18

                reward += damage_reward
                info["damage_dealt"] = damage_dealt

                # Bonus per colpire durante recovery
                if boss_recovering:
                    reward += 0.30
                    info["smart_hit"] = True

        # Threshold bonus per progressione HP nel 2v1
        if self.prev_boss_hp is not None and in_dual_phase:
            for threshold, bonus in [(500, 1.5), (350, 2.5), (200, 4.0), (100, 6.0)]:
                if self.prev_boss_hp > threshold >= boss_hp:
                    reward += bonus
                    info[f"dual_threshold_{threshold}"] = True

        # ─── 4. DANNI SUBITI — penalità ridotta nel 2v1 ───
        if hp_lost > 0:
            if in_dual_phase:
                reward -= 3.0 * hp_lost  # Ridotta da 4.0: più margine di esplorazione
            else:
                reward -= 3.5 * hp_lost
            self.total_damage_taken += hp_lost

            if hazard_type == 2:
                reward -= 2.0  # Spike damage
                info["damage_source"] = "SPIKES"

        # ─── 5. TELEGRAPH AWARENESS + ANTI-STALL ───
        if boss_windup and self.last_action == 5:
            reward -= 0.2  # Punito per attaccare durante il wind-up
            info["bad_attack_timing"] = True

        if boss_windup and self.last_action in [4, 6]:
            reward += 0.1  # Premiato per evadere durante il wind-up

        if self.steps_since_attack > 60:
            reward -= 0.04  # Anti-stall: non stare fermo senza attaccare
            info["stall_penalty"] = True

        if self.consecutive_idle_steps > 30:
            reward -= 0.015

        # ─── 6. DUAL-SPECIFIC TACTICS ───
        if in_dual_phase and secondary_active:
            # Dodge bonus amplificato quando entrambe attaccano
            if self.last_action in [4, 6]:
                reward += 0.15
                info["dual_dodge"] = True

            # Posizionamento: premia il centro dell'arena
            player_x = state.get("playerPositionX", 0.0)
            arena_center = state.get("arenaCenterX", 0.0)
            if arena_center != 0.0:
                center_dist = abs(player_x - arena_center) / 50.0
                reward += max(0, 0.03 - center_dist * 0.015)

        # ─── 7. KILL REWARDS — ribilanciati per incentivare la 2nd kill ───
        if mantis_killed > self.prev_mantis_killed:
            kills_diff = mantis_killed - self.prev_mantis_killed

            if mantis_killed == 1:
                # Prima kill: ridotta da 60 a 35 per non dominare il segnale
                reward += 35.0
                info["event"] = "FIRST_MANTIS_KILLED"
            elif mantis_killed >= 2:
                # Seconda kill: molto più alta per incentivare il proseguimento
                reward += 85.0 * kills_diff
                info["event"] = f"MANTIS_KILLED_x{mantis_killed}"

        # ─── 8. TERMINAL REWARDS ───
        if done:
            if state.get("bossDefeated", False):
                if self.phase == 4:
                    # MASTERY: victory bonus potenziato + no-hit bonus
                    reward += 100.0
                    time_bonus = max(0, (2500 - self.episode_steps) / 100.0)
                    reward += time_bonus
                    if self.total_damage_taken == 0:
                        reward += 40.0
                        info["outcome"] = "PERFECT_VICTORY"
                    elif self.total_damage_taken <= 2:
                        reward += 20.0
                        info["outcome"] = "NEAR_PERFECT"
                    else:
                        info["outcome"] = "VICTORY"
                    info["time_bonus"] = time_bonus
                else:
                    # DUAL MANTIS: reward standard
                    reward += 50.0
                    time_bonus = max(0, (3000 - self.episode_steps) / 200.0)
                    reward += time_bonus
                    info["outcome"] = "VICTORY"
                    info["time_bonus"] = time_bonus
            elif is_dead:
                if in_dual_phase:
                    reward -= 2.0
                else:
                    reward -= 3.5

        # Update per frame successivo
        self.prev_boss_hp = boss_hp
        self.prev_mantis_killed = mantis_killed
        self.prev_player_hp = current_player_hp
        return reward, info

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
