"""
═══════════════════════════════════════════════════════════════════════
  PPO TRAINING ORCHESTRATOR — Mantis Lords
  Multi-Instance, Multi-Phase with Hall of Fame

  PPO-SPECIFIC DESIGN CHOICES:
  ● On-policy: rollouts collected and discarded after each update
  ● GAE(λ) for advantage estimation — needs dense rewards (env_ppo.py)
  ● Entropy coefficient cosine decay — encourages early exploration
  ● LSTM hidden state management — reset per episode, preserve in batch
  ● Kill buffer replay — re-learn from successful kill trajectories
  ● Larger update intervals (256 steps) — PPO needs full rollouts
  ● Learning rate warmup + cosine decay via scheduler

  USAGE:
    # 2 instances on ports 5555, 5556
    python train_ppo.py --instances 2 --ports 5555 5556

    # Single instance, phase 1 only
    python train_ppo.py --phase 1 --ports 5555

    # 3 instances, full pipeline phases 1→5
    python train_ppo.py --phase 1 --instances 3 --ports 5555 5556 5557

    # Resume from phase 3 with pretrained model
    python train_ppo.py --instances 3 --start-phase 1

    python train_ppo.py --phase 1 --pretrained training_output_ppo/phase1_best.pth --instances 3 --ports 5555 5556 5557
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
import math
import json
import csv
import random
import numpy as np
import multiprocessing as mp
import filelock
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List

# ═══ Setup paths ═══
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(SRC_DIR, "agents"))
sys.path.insert(0, os.path.join(SRC_DIR, "env"))
sys.path.insert(0, os.path.join(SRC_DIR, "models"))
sys.path.insert(0, SCRIPT_DIR)

# ═══ Imports ═══
from ppo_agent import PPOAgent
from env_ppo import HollowKnightEnvPPO  # PPO-specific dense reward env
from preprocess import (
    preprocess_state_v1,
    preprocess_state_v2,
    compute_pattern_reward_bonus,
    STATE_DIM_V1,
    STATE_DIM_V2,
)


# ═══════════════════════════════════════════════════════════════
# FRAME STACKER
# ═══════════════════════════════════════════════════════════════


class FrameStacker:
    """Stack N consecutive frames. State [F] → [F × N]."""

    def __init__(self, stack_size: int, state_dim: int):
        self.stack_size = stack_size
        self.state_dim = state_dim
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_state)
        return np.concatenate(list(self.frames))

    def step(self, state: np.ndarray) -> np.ndarray:
        self.frames.append(state)
        return np.concatenate(list(self.frames))


# ═══════════════════════════════════════════════════════════════
# HALL OF FAME — SHARED STATE (FILE-LOCKED)
# ═══════════════════════════════════════════════════════════════


class HallOfFame:
    """
    Shared state across instances: maintains Top-K best models.
    Thread/process-safe via file locking.
    """

    def __init__(self, checkpoint_dir: str, keep_top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k

        self.models_dir = os.path.join(checkpoint_dir, "best_pool")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.state_file = os.path.join(checkpoint_dir, "shared_state.json")
        self.lock_file = os.path.join(checkpoint_dir, "shared_state.lock")
        self.log_file = os.path.join(checkpoint_dir, "training_log_ppo.csv")

        if not os.path.exists(self.state_file):
            self._write_state(
                {
                    "best_models": [],
                    "total_episodes": 0,
                    "global_best_reward": -float("inf"),
                    "agent_type": "ppo",
                }
            )

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "instance_id",
                        "phase",
                        "episode",
                        "reward",
                        "steps",
                        "mantis_killed",
                        "boss_hp",
                        "boss_defeated",
                        "entropy",
                        "learning_rate",
                        "num_updates",
                    ]
                )

    def _read_state(self) -> dict:
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "best_models": [],
                "total_episodes": 0,
                "global_best_reward": -float("inf"),
            }

    def _write_state(self, state: dict):
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def update_best_model(
        self, instance_id: int, reward: float, model_path: str
    ) -> bool:
        """Try to insert model into Hall of Fame. Returns True if accepted."""
        lock = filelock.FileLock(self.lock_file, timeout=15)
        with lock:
            state = self._read_state()
            best_models = state.get("best_models", [])

            # Each instance occupies at most 1 slot (diversification)
            existing_idx = None
            existing_reward = -float("inf")
            for i, entry in enumerate(best_models):
                if entry.get("instance_id") == instance_id:
                    existing_idx = i
                    existing_reward = entry.get("reward", -float("inf"))
                    break

            MIN_IMPROVEMENT = 0.5
            if existing_idx is not None:
                if reward < existing_reward + MIN_IMPROVEMENT:
                    return False
                old_entry = best_models.pop(existing_idx)
                try:
                    if os.path.exists(old_entry.get("path", "")):
                        os.remove(old_entry["path"])
                except Exception:
                    pass

            insert_idx = len(best_models)
            for i, entry in enumerate(best_models):
                if reward > entry.get("reward", -float("inf")):
                    insert_idx = i
                    break

            if insert_idx >= self.keep_top_k:
                return False

            import shutil

            pool_filename = f"hof_ppo_inst{instance_id}.pth"
            pool_path = os.path.join(self.models_dir, pool_filename)
            try:
                shutil.copy2(model_path, pool_path)
            except Exception as e:
                print(f"  [HoF-PPO] Copy failed: {e}")
                return False

            new_entry = {
                "instance_id": instance_id,
                "reward": reward,
                "path": pool_path,
                "timestamp": datetime.now().isoformat(),
            }
            best_models.insert(insert_idx, new_entry)

            while len(best_models) > self.keep_top_k:
                removed = best_models.pop()
                try:
                    if os.path.exists(removed.get("path", "")):
                        os.remove(removed["path"])
                except Exception:
                    pass

            state["best_models"] = best_models
            if reward > state.get("global_best_reward", -float("inf")):
                state["global_best_reward"] = reward
            self._write_state(state)

            rank = insert_idx + 1
            print(f"\n  {'★'*50}")
            print(
                f"  ★ [PPO Inst {instance_id}] HALL OF FAME! R={reward:.2f} (Rank {rank}/{self.keep_top_k})"
            )
            print(f"  {'★'*50}\n")
            return True

    def get_random_best_model_path(self, exclude_instance: int = -1) -> Optional[str]:
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            best_models = state.get("best_models", [])
            candidates = [
                m for m in best_models if m.get("instance_id") != exclude_instance
            ]
            if not candidates:
                candidates = best_models
            if not candidates:
                return None
            chosen = random.choice(candidates)
            path = chosen.get("path", "")
            return path if os.path.exists(path) else None

    def get_global_best_reward(self) -> float:
        state = self._read_state()
        return state.get("global_best_reward", -float("inf"))

    def increment_episodes(self) -> int:
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            state["total_episodes"] = state.get("total_episodes", 0) + 1
            self._write_state(state)
            return state["total_episodes"]

    def get_last_episode(self, instance_id: int, phase: int) -> int:
        """Legge il log CSV e restituisce l'ultimo numero di episodio
        per questa istanza/fase. Se non trova nulla, restituisce 0."""
        lock = filelock.FileLock(self.lock_file, timeout=5)
        last_ep = 0
        try:
            with lock:
                if not os.path.exists(self.log_file):
                    return 0
                with open(self.log_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get("instance_id", "")) == str(instance_id) and str(
                            row.get("phase", "")
                        ) == str(phase):
                            try:
                                ep = int(row["episode"])
                                if ep > last_ep:
                                    last_ep = ep
                            except (ValueError, KeyError):
                                pass
        except Exception:
            pass
        return last_ep

    def log_episode(
        self,
        instance_id,
        phase,
        episode,
        reward,
        steps,
        mantis_killed,
        boss_hp,
        boss_defeated,
        entropy,
        lr,
        updates,
    ):
        lock = filelock.FileLock(self.lock_file, timeout=5)
        try:
            with lock:
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().strftime("%H:%M:%S"),
                            instance_id,
                            phase,
                            episode,
                            f"{reward:.2f}",
                            steps,
                            mantis_killed,
                            f"{boss_hp:.0f}",
                            1 if boss_defeated else 0,
                            f"{entropy:.4f}",
                            f"{lr:.2e}",
                            updates,
                        ]
                    )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# PPO-SPECIFIC PHASE CONFIGS
#
# KEY DIFFERENCES FROM DQN:
# ● Higher learning rates (PPO can handle 3e-4, DQN needs 1e-4)
# ● Entropy-based exploration (cosine decay) instead of ε-greedy
# ● LR end factor for cosine LR scheduling
# ● Same gamma/phase structure to keep comparison fair
# ═══════════════════════════════════════════════════════════════

PHASE_CONFIGS = {
    1: {
        "name": "SURVIVE",
        "description": "Learn to dodge and not die",
        "episodes": 1000,
        "lr": 3e-4,
        "lr_end_factor": 0.3,
        "entropy_start": 0.08,
        "entropy_end": 0.03,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 256,
        "gamma": 0.99,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_survival_steps >= 850",
        "promotion_avg_window": 25,
    },
    2: {
        "name": "FIRST BLOOD",
        "description": "Deal damage and kill the first mantis",
        "episodes": 1500,
        "lr": 1.5e-4,
        "lr_end_factor": 0.2,
        "entropy_start": 0.05,
        "entropy_end": 0.005,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 256,
        "gamma": 0.995,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 0.8",
        "promotion_avg_window": 30,
    },
    3: {
        "name": "DUAL MANTIS",
        "description": "Handle two mantises at once",
        "episodes": 2000,
        "lr": 5e-5,
        "lr_end_factor": 0.25,
        "entropy_start": 0.03,
        "entropy_end": 0.005,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 384,
        "gamma": 0.995,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 1.5",
        "promotion_avg_window": 40,
    },
    4: {
        "name": "MASTERY",
        "description": "Full victory, optimize time and no-hit",
        "episodes": 1500,
        "lr": 3e-5,
        "lr_end_factor": 0.25,
        "entropy_start": 0.02,
        "entropy_end": 0.003,
        "gae_lambda": 0.97,
        "n_epochs": 6,
        "batch_size": 64,
        "update_interval": 512,
        "gamma": 0.998,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 2.5",
        "promotion_avg_window": 50,
    },
}


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════


def cosine_entropy_decay(episode, total_episodes, start, end):
    """PPO-SPECIFIC: Cosine annealing for entropy coefficient."""
    progress = episode / max(total_episodes, 1)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def check_promotion(
    episode_steps_history, episode_kills, episode_damage, wins, total_episodes, config
):
    window = config["promotion_avg_window"]
    if len(episode_steps_history) < window:
        return False

    avg_survival_steps = np.mean(list(episode_steps_history)[-window:])
    avg_mantis_killed = np.mean(list(episode_kills)[-window:])
    avg_damage_dealt = np.mean(list(episode_damage)[-window:])
    win_rate = wins / max(total_episodes, 1)

    context = {
        "avg_survival_steps": avg_survival_steps,
        "avg_mantis_killed": avg_mantis_killed,
        "avg_damage_dealt": avg_damage_dealt,
        "win_rate": win_rate,
    }
    try:
        return eval(config["promotion_condition"], {"__builtins__": {}}, context)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# PPO TRAINING WORKER
#
# KEY PPO TRAINING CHARACTERISTICS:
# ● Collect rollout of `update_interval` steps → run PPO update
# ● On-policy: buffer is cleared after each update
# ● LSTM hidden state: reset per episode, preserved within episode
# ● Entropy coefficient decayed via cosine schedule (not ε-greedy)
# ● Kill buffer: store & replay successful kill trajectories
# ● GAE(λ) advantage estimation works best with dense rewards
#   from env_ppo.py
# ═══════════════════════════════════════════════════════════════


def train_ppo_instance(
    instance_id: int,
    port: int,
    phase: int,
    checkpoint_dir: str,
    pretrained_path: Optional[str] = None,
    auto_promote: bool = True,
    sync_interval: int = 15,
    max_steps: int = 5000,
):
    """Worker function for PPO training on a single instance."""
    cfg = PHASE_CONFIGS[phase]
    phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
    instance_dir = os.path.join(phase_dir, f"instance_{instance_id}")
    os.makedirs(instance_dir, exist_ok=True)

    print(f"\n[PPO Inst {instance_id}] Phase {phase}: {cfg['name']} | Port {port}")

    hof = HallOfFame(phase_dir, keep_top_k=3)

    # ═══ PPO ENV: Dense reward environment ═══
    try:
        env = HollowKnightEnvPPO(
            host="localhost", port=port, phase=phase, reward_scale=5.0
        )
    except Exception as e:
        print(f"[PPO Inst {instance_id}] Connection failed: {e}")
        return None

    # Preprocessing (identical to DQN)
    STACK_SIZE = 4
    version = cfg["preprocess_version"]
    raw_dim = STATE_DIM_V2 if version == 2 else STATE_DIM_V1
    stacked_dim = raw_dim * STACK_SIZE
    preprocess_fn = preprocess_state_v2 if version == 2 else preprocess_state_v1
    stacker = FrameStacker(STACK_SIZE, raw_dim)

    print(f"[PPO Inst {instance_id}] State: {raw_dim} × {STACK_SIZE} = {stacked_dim}")

    # ═══ PPO AGENT: On-policy with LSTM ═══
    agent = PPOAgent(
        state_size=stacked_dim,
        action_size=8,  # Same action space as DQN
        learning_rate=cfg["lr"],
        lr_end_factor=cfg["lr_end_factor"],
        total_episodes=cfg["episodes"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],  # PPO-specific: GAE lambda
        entropy_coef=cfg["entropy_start"],  # PPO-specific: entropy bonus
        use_lstm=False,  # PPO-specific: recurrent policy
        n_epochs=cfg["n_epochs"],  # PPO-specific: epochs per update
        batch_size=cfg["batch_size"],
    )

    # Load pretrained
    loaded = False
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            agent.load(pretrained_path)
            print(f"[PPO Inst {instance_id}] ✓ Loaded pretrained: {pretrained_path}")
            loaded = True
        except Exception as e:
            print(f"[PPO Inst {instance_id}] ⚠ Pretrained load failed: {e}")

    if not loaded:
        pool_path = hof.get_random_best_model_path(exclude_instance=instance_id)
        if pool_path:
            try:
                agent.load(pool_path)
                print(f"[PPO Inst {instance_id}] ✓ Loaded from Hall of Fame")
            except Exception:
                pass

    # Tracking
    best_reward = -float("inf")
    best_model_path = None
    episode_rewards = deque(maxlen=200)
    episode_steps_hist = deque(maxlen=200)
    episode_kills = deque(maxlen=200)
    episode_damage = deque(maxlen=200)
    wins = 0
    total_ep = 0
    update_interval = cfg["update_interval"]  # PPO-specific: rollout length

    # ═══ RESUME: offset episodi dal log precedente ═══
    episode_offset = hof.get_last_episode(instance_id, phase)
    if episode_offset > 0:
        print(
            f"[PPO Inst {instance_id}] ✓ Ripresa dal log: ultimo ep={episode_offset}, i nuovi partiranno da {episode_offset + 1}"
        )

    # ═══ TRAINING LOOP ═══
    for episode in range(cfg["episodes"]):
        # ═══ PPO-SPECIFIC: Cosine entropy decay ═══
        current_entropy = cosine_entropy_decay(
            episode, cfg["episodes"], cfg["entropy_start"], cfg["entropy_end"]
        )
        agent.entropy_coef = current_entropy

        raw_state_dict = env.reset()
        raw_state = preprocess_fn(raw_state_dict)
        state = stacker.reset(raw_state)
        agent.reset_hidden()  # PPO-SPECIFIC: Reset LSTM state per episode

        episode_reward = 0.0
        num_updates = 0
        ep_boss_hp_start = raw_state_dict.get("bossHealth", 100.0)
        ep_transitions = []

        for step in range(max_steps):
            # ═══ PPO-SPECIFIC: Action selection returns (action, log_prob, value) ═══
            action, log_prob, value = agent.select_action(state)
            next_state_dict, reward, done, info = env.step(action)

            if cfg["use_pattern_bonus"]:
                reward += compute_pattern_reward_bonus(next_state_dict, action)

            next_raw = preprocess_fn(next_state_dict)
            next_state = stacker.step(next_raw)

            episode_reward += reward

            # ═══ PPO-SPECIFIC: Store (state, action, log_prob, value, reward, done) ═══
            # PPO needs log_prob and value for advantage computation
            agent.store_transition(state, action, log_prob, value, reward, done)
            ep_transitions.append((state.copy(), action, reward, done))
            state = next_state

            # ═══ PPO-SPECIFIC: Update after collecting rollout ═══
            # PPO collects `update_interval` steps then runs multiple epochs
            if len(agent.buffer) >= update_interval:
                metrics = agent.learn()
                if metrics:
                    num_updates += 1

            if done:
                break

        # ═══ PPO-SPECIFIC: Final learn on remaining buffer ═══
        if len(agent.buffer) > 0:
            metrics = agent.learn()
            if metrics:
                num_updates += 1

        # ═══ PPO-SPECIFIC: Kill replay — re-learn from kill trajectories ═══
        # MODIFICA: replay più aggressivo, specialmente per multi-kill
        ep_mantis_killed = next_state_dict.get("mantisLordsKilled", 0)
        if len(agent.kill_buffer) >= 3 and num_updates > 0:
            agent.learn_from_kills()
            # Extra replay per episodi 2+ kill (la risorsa più preziosa)
            if ep_mantis_killed >= 2:
                agent.learn_from_kills()
                agent.learn_from_kills()
                print(f"  [PPO {instance_id}] ★ EXTRA REPLAY per 2-kill episode")

        if num_updates > 0:
            agent.step_scheduler()  # PPO-SPECIFIC: LR scheduler step

        # Post-episode tracking
        total_ep += 1
        mantis_killed = next_state_dict.get("mantisLordsKilled", 0)
        boss_hp_end = next_state_dict.get("bossHealth", 0)
        boss_defeated = next_state_dict.get("bossDefeated", False)
        damage_dealt = max(0, ep_boss_hp_start - boss_hp_end)

        episode_rewards.append(episode_reward)
        episode_steps_hist.append(step + 1)
        episode_kills.append(mantis_killed)
        episode_damage.append(damage_dealt)
        if boss_defeated:
            wins += 1

        # Kill buffer: save kill & near-kill trajectories
        # MODIFICA: 2+ kill episodes salvati sempre; 1-kill con soglia minima
        if mantis_killed >= 2:
            # Multi-kill: salva SEMPRE, è la risorsa più preziosa
            if len(ep_transitions) >= 10:
                states_t, actions_t, rewards_t, dones_t = zip(*ep_transitions)
                # Salva 2 volte per sovrappesare nel buffer
                agent.kill_buffer.add_episode(states_t, actions_t, rewards_t, dones_t)
                agent.kill_buffer.add_episode(states_t, actions_t, rewards_t, dones_t)
                print(
                    f"  [PPO {instance_id}] ★★ MULTI-KILL x{mantis_killed} saved 2× to buffer ({len(agent.kill_buffer)} stored)"
                )
        elif mantis_killed >= 1 or boss_hp_end <= 50:
            if len(ep_transitions) >= 10:
                states_t, actions_t, rewards_t, dones_t = zip(*ep_transitions)
                agent.kill_buffer.add_episode(states_t, actions_t, rewards_t, dones_t)
                label = (
                    "KILL"
                    if mantis_killed >= 1
                    else f"NEAR-KILL (HP={boss_hp_end:.0f})"
                )
                print(
                    f"  [PPO {instance_id}] → {label} saved to buffer ({len(agent.kill_buffer)} stored)"
                )

        # Save checkpoints
        latest_path = os.path.join(instance_dir, "latest.pth")
        agent.save(latest_path)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(instance_dir, "best.pth")
            agent.save(best_model_path)
            hof.update_best_model(instance_id, episode_reward, best_model_path)

        if (episode + 1) % 100 == 0:
            agent.save(
                os.path.join(
                    instance_dir, f"checkpoint_ep{episode_offset + episode + 1}.pth"
                )
            )

        # Sync with Hall of Fame (conservative)
        if (episode + 1) % (sync_interval * 3) == 0 and len(episode_rewards) >= 20:
            my_avg = np.mean(list(episode_rewards)[-20:])
            global_best = hof.get_global_best_reward()
            if my_avg < global_best * 0.3 and global_best > 0:
                sync_path = hof.get_random_best_model_path(exclude_instance=instance_id)
                if sync_path:
                    try:
                        agent.load(sync_path)
                        print(
                            f"  [PPO {instance_id}] Synced with HoF (avg={my_avg:.1f} vs best={global_best:.1f})"
                        )
                    except Exception:
                        pass

        # Log
        lr = agent.get_current_lr()
        ep_num = episode_offset + episode + 1  # Continua dal log precedente
        hof.log_episode(
            instance_id,
            phase,
            ep_num,
            episode_reward,
            step + 1,
            mantis_killed,
            boss_hp_end,
            boss_defeated,
            current_entropy,
            lr,
            num_updates,
        )
        hof.increment_episodes()

        kb = len(agent.kill_buffer)
        print(
            f"  [PPO {instance_id}] P{phase} Ep {ep_num:>4}/{episode_offset + cfg['episodes']} | "
            f"R={episode_reward:>+7.2f} | Steps={step+1:>4} | "
            f"HP={boss_hp_end:>5.0f} | K={mantis_killed} | "
            f"Ent={current_entropy:.3f} | LR={lr:.1e} | KB={kb}"
        )

        if boss_defeated:
            wr = wins / total_ep
            print(f"\n  {'★'*20}  [PPO {instance_id}] VICTORY!  {'★'*20}")
            print(
                f"  P{phase} Ep {ep_num} | R={episode_reward:+.2f} | Wins={wins} WR={wr:.0%}"
            )
            print(f"  {'★'*52}\n")
        elif mantis_killed >= 2:
            print(f"  >>>> [PPO {instance_id}] KILL x{mantis_killed}! Ep {ep_num} <<<<")

        # Promotion check (usa episode locale, non offset)
        if auto_promote and (episode + 1) >= cfg.get("promotion_avg_window", 20):
            if check_promotion(
                episode_steps_hist, episode_kills, episode_damage, wins, total_ep, cfg
            ):
                print(
                    f"\n  ▲▲▲ [PPO {instance_id}] PROMOTED! Phase {phase} complete! ▲▲▲"
                )
                break

    env.close()
    print(
        f"[PPO Inst {instance_id}] Phase {phase} DONE | Best R={best_reward:+.2f} | Wins={wins}/{total_ep}"
    )
    return best_model_path


# ═══════════════════════════════════════════════════════════════
# MULTI-INSTANCE LAUNCHER
# ═══════════════════════════════════════════════════════════════


def run_phase_multi_instance(
    phase: int,
    ports: List[int],
    checkpoint_dir: str,
    pretrained_path: Optional[str] = None,
    auto_promote: bool = True,
    sync_interval: int = 15,
):
    cfg = PHASE_CONFIGS[phase]
    n_instances = len(ports)

    print(f"\n{'═'*70}")
    print(f"  PHASE {phase}: {cfg['name']} | Agent: PPO | Instances: {n_instances}")
    print(f"  Ports: {ports}")
    print(f"  Episodes: {cfg['episodes']} | Promotion: {cfg['promotion_condition']}")
    print(f"{'═'*70}\n")

    if n_instances == 1:
        return train_ppo_instance(
            instance_id=0,
            port=ports[0],
            phase=phase,
            checkpoint_dir=checkpoint_dir,
            pretrained_path=pretrained_path,
            auto_promote=auto_promote,
            sync_interval=sync_interval,
        )
    else:
        processes = []
        for i, port in enumerate(ports):
            p = mp.Process(
                target=train_ppo_instance,
                kwargs={
                    "instance_id": i,
                    "port": port,
                    "phase": phase,
                    "checkpoint_dir": checkpoint_dir,
                    "pretrained_path": pretrained_path,
                    "auto_promote": auto_promote,
                    "sync_interval": sync_interval,
                },
            )
            p.start()
            processes.append(p)
            time.sleep(1)

        for p in processes:
            p.join()

        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
        hof = HallOfFame(phase_dir, keep_top_k=3)
        best_path = hof.get_random_best_model_path()

        if not best_path:
            for i in range(n_instances):
                candidate = os.path.join(phase_dir, f"instance_{i}", "best.pth")
                if os.path.exists(candidate):
                    best_path = candidate
                    break

        return best_path


# ═══════════════════════════════════════════════════════════════
# MULTI-PHASE PIPELINE
# ═══════════════════════════════════════════════════════════════


def run_all_phases(
    ports: List[int],
    base_dir: str,
    start_phase: int = 1,
    end_phase: int = 4,
    pretrained_path: Optional[str] = None,
    auto_promote: bool = True,
    sync_interval: int = 15,
):
    print(f"\n{'═'*70}")
    print(f"  PPO MULTI-PHASE PIPELINE — MANTIS LORDS")
    print(f"  Instances: {len(ports)} | Phases: {start_phase}→{end_phase}")
    print(f"  Ports: {ports}")
    print(f"{'═'*70}\n")

    os.makedirs(base_dir, exist_ok=True)
    current_model = pretrained_path
    n_instances = len(ports)

    for phase in range(start_phase, end_phase + 1):
        if phase not in PHASE_CONFIGS:
            print(f"[ERROR] Phase {phase} not configured!")
            break

        best_model = run_phase_multi_instance(
            phase=phase,
            ports=ports,
            checkpoint_dir=base_dir,
            pretrained_path=current_model,
            auto_promote=auto_promote,
            sync_interval=sync_interval,
        )

        if best_model and os.path.exists(best_model):
            current_model = best_model
            print(f"\n[PPO Pipeline] Phase {phase} → best model: {best_model}")
        else:
            print(f"\n[PPO Pipeline] ⚠ Phase {phase} produced no model. Stopping.")
            break

        # Eleggi il champion per questa fase
        select_champion(base_dir, phase=phase, n_instances=n_instances)

        print(f"\n{'─'*70}")
        print(f"  Pausing 5s before next phase...")
        print(f"{'─'*70}\n")
        time.sleep(5)

    # Raccogli tutti i champion in un'unica cartella
    collect_all_champions(base_dir, n_instances=n_instances)

    print(f"\n{'═'*70}")
    print(f"  PPO TRAINING PIPELINE COMPLETE")
    print(f"  Final model: {current_model}")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════
# CHAMPION SYSTEM — Seleziona il miglior modello dopo la Fase 5
#
# Dopo ogni run della Fase 5:
#  1. Confronta i best.pth di tutte le istanze
#  2. Elegge il "champion" della run corrente
#  3. Se esiste un champion precedente, li confronta
#  4. Salva il vincitore come champion.pth + champion.json
# ═══════════════════════════════════════════════════════════════


def select_champion(
    checkpoint_dir: str, phase: int = 4, n_instances: int = 3
) -> Optional[str]:
    """
    Trova il miglior modello tra tutte le istanze di una fase.
    Restituisce il percorso del champion.pth salvato.
    Funziona per QUALSIASI fase, non solo la 5.
    """
    import shutil

    phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
    champion_dir = os.path.join(checkpoint_dir, "champion")
    os.makedirs(champion_dir, exist_ok=True)

    phase_name = PHASE_CONFIGS.get(phase, {}).get("name", f"PHASE_{phase}")
    champion_model = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
    champion_meta = os.path.join(champion_dir, f"phase_{phase}_champion.json")
    history_file = os.path.join(champion_dir, f"phase_{phase}_history.json")

    # ─── 1. Trova il best di ogni istanza dalla HoF ───
    candidates = []

    # Prova dalla shared_state.json (Hall of Fame)
    state_file = os.path.join(phase_dir, "shared_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            for entry in state.get("best_models", []):
                path = entry.get("path", "")
                reward = entry.get("reward", -float("inf"))
                inst_id = entry.get("instance_id", -1)
                if os.path.exists(path):
                    candidates.append(
                        {
                            "path": path,
                            "reward": reward,
                            "instance_id": inst_id,
                            "source": "hall_of_fame",
                        }
                    )
        except Exception:
            pass

    # Fallback: controlla i best.pth di ogni istanza
    for i in range(n_instances):
        best_path = os.path.join(phase_dir, f"instance_{i}", "best.pth")
        if os.path.exists(best_path):
            already = any(c["instance_id"] == i for c in candidates)
            if not already:
                candidates.append(
                    {
                        "path": best_path,
                        "reward": -1.0,
                        "instance_id": i,
                        "source": "instance_best",
                    }
                )

    if not candidates:
        print(f"\n  [Champion] Nessun modello trovato in {phase_dir}")
        return None

    # ─── 2. Seleziona il migliore della run corrente ───
    candidates.sort(key=lambda c: c["reward"], reverse=True)
    new_champion = candidates[0]

    print(f"\n{'═'*60}")
    print(f"  CHAMPION SELECTION — Phase {phase}: {phase_name}")
    print(f"{'═'*60}")
    print(f"  Candidati trovati: {len(candidates)}")
    for i, c in enumerate(candidates):
        marker = " ◄ BEST" if i == 0 else ""
        print(
            f"    Inst {c['instance_id']}: R={c['reward']:.2f} ({c['source']}){marker}"
        )

    # ─── 3. Confronta con champion precedente ───
    old_champion = None
    if os.path.exists(champion_meta):
        try:
            with open(champion_meta, "r") as f:
                old_champion = json.load(f)
            print(
                f"\n  Champion precedente: R={old_champion.get('reward', '?'):.2f} "
                f"(Inst {old_champion.get('instance_id', '?')}, "
                f"run {old_champion.get('run_id', '?')})"
            )
        except Exception:
            old_champion = None

    # Determina il run_id
    run_id = 1
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            run_id = len(history) + 1
        except Exception:
            pass

    # ─── 4. Confronto e salvataggio ───
    save_new = False
    if old_champion is None:
        save_new = True
        print(f"\n  Nessun champion precedente → il nuovo diventa champion")
    elif new_champion["reward"] > old_champion.get("reward", -float("inf")):
        save_new = True
        improvement = new_champion["reward"] - old_champion.get("reward", 0)
        print(
            f"\n  ★ NUOVO CHAMPION! R={new_champion['reward']:.2f} > "
            f"R={old_champion.get('reward', 0):.2f} (+{improvement:.2f})"
        )
    else:
        print(
            f"\n  Champion precedente confermato (R={old_champion.get('reward', 0):.2f} >= "
            f"R={new_champion['reward']:.2f})"
        )
        print(f"  Il modello phase_{phase}_champion.pth resta invariato.")

    if save_new:
        try:
            shutil.copy2(new_champion["path"], champion_model)
        except Exception as e:
            print(f"  [Champion] Errore copia: {e}")
            return None

        meta = {
            "reward": new_champion["reward"],
            "instance_id": new_champion["instance_id"],
            "source": new_champion["source"],
            "source_path": new_champion["path"],
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "phase_name": phase_name,
        }
        with open(champion_meta, "w") as f:
            json.dump(meta, f, indent=2)

    # ─── 5. Aggiorna la history ───
    history_entry = {
        "run_id": run_id,
        "reward": new_champion["reward"],
        "instance_id": new_champion["instance_id"],
        "is_new_champion": save_new,
        "timestamp": datetime.now().isoformat(),
    }

    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(history_entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # ─── 6. Stampa riepilogo ───
    print(f"\n  Champion Phase {phase}: {champion_model}")
    if len(history) > 1:
        print(f"\n  History ({len(history)} run):")
        for h in history:
            marker = " ★ CHAMPION" if h["is_new_champion"] else ""
            print(
                f"    Run {h['run_id']}: R={h['reward']:.2f} (Inst {h['instance_id']}){marker}"
            )

    print(f"{'═'*60}\n")
    return champion_model


def collect_all_champions(checkpoint_dir: str, n_instances: int = 3):
    """
    Raccoglie i champion di tutte le fasi in un'unica cartella.
    Scansiona quali fasi hanno dati e crea/aggiorna il champion per ognuna.
    Stampa un riepilogo con i comandi play pronti.
    """
    # Check se esistono cartelle di fasi
    existing_phases = []
    for phase in range(1, 5):
        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
        if os.path.exists(phase_dir):
            existing_phases.append(phase)

    if not existing_phases:
        return  # Nessuna fase allenata, niente da fare

    champion_dir = os.path.join(checkpoint_dir, "champion")
    os.makedirs(champion_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  RACCOLTA CHAMPION — Tutte le fasi")
    print(f"{'═'*60}")

    found = []
    for phase in existing_phases:
        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")

        # Se non esiste ancora il champion per questa fase, crealo
        champion_file = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        champion_meta = os.path.join(champion_dir, f"phase_{phase}_champion.json")

        if not os.path.exists(champion_file):
            # Eleggi il champion per questa fase
            select_champion(checkpoint_dir, phase=phase, n_instances=n_instances)

        # Leggi metadata se esiste
        if os.path.exists(champion_meta):
            try:
                with open(champion_meta, "r") as f:
                    meta = json.load(f)
                found.append(meta)
            except Exception:
                found.append({"phase": phase, "reward": "?", "phase_name": "?"})
        elif os.path.exists(champion_file):
            found.append({"phase": phase, "reward": "?", "phase_name": "?"})

    if not found:
        print(f"\n  Nessun champion trovato.")
        return

    print(f"\n  Champion disponibili:")
    print(f"  {'─'*56}")
    for meta in found:
        phase = meta.get("phase", "?")
        name = meta.get("phase_name", PHASE_CONFIGS.get(phase, {}).get("name", "?"))
        reward = meta.get("reward", "?")
        inst = meta.get("instance_id", "?")
        runs = meta.get("run_id", "?")
        reward_str = (
            f"R={reward:.2f}" if isinstance(reward, (int, float)) else f"R={reward}"
        )
        champ_path = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        print(
            f"    Phase {phase} ({name:>12}): {reward_str} | Inst {inst} | run {runs}"
        )
        print(f"      → python play_ppo.py --model {champ_path}")

    # Segnala il champion globale (fase 5 se esiste, altrimenti la più alta)
    global_champ = None
    for phase in [4, 3, 2, 1]:
        candidate = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        if os.path.exists(candidate):
            global_champ = candidate
            break
    if not os.path.exists(global_champ):
        # Prendi la fase più alta disponibile
        for phase in [5, 4, 3, 2, 1]:
            candidate = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
            if os.path.exists(candidate):
                global_champ = candidate
                break

    print(f"\n  {'─'*56}")
    print(f"  Miglior modello complessivo:")
    print(f"    python play_ppo.py --model {global_champ}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="PPO Training for Mantis Lords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppo.py --instances 2 --ports 5555 5556
  python train_ppo.py --phase 1 --ports 5555
  python train_ppo.py --instances 3 --ports 5555 5556 5557
  python train_ppo.py --start-phase 3 --pretrained best.pth
        """,
    )

    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--ports", type=int, nargs="+", default=[5555])
    parser.add_argument("--start-phase", type=int, default=1)
    parser.add_argument("--end-phase", type=int, default=4)
    parser.add_argument("--phase", type=int, default=None, help="Run ONLY this phase")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output", type=str, default="training_output_ppo")
    parser.add_argument("--no-auto-promote", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override episodes per phase"
    )
    parser.add_argument("--sync-interval", type=int, default=15)

    args = parser.parse_args()

    if args.episodes:
        for pid in PHASE_CONFIGS:
            PHASE_CONFIGS[pid]["episodes"] = args.episodes

    ports = list(args.ports)
    while len(ports) < args.instances:
        ports.append(ports[-1] + 1)
    ports = ports[: args.instances]

    print(f"\n  Agent: PPO | Instances: {args.instances} | Ports: {ports}")

    # ═══ Prima di tutto: raccogli i champion da tutte le fasi già esistenti ═══
    collect_all_champions(
        checkpoint_dir=args.output,
        n_instances=args.instances,
    )

    if args.phase:
        run_phase_multi_instance(
            phase=args.phase,
            ports=ports,
            checkpoint_dir=args.output,
            pretrained_path=args.pretrained,
            auto_promote=not args.no_auto_promote,
            sync_interval=args.sync_interval,
        )

        # Eleggi il champion per la fase appena completata
        select_champion(
            checkpoint_dir=args.output,
            phase=args.phase,
            n_instances=args.instances,
        )

        # Raccogli tutti i champion disponibili (anche di fasi precedenti)
        collect_all_champions(
            checkpoint_dir=args.output,
            n_instances=args.instances,
        )
    else:
        run_all_phases(
            ports=ports,
            base_dir=args.output,
            start_phase=args.start_phase,
            end_phase=args.end_phase,
            pretrained_path=args.pretrained,
            auto_promote=not args.no_auto_promote,
            sync_interval=args.sync_interval,
        )
