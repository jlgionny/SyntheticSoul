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
    python train_ppo.py --instances 3 --ports 5555 5556 5557

    # Resume from phase 3 with pretrained model
    python train_ppo.py --start-phase 3 --pretrained best.pth
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
            self._write_state({
                "best_models": [],
                "total_episodes": 0,
                "global_best_reward": -float("inf"),
                "agent_type": "ppo",
            })

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'instance_id', 'phase', 'episode', 'reward',
                    'steps', 'mantis_killed', 'boss_hp', 'boss_defeated',
                    'entropy', 'learning_rate', 'num_updates'
                ])

    def _read_state(self) -> dict:
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"best_models": [], "total_episodes": 0, "global_best_reward": -float("inf")}

    def _write_state(self, state: dict):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def update_best_model(self, instance_id: int, reward: float, model_path: str) -> bool:
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
            print(f"  ★ [PPO Inst {instance_id}] HALL OF FAME! R={reward:.2f} (Rank {rank}/{self.keep_top_k})")
            print(f"  {'★'*50}\n")
            return True

    def get_random_best_model_path(self, exclude_instance: int = -1) -> Optional[str]:
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            best_models = state.get("best_models", [])
            candidates = [m for m in best_models if m.get("instance_id") != exclude_instance]
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

    def log_episode(self, instance_id, phase, episode, reward, steps,
                    mantis_killed, boss_hp, boss_defeated, entropy, lr, updates):
        lock = filelock.FileLock(self.lock_file, timeout=5)
        try:
            with lock:
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%H:%M:%S"),
                        instance_id, phase, episode, f"{reward:.2f}",
                        steps, mantis_killed, f"{boss_hp:.0f}",
                        1 if boss_defeated else 0,
                        f"{entropy:.4f}", f"{lr:.2e}", updates
                    ])
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
        "name": "SURVIVAL",
        "description": "Learn to dodge and not die",
        "episodes": 400,
        # ═══ PPO-SPECIFIC hyperparameters ═══
        "lr": 3e-4,                     # PPO tolerates higher LR
        "lr_end_factor": 0.3,           # Cosine decay to 30% of initial
        "entropy_start": 0.08,          # High initial exploration
        "entropy_end": 0.03,            # Maintain some exploration
        "gae_lambda": 0.95,             # Standard GAE λ
        "n_epochs": 4,                  # PPO epochs per update
        "batch_size": 64,               # Mini-batch size
        "update_interval": 256,         # Steps between PPO updates
        # Common
        "gamma": 0.99,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_survival_steps >= 700",
        "promotion_avg_window": 20,
    },
    2: {
        "name": "FIRST HITS",
        "description": "Learn to punish during recovery windows",
        "episodes": 600,
        "lr": 2e-4,
        "lr_end_factor": 0.3,
        "entropy_start": 0.06,
        "entropy_end": 0.02,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 256,
        "gamma": 0.995,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_damage_dealt >= 200",
        "promotion_avg_window": 25,
    },
    3: {
        "name": "AGGRESSION",
        "description": "Kill the first mantis",
        "episodes": 600,
        "lr": 1.5e-4,
        "lr_end_factor": 1.0,           # No LR decay in aggression phase
        "entropy_start": 0.05,
        "entropy_end": 0.015,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 256,
        "gamma": 0.995,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 0.5",
        "promotion_avg_window": 30,
    },
    4: {
        "name": "DUAL MANTIS",
        "description": "Handle two mantises at once",
        "episodes": 700,
        "lr": 1e-4,
        "lr_end_factor": 0.2,
        "entropy_start": 0.04,
        "entropy_end": 0.01,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "batch_size": 64,
        "update_interval": 256,
        "gamma": 0.995,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 2.0",
        "promotion_avg_window": 30,
    },
    5: {
        "name": "MASTERY",
        "description": "Full victory, optimize time and no-hit",
        "episodes": 1000,
        "lr": 5e-5,                     # Very low for fine-tuning
        "lr_end_factor": 0.2,
        "entropy_start": 0.03,
        "entropy_end": 0.005,
        "gae_lambda": 0.97,             # Higher λ for long-horizon mastery
        "n_epochs": 6,                  # More epochs for fine-tuning
        "batch_size": 64,
        "update_interval": 512,         # Longer rollouts for mastery
        "gamma": 0.998,
        "use_pattern_bonus": True,
        "preprocess_version": 2,
        "promotion_condition": "win_rate >= 0.5",
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


def check_promotion(episode_steps_history, episode_kills, episode_damage,
                     wins, total_episodes, config):
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
        env = HollowKnightEnvPPO(host="localhost", port=port, phase=phase, reward_scale=5.0)
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
        action_size=8,                       # Same action space as DQN
        learning_rate=cfg["lr"],
        lr_end_factor=cfg["lr_end_factor"],
        total_episodes=cfg["episodes"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],        # PPO-specific: GAE lambda
        entropy_coef=cfg["entropy_start"],   # PPO-specific: entropy bonus
        use_lstm=True,                       # PPO-specific: recurrent policy
        n_epochs=cfg["n_epochs"],            # PPO-specific: epochs per update
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
        if len(agent.kill_buffer) >= 3 and num_updates > 0:
            agent.learn_from_kills()

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
        if mantis_killed >= 1 or boss_hp_end <= 50:
            if len(ep_transitions) >= 10:
                states_t, actions_t, rewards_t, dones_t = zip(*ep_transitions)
                agent.kill_buffer.add_episode(states_t, actions_t, rewards_t, dones_t)
                label = "KILL" if mantis_killed >= 1 else f"NEAR-KILL (HP={boss_hp_end:.0f})"
                print(f"  [PPO {instance_id}] → {label} saved to buffer ({len(agent.kill_buffer)} stored)")

        # Save checkpoints
        latest_path = os.path.join(instance_dir, "latest.pth")
        agent.save(latest_path)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(instance_dir, "best.pth")
            agent.save(best_model_path)
            hof.update_best_model(instance_id, episode_reward, best_model_path)

        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(instance_dir, f"checkpoint_ep{episode+1}.pth"))

        # Sync with Hall of Fame (conservative)
        if (episode + 1) % (sync_interval * 3) == 0 and len(episode_rewards) >= 20:
            my_avg = np.mean(list(episode_rewards)[-20:])
            global_best = hof.get_global_best_reward()
            if my_avg < global_best * 0.3 and global_best > 0:
                sync_path = hof.get_random_best_model_path(exclude_instance=instance_id)
                if sync_path:
                    try:
                        agent.load(sync_path)
                        print(f"  [PPO {instance_id}] Synced with HoF (avg={my_avg:.1f} vs best={global_best:.1f})")
                    except Exception:
                        pass

        # Log
        lr = agent.get_current_lr()
        hof.log_episode(instance_id, phase, episode+1, episode_reward, step+1,
                        mantis_killed, boss_hp_end, boss_defeated, current_entropy, lr, num_updates)
        hof.increment_episodes()

        ep_num = episode + 1
        kb = len(agent.kill_buffer)
        print(
            f"  [PPO {instance_id}] P{phase} Ep {ep_num:>4}/{cfg['episodes']} | "
            f"R={episode_reward:>+7.2f} | Steps={step+1:>4} | "
            f"HP={boss_hp_end:>5.0f} | K={mantis_killed} | "
            f"Ent={current_entropy:.3f} | LR={lr:.1e} | KB={kb}"
        )

        if boss_defeated:
            wr = wins / total_ep
            print(f"\n  {'★'*20}  [PPO {instance_id}] VICTORY!  {'★'*20}")
            print(f"  P{phase} Ep {ep_num} | R={episode_reward:+.2f} | Wins={wins} WR={wr:.0%}")
            print(f"  {'★'*52}\n")
        elif mantis_killed >= 2:
            print(f"  >>>> [PPO {instance_id}] KILL x{mantis_killed}! Ep {ep_num} <<<<")

        # Promotion check
        if auto_promote and ep_num >= cfg.get("promotion_avg_window", 20):
            if check_promotion(episode_steps_hist, episode_kills, episode_damage, wins, total_ep, cfg):
                print(f"\n  ▲▲▲ [PPO {instance_id}] PROMOTED! Phase {phase} complete! ▲▲▲")
                break

    env.close()
    print(f"[PPO Inst {instance_id}] Phase {phase} DONE | Best R={best_reward:+.2f} | Wins={wins}/{total_ep}")
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
    end_phase: int = 5,
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

        print(f"\n{'─'*70}")
        print(f"  Pausing 5s before next phase...")
        print(f"{'─'*70}\n")
        time.sleep(5)

    print(f"\n{'═'*70}")
    print(f"  PPO TRAINING PIPELINE COMPLETE")
    print(f"  Final model: {current_model}")
    print(f"{'═'*70}\n")


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
    parser.add_argument("--end-phase", type=int, default=5)
    parser.add_argument("--phase", type=int, default=None, help="Run ONLY this phase")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output", type=str, default="training_output_ppo")
    parser.add_argument("--no-auto-promote", action="store_true")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes per phase")
    parser.add_argument("--sync-interval", type=int, default=15)

    args = parser.parse_args()

    if args.episodes:
        for pid in PHASE_CONFIGS:
            PHASE_CONFIGS[pid]["episodes"] = args.episodes

    ports = list(args.ports)
    while len(ports) < args.instances:
        ports.append(ports[-1] + 1)
    ports = ports[:args.instances]

    print(f"\n  Agent: PPO | Instances: {args.instances} | Ports: {ports}")

    if args.phase:
        run_phase_multi_instance(
            phase=args.phase,
            ports=ports,
            checkpoint_dir=args.output,
            pretrained_path=args.pretrained,
            auto_promote=not args.no_auto_promote,
            sync_interval=args.sync_interval,
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
