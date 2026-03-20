"""
═══════════════════════════════════════════════════════════════════════
  DQN TRAINING ORCHESTRATOR — Mantis Lords
  Multi-Instance, Multi-Phase with Hall of Fame

  DQN-SPECIFIC DESIGN CHOICES:
  ● Off-policy: replay buffer stores all transitions for resampling
  ● ε-greedy exploration with exponential decay (not entropy-based)
  ● Target network soft-updates (τ=0.005) for stable Q-targets
  ● Learns every step (when buffer has enough samples)
  ● No LSTM — DQN uses frame stacking for temporal context
  ● No kill buffer — replay buffer already stores all experiences
  ● Works best with sparse rewards (env_dqn.py)

  USAGE:
    python train_dqn.py --instances 2 --ports 5555 5556
    python train_dqn.py --phase 1 --ports 5555
    python train_dqn.py --start-phase 3 --pretrained best.pth
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, argparse, math, json, csv, random
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

from dqn_agent import DQNAgent
from env_dqn import HollowKnightEnvDQN  # DQN-specific sparse reward env
from preprocess import (
    preprocess_state_v1, preprocess_state_v2,
    compute_pattern_reward_bonus, STATE_DIM_V1, STATE_DIM_V2,
)


# ═══════════════════════════════════════════════════════════════
# FRAME STACKER
# ═══════════════════════════════════════════════════════════════

class FrameStacker:
    """Stack N consecutive frames. State [F] -> [F x N]."""
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
    """Shared state across instances: maintains Top-K best models."""

    def __init__(self, checkpoint_dir: str, keep_top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self.models_dir = os.path.join(checkpoint_dir, "best_pool")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, "shared_state.json")
        self.lock_file = os.path.join(checkpoint_dir, "shared_state.lock")
        self.log_file = os.path.join(checkpoint_dir, "training_log_dqn.csv")

        if not os.path.exists(self.state_file):
            self._write_state({
                "best_models": [], "total_episodes": 0,
                "global_best_reward": -float("inf"), "agent_type": "dqn",
            })
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'timestamp', 'instance_id', 'phase', 'episode', 'reward',
                    'steps', 'mantis_killed', 'boss_hp', 'boss_defeated',
                    'epsilon', 'learning_rate', 'avg_loss'
                ])

    def _read_state(self) -> dict:
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"best_models": [], "total_episodes": 0,
                    "global_best_reward": -float("inf")}

    def _write_state(self, state: dict):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def update_best_model(self, instance_id: int, reward: float,
                          model_path: str) -> bool:
        lock = filelock.FileLock(self.lock_file, timeout=15)
        with lock:
            state = self._read_state()
            best_models = state.get("best_models", [])
            existing_idx, existing_reward = None, -float("inf")
            for i, entry in enumerate(best_models):
                if entry.get("instance_id") == instance_id:
                    existing_idx, existing_reward = i, entry.get("reward", -float("inf"))
                    break
            if existing_idx is not None:
                if reward < existing_reward + 0.5:
                    return False
                old = best_models.pop(existing_idx)
                try:
                    if os.path.exists(old.get("path", "")):
                        os.remove(old["path"])
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
            pool_path = os.path.join(self.models_dir,
                                     f"hof_dqn_inst{instance_id}.pth")
            try:
                shutil.copy2(model_path, pool_path)
            except Exception as e:
                print(f"  [HoF-DQN] Copy failed: {e}")
                return False
            best_models.insert(insert_idx, {
                "instance_id": instance_id, "reward": reward,
                "path": pool_path, "timestamp": datetime.now().isoformat(),
            })
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
            print(f"\n  {'*'*50}")
            print(f"  * [DQN Inst {instance_id}] HALL OF FAME! "
                  f"R={reward:.2f} (Rank {insert_idx+1}/{self.keep_top_k})")
            print(f"  {'*'*50}\n")
            return True

    def get_random_best_model_path(self, exclude_instance: int = -1):
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            models = state.get("best_models", [])
            cands = [m for m in models
                     if m.get("instance_id") != exclude_instance]
            if not cands:
                cands = models
            if not cands:
                return None
            path = random.choice(cands).get("path", "")
            return path if os.path.exists(path) else None

    def get_global_best_reward(self) -> float:
        return self._read_state().get("global_best_reward", -float("inf"))

    def increment_episodes(self) -> int:
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            state["total_episodes"] = state.get("total_episodes", 0) + 1
            self._write_state(state)
            return state["total_episodes"]

    def log_episode(self, instance_id, phase, episode, reward, steps,
                    mantis_killed, boss_hp, boss_defeated, epsilon,
                    lr, avg_loss):
        lock = filelock.FileLock(self.lock_file, timeout=5)
        try:
            with lock:
                with open(self.log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%H:%M:%S"),
                        instance_id, phase, episode, f"{reward:.2f}",
                        steps, mantis_killed, f"{boss_hp:.0f}",
                        1 if boss_defeated else 0,
                        f"{epsilon:.4f}", f"{lr:.2e}", f"{avg_loss:.4f}"
                    ])
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# DQN-SPECIFIC PHASE CONFIGS
#
# vs PPO: Lower LR, epsilon-greedy (not entropy), replay buffer
# capacity, target network tau, no GAE/n_epochs/LSTM params
# ═══════════════════════════════════════════════════════════════

PHASE_CONFIGS = {
    1: {
        "name": "SURVIVAL",
        "description": "Learn to dodge and not die",
        "episodes": 1000,
        "lr": 1e-4,
        "epsilon_start": 0.60, "epsilon_end": 0.10, "epsilon_decay": 30000,
        "hidden_sizes": [128, 256, 128],
        "buffer_capacity": 100000, "batch_size": 256,
        "target_update_tau": 0.005,
        "gamma": 0.99,
        "use_pattern_bonus": True, "preprocess_version": 2,
        "promotion_condition": "avg_survival_steps >= 850",
        "promotion_avg_window": 25,
    },
    2: {
        "name": "FIRST HITS",
        "description": "Learn to punish during recovery windows",
        "episodes": 1000,
        "lr": 5e-5,
        "epsilon_start": 0.40, "epsilon_end": 0.08, "epsilon_decay": 25000,
        "hidden_sizes": [128, 256, 128],
        "buffer_capacity": 100000, "batch_size": 256,
        "target_update_tau": 0.005,
        "gamma": 0.995,
        "use_pattern_bonus": True, "preprocess_version": 2,
        "promotion_condition": "avg_damage_dealt >= 250",
        "promotion_avg_window": 25,
    },
    3: {
        "name": "AGGRESSION",
        "description": "Kill the first mantis",
        "episodes": 1200,
        "lr": 3e-5,
        "epsilon_start": 0.30, "epsilon_end": 0.05, "epsilon_decay": 40000,
        "hidden_sizes": [128, 256, 128],
        "buffer_capacity": 100000, "batch_size": 256,
        "target_update_tau": 0.005,
        "gamma": 0.995,
        "use_pattern_bonus": True, "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 0.8",
        "promotion_avg_window": 30,
    },
    4: {
        "name": "DUAL MANTIS",
        "description": "Handle two mantises at once",
        "episodes": 2000,
        "lr": 2e-5,
        "epsilon_start": 0.25, "epsilon_end": 0.05, "epsilon_decay": 40000,
        "hidden_sizes": [128, 256, 128],
        "buffer_capacity": 150000, "batch_size": 256,
        "target_update_tau": 0.005,
        "gamma": 0.995,
        "use_pattern_bonus": True, "preprocess_version": 2,
        "promotion_condition": "avg_mantis_killed >= 1.8",
        "promotion_avg_window": 40,
    },
    5: {
        "name": "MASTERY",
        "description": "Full victory, optimize time and no-hit",
        "episodes": 1500,
        "lr": 1e-5,
        "epsilon_start": 0.15, "epsilon_end": 0.03, "epsilon_decay": 50000,
        "hidden_sizes": [128, 256, 128],
        "buffer_capacity": 200000, "batch_size": 512,
        "target_update_tau": 0.002,
        "gamma": 0.998,
        "use_pattern_bonus": True, "preprocess_version": 2,
        "promotion_condition": "win_rate >= 0.5",
        "promotion_avg_window": 50,
    },
}


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def check_promotion(episode_steps_history, episode_kills, episode_damage,
                     wins, total_episodes, config):
    window = config["promotion_avg_window"]
    if len(episode_steps_history) < window:
        return False
    ctx = {
        "avg_survival_steps": np.mean(list(episode_steps_history)[-window:]),
        "avg_mantis_killed":  np.mean(list(episode_kills)[-window:]),
        "avg_damage_dealt":   np.mean(list(episode_damage)[-window:]),
        "win_rate":           wins / max(total_episodes, 1),
    }
    try:
        return eval(config["promotion_condition"], {"__builtins__": {}}, ctx)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# DQN TRAINING WORKER
#
# KEY DQN CHARACTERISTICS vs PPO:
# ● epsilon-greedy action selection (not policy sampling)
# ● Store (s, a, r, s', done) — no log_prob or value
# ● Learn EVERY step from replay buffer random sampling
# ● Soft target network update after each optimization
# ● No LSTM, no kill buffer, no entropy coefficient
# ═══════════════════════════════════════════════════════════════

def train_dqn_instance(
    instance_id: int, port: int, phase: int, checkpoint_dir: str,
    pretrained_path: Optional[str] = None, auto_promote: bool = True,
    sync_interval: int = 15, max_steps: int = 5000,
):
    """Worker function for DQN training on a single instance."""
    cfg = PHASE_CONFIGS[phase]
    phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
    instance_dir = os.path.join(phase_dir, f"instance_{instance_id}")
    os.makedirs(instance_dir, exist_ok=True)

    print(f"\n[DQN Inst {instance_id}] Phase {phase}: {cfg['name']} | Port {port}")
    hof = HallOfFame(phase_dir, keep_top_k=3)

    # ═══ DQN ENV: Sparse reward environment ═══
    try:
        env = HollowKnightEnvDQN(
            host="localhost", port=port, phase=phase, reward_scale=5.0)
    except Exception as e:
        print(f"[DQN Inst {instance_id}] Connection failed: {e}")
        return None

    # Preprocessing (identical to PPO)
    STACK_SIZE = 4
    version = cfg["preprocess_version"]
    raw_dim = STATE_DIM_V2 if version == 2 else STATE_DIM_V1
    stacked_dim = raw_dim * STACK_SIZE
    preprocess_fn = preprocess_state_v2 if version == 2 else preprocess_state_v1
    stacker = FrameStacker(STACK_SIZE, raw_dim)
    print(f"[DQN Inst {instance_id}] State: {raw_dim} x {STACK_SIZE} = {stacked_dim}")

    # ═══ DQN AGENT: Off-policy with Dueling architecture ═══
    agent = DQNAgent(
        state_size=stacked_dim,
        action_size=8,                            # Same action space as PPO
        hidden_sizes=cfg["hidden_sizes"],          # DQN: MLP (no LSTM)
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        buffer_capacity=cfg["buffer_capacity"],    # DQN: replay buffer
    )

    # Load pretrained
    loaded = False
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            agent.load(pretrained_path)
            print(f"[DQN Inst {instance_id}] Loaded pretrained: {pretrained_path}")
            loaded = True
        except Exception as e:
            print(f"[DQN Inst {instance_id}] Pretrained load failed: {e}")
    if not loaded:
        pool_path = hof.get_random_best_model_path(exclude_instance=instance_id)
        if pool_path:
            try:
                agent.load(pool_path)
                print(f"[DQN Inst {instance_id}] Loaded from Hall of Fame")
            except Exception:
                pass

    # Tracking
    best_reward = -float("inf")
    best_model_path = None
    episode_rewards = deque(maxlen=200)
    episode_steps_hist = deque(maxlen=200)
    episode_kills = deque(maxlen=200)
    episode_damage = deque(maxlen=200)
    wins, total_ep = 0, 0
    dqn_batch_size = cfg["batch_size"]
    target_tau = cfg["target_update_tau"]

    # ═══ TRAINING LOOP ═══
    for episode in range(cfg["episodes"]):
        raw_state_dict = env.reset()
        raw_state = preprocess_fn(raw_state_dict)
        state = stacker.reset(raw_state)
        # DQN: No LSTM hidden state to reset

        episode_reward = 0.0
        episode_losses = []
        ep_boss_hp_start = raw_state_dict.get("bossHealth", 100.0)

        # ═══ DQN: epsilon-greedy (exponential decay over total steps) ═══
        epsilon = agent.get_epsilon(
            cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay"])

        for step in range(max_steps):
            # ═══ DQN: epsilon-greedy selection (no log_prob/value) ═══
            action = agent.select_action(state, epsilon=epsilon)
            next_state_dict, reward, done, info = env.step(action)

            if cfg["use_pattern_bonus"]:
                reward += compute_pattern_reward_bonus(next_state_dict, action)

            next_raw = preprocess_fn(next_state_dict)
            next_state = stacker.step(next_raw)
            episode_reward += reward

            # ═══ DQN: Store (s, a, r, s', done) in replay buffer ═══
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            # ═══ DQN: Learn every step from replay buffer ═══
            if len(agent.memory) >= dqn_batch_size:
                loss = agent.optimize_model(batch_size=dqn_batch_size)
                if loss is not None:
                    episode_losses.append(loss)
                # ═══ DQN: Soft target network update ═══
                agent.update_target_network(tau=target_tau)

            if done:
                break

        # Post-episode
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

        # Save
        latest_path = os.path.join(instance_dir, "latest.pth")
        agent.save(latest_path)
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(instance_dir, "best.pth")
            agent.save(best_model_path)
            hof.update_best_model(instance_id, episode_reward, best_model_path)
        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(
                instance_dir, f"checkpoint_ep{episode+1}.pth"))

        # Sync with Hall of Fame
        if ((episode + 1) % (sync_interval * 3) == 0
                and len(episode_rewards) >= 20):
            my_avg = np.mean(list(episode_rewards)[-20:])
            global_best = hof.get_global_best_reward()
            if my_avg < global_best * 0.3 and global_best > 0:
                sync_path = hof.get_random_best_model_path(
                    exclude_instance=instance_id)
                if sync_path:
                    try:
                        agent.load(sync_path)
                        print(f"  [DQN {instance_id}] Synced with HoF "
                              f"(avg={my_avg:.1f} vs best={global_best:.1f})")
                    except Exception:
                        pass

        # Log
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        hof.log_episode(
            instance_id, phase, episode+1, episode_reward, step+1,
            mantis_killed, boss_hp_end, boss_defeated,
            epsilon, cfg["lr"], avg_loss)
        hof.increment_episodes()

        ep_num = episode + 1
        print(
            f"  [DQN {instance_id}] P{phase} Ep {ep_num:>4}/{cfg['episodes']}"
            f" | R={episode_reward:>+7.2f} | Steps={step+1:>4}"
            f" | HP={boss_hp_end:>5.0f} | K={mantis_killed}"
            f" | eps={epsilon:.3f} | Loss={avg_loss:.4f}"
            f" | Buf={len(agent.memory)}")

        if boss_defeated:
            wr = wins / total_ep
            print(f"\n  {'*'*20}  [DQN {instance_id}] VICTORY!  {'*'*20}")
            print(f"  P{phase} Ep {ep_num} | R={episode_reward:+.2f} "
                  f"| Wins={wins} WR={wr:.0%}")
            print(f"  {'*'*52}\n")
        elif mantis_killed >= 2:
            print(f"  >>>> [DQN {instance_id}] KILL x{mantis_killed}! "
                  f"Ep {ep_num} <<<<")

        # Promotion
        if auto_promote and ep_num >= cfg.get("promotion_avg_window", 20):
            if check_promotion(episode_steps_hist, episode_kills,
                               episode_damage, wins, total_ep, cfg):
                print(f"\n  ^^^ [DQN {instance_id}] PROMOTED! "
                      f"Phase {phase} complete! ^^^")
                break

    env.close()
    print(f"[DQN Inst {instance_id}] Phase {phase} DONE "
          f"| Best R={best_reward:+.2f} | Wins={wins}/{total_ep}")
    return best_model_path


# ═══════════════════════════════════════════════════════════════
# MULTI-INSTANCE LAUNCHER
# ═══════════════════════════════════════════════════════════════

def run_phase_multi_instance(
    phase: int, ports: List[int], checkpoint_dir: str,
    pretrained_path: Optional[str] = None,
    auto_promote: bool = True, sync_interval: int = 15,
):
    cfg = PHASE_CONFIGS[phase]
    n = len(ports)
    print(f"\n{'='*70}")
    print(f"  PHASE {phase}: {cfg['name']} | Agent: DQN | Instances: {n}")
    print(f"  Ports: {ports}")
    print(f"  Episodes: {cfg['episodes']} | Promotion: "
          f"{cfg['promotion_condition']}")
    print(f"{'='*70}\n")

    if n == 1:
        return train_dqn_instance(
            instance_id=0, port=ports[0], phase=phase,
            checkpoint_dir=checkpoint_dir,
            pretrained_path=pretrained_path,
            auto_promote=auto_promote, sync_interval=sync_interval)
    else:
        processes = []
        for i, port in enumerate(ports):
            p = mp.Process(target=train_dqn_instance, kwargs={
                "instance_id": i, "port": port, "phase": phase,
                "checkpoint_dir": checkpoint_dir,
                "pretrained_path": pretrained_path,
                "auto_promote": auto_promote,
                "sync_interval": sync_interval,
            })
            p.start()
            processes.append(p)
            time.sleep(1)
        for p in processes:
            p.join()

        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
        hof = HallOfFame(phase_dir, keep_top_k=3)
        best_path = hof.get_random_best_model_path()
        if not best_path:
            for i in range(n):
                cand = os.path.join(phase_dir, f"instance_{i}", "best.pth")
                if os.path.exists(cand):
                    best_path = cand
                    break
        return best_path


# ═══════════════════════════════════════════════════════════════
# MULTI-PHASE PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_all_phases(
    ports: List[int], base_dir: str,
    start_phase: int = 1, end_phase: int = 5,
    pretrained_path: Optional[str] = None,
    auto_promote: bool = True, sync_interval: int = 15,
):
    print(f"\n{'='*70}")
    print(f"  DQN MULTI-PHASE PIPELINE — MANTIS LORDS")
    print(f"  Instances: {len(ports)} | Phases: {start_phase}->{end_phase}")
    print(f"  Ports: {ports}")
    print(f"{'='*70}\n")

    os.makedirs(base_dir, exist_ok=True)
    current_model = pretrained_path
    n_instances = len(ports)

    for phase in range(start_phase, end_phase + 1):
        if phase not in PHASE_CONFIGS:
            print(f"[ERROR] Phase {phase} not configured!")
            break
        best_model = run_phase_multi_instance(
            phase=phase, ports=ports, checkpoint_dir=base_dir,
            pretrained_path=current_model,
            auto_promote=auto_promote, sync_interval=sync_interval)
        if best_model and os.path.exists(best_model):
            current_model = best_model
            print(f"\n[DQN Pipeline] Phase {phase} -> best: {best_model}")
        else:
            print(f"\n[DQN Pipeline] Phase {phase} produced no model. Stop.")
            break

        # Eleggi il champion per questa fase
        select_champion(base_dir, phase=phase, n_instances=n_instances)

        print(f"\n{'-'*70}\n  Pausing 5s before next phase...\n{'-'*70}\n")
        time.sleep(5)

    # Raccogli tutti i champion
    collect_all_champions(base_dir, n_instances=n_instances)

    print(f"\n{'='*70}")
    print(f"  DQN TRAINING PIPELINE COMPLETE")
    print(f"  Final model: {current_model}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# CHAMPION SYSTEM — Seleziona il miglior modello per ogni fase
# ═══════════════════════════════════════════════════════════════

def select_champion(checkpoint_dir: str, phase: int = 5, n_instances: int = 3) -> Optional[str]:
    """
    Trova il miglior modello tra tutte le istanze di una fase.
    Confronta con il champion precedente se esiste.
    """
    import shutil

    phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
    champion_dir = os.path.join(checkpoint_dir, "champion")
    os.makedirs(champion_dir, exist_ok=True)

    phase_name = PHASE_CONFIGS.get(phase, {}).get("name", f"PHASE_{phase}")
    champion_model = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
    champion_meta = os.path.join(champion_dir, f"phase_{phase}_champion.json")
    history_file = os.path.join(champion_dir, f"phase_{phase}_history.json")

    candidates = []

    state_file = os.path.join(phase_dir, "shared_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            for entry in state.get("best_models", []):
                path = entry.get("path", "")
                reward = entry.get("reward", -float("inf"))
                inst_id = entry.get("instance_id", -1)
                if os.path.exists(path):
                    candidates.append({
                        "path": path, "reward": reward,
                        "instance_id": inst_id, "source": "hall_of_fame",
                    })
        except Exception:
            pass

    for i in range(n_instances):
        best_path = os.path.join(phase_dir, f"instance_{i}", "best.pth")
        if os.path.exists(best_path):
            already = any(c["instance_id"] == i for c in candidates)
            if not already:
                candidates.append({
                    "path": best_path, "reward": -1.0,
                    "instance_id": i, "source": "instance_best",
                })

    if not candidates:
        print(f"\n  [Champion] Nessun modello trovato in {phase_dir}")
        return None

    candidates.sort(key=lambda c: c["reward"], reverse=True)
    new_champion = candidates[0]

    print(f"\n{'='*60}")
    print(f"  CHAMPION SELECTION — Phase {phase}: {phase_name}")
    print(f"{'='*60}")
    print(f"  Candidati trovati: {len(candidates)}")
    for i, c in enumerate(candidates):
        marker = " << BEST" if i == 0 else ""
        print(f"    Inst {c['instance_id']}: R={c['reward']:.2f} ({c['source']}){marker}")

    old_champion = None
    if os.path.exists(champion_meta):
        try:
            with open(champion_meta, 'r') as f:
                old_champion = json.load(f)
            print(f"\n  Champion precedente: R={old_champion.get('reward', '?'):.2f} "
                  f"(Inst {old_champion.get('instance_id', '?')}, "
                  f"run {old_champion.get('run_id', '?')})")
        except Exception:
            old_champion = None

    run_id = 1
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                run_id = len(json.load(f)) + 1
        except Exception:
            pass

    save_new = False
    if old_champion is None:
        save_new = True
        print(f"\n  Nessun champion precedente -> il nuovo diventa champion")
    elif new_champion["reward"] > old_champion.get("reward", -float("inf")):
        save_new = True
        improvement = new_champion["reward"] - old_champion.get("reward", 0)
        print(f"\n  * NUOVO CHAMPION! R={new_champion['reward']:.2f} > "
              f"R={old_champion.get('reward', 0):.2f} (+{improvement:.2f})")
    else:
        print(f"\n  Champion precedente confermato (R={old_champion.get('reward', 0):.2f} >= "
              f"R={new_champion['reward']:.2f})")

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
            "phase": phase, "phase_name": phase_name,
        }
        with open(champion_meta, 'w') as f:
            json.dump(meta, f, indent=2)

    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            pass
    history.append({
        "run_id": run_id, "reward": new_champion["reward"],
        "instance_id": new_champion["instance_id"],
        "is_new_champion": save_new,
        "timestamp": datetime.now().isoformat(),
    })
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Champion Phase {phase}: {champion_model}")
    if len(history) > 1:
        print(f"\n  History ({len(history)} run):")
        for h in history:
            marker = " * CHAMPION" if h["is_new_champion"] else ""
            print(f"    Run {h['run_id']}: R={h['reward']:.2f} (Inst {h['instance_id']}){marker}")
    print(f"{'='*60}\n")
    return champion_model


def collect_all_champions(checkpoint_dir: str, n_instances: int = 3):
    """
    Raccoglie i champion di tutte le fasi in un'unica cartella.
    Crea retroattivamente i champion per le fasi già completate.
    """
    existing_phases = []
    for phase in range(1, 6):
        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
        if os.path.exists(phase_dir):
            existing_phases.append(phase)

    if not existing_phases:
        return

    champion_dir = os.path.join(checkpoint_dir, "champion")
    os.makedirs(champion_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RACCOLTA CHAMPION — Tutte le fasi")
    print(f"{'='*60}")

    found = []
    for phase in existing_phases:
        phase_dir = os.path.join(checkpoint_dir, f"phase_{phase}")
        champion_file = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        champion_meta = os.path.join(champion_dir, f"phase_{phase}_champion.json")

        if not os.path.exists(champion_file):
            select_champion(checkpoint_dir, phase=phase, n_instances=n_instances)

        if os.path.exists(champion_meta):
            try:
                with open(champion_meta, 'r') as f:
                    found.append(json.load(f))
            except Exception:
                found.append({"phase": phase, "reward": "?", "phase_name": "?"})
        elif os.path.exists(champion_file):
            found.append({"phase": phase, "reward": "?", "phase_name": "?"})

    if not found:
        print(f"\n  Nessun champion trovato.")
        return

    print(f"\n  Champion disponibili:")
    print(f"  {'-'*56}")
    for meta in found:
        phase = meta.get("phase", "?")
        name = meta.get("phase_name", PHASE_CONFIGS.get(phase, {}).get("name", "?"))
        reward = meta.get("reward", "?")
        inst = meta.get("instance_id", "?")
        runs = meta.get("run_id", "?")
        reward_str = f"R={reward:.2f}" if isinstance(reward, (int, float)) else f"R={reward}"
        champ_path = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        print(f"    Phase {phase} ({name:>12}): {reward_str} | Inst {inst} | run {runs}")
        print(f"      -> python play_dqn.py --model {champ_path}")

    global_champ = None
    for phase in [5, 4, 3, 2, 1]:
        candidate = os.path.join(champion_dir, f"phase_{phase}_champion.pth")
        if os.path.exists(candidate):
            global_champ = candidate
            break

    if global_champ:
        print(f"\n  {'-'*56}")
        print(f"  Miglior modello complessivo:")
        print(f"    python play_dqn.py --model {global_champ}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="DQN Training for Mantis Lords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_dqn.py --instances 2 --ports 5555 5556
  python train_dqn.py --phase 1 --ports 5555
  python train_dqn.py --start-phase 3 --pretrained best.pth
        """)

    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--ports", type=int, nargs="+", default=[5555])
    parser.add_argument("--start-phase", type=int, default=1)
    parser.add_argument("--end-phase", type=int, default=5)
    parser.add_argument("--phase", type=int, default=None,
                        help="Run ONLY this phase")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output", type=str, default="training_output_dqn")
    parser.add_argument("--no-auto-promote", action="store_true")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes per phase")
    parser.add_argument("--sync-interval", type=int, default=15)

    args = parser.parse_args()

    if args.episodes:
        for pid in PHASE_CONFIGS:
            PHASE_CONFIGS[pid]["episodes"] = args.episodes

    ports = list(args.ports)
    while len(ports) < args.instances:
        ports.append(ports[-1] + 1)
    ports = ports[:args.instances]

    print(f"\n  Agent: DQN | Instances: {args.instances} | Ports: {ports}")

    # ═══ Raccogli i champion da tutte le fasi già esistenti ═══
    collect_all_champions(
        checkpoint_dir=args.output,
        n_instances=args.instances,
    )

    if args.phase:
        run_phase_multi_instance(
            phase=args.phase, ports=ports,
            checkpoint_dir=args.output,
            pretrained_path=args.pretrained,
            auto_promote=not args.no_auto_promote,
            sync_interval=args.sync_interval)

        # Eleggi il champion per la fase completata
        select_champion(
            checkpoint_dir=args.output,
            phase=args.phase,
            n_instances=args.instances,
        )

        # Riepilogo completo
        collect_all_champions(
            checkpoint_dir=args.output,
            n_instances=args.instances,
        )
    else:
        run_all_phases(
            ports=ports, base_dir=args.output,
            start_phase=args.start_phase, end_phase=args.end_phase,
            pretrained_path=args.pretrained,
            auto_promote=not args.no_auto_promote,
            sync_interval=args.sync_interval)