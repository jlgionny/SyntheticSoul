"""
Multi-Instance Training Orchestrator for Hollow Knight AI.

Permette di:
- Scegliere l'agente (DQN o PPO)
- Avviare multiple istanze in parallelo
- Condividere experience buffer e modelli tra istanze (Hall of Fame Top 3)
- Sincronizzare i modelli automaticamente
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Optional
import numpy as np
import json
import shutil
import filelock
import random

# Setup paths - Aggiungi la directory AI_Agents al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_AGENTS_DIR = os.path.dirname(SCRIPT_DIR)  # AI_Agents/
sys.path.insert(0, AI_AGENTS_DIR)

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv

# ============ FILE-BASED SHARED STATE (HALL OF FAME VERSION) ============
class FileBasedSharedState:
    """
    Stato condiviso avanzato: Mantiene una Hall of Fame dei Top K modelli.
    Thread-safe tramite file locking.
    """

    def __init__(self, checkpoint_dir: str, agent_type: str, keep_top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.agent_type = agent_type
        self.keep_top_k = keep_top_k  # Quanti modelli tenere (es. 3)

        # Setup directories
        self.models_dir = os.path.join(checkpoint_dir, "best_pool")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # File paths
        self.log_file = os.path.join(checkpoint_dir, "training_log.txt")
        self.state_file = os.path.join(checkpoint_dir, "shared_state.json")
        self.lock_file = os.path.join(checkpoint_dir, ".lock")

        # Initialize files
        self._init_files()

    def _init_files(self):
        """Initialize shared files."""
        # Log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(
                    "timestamp,instance,episode,reward,steps,boss_hp,mantis_killed,loss,epsilon_or_entropy\n"
                )

        # State file
        if not os.path.exists(self.state_file):
            # best_models è una lista di dict: [{'reward': float, 'path': str, 'instance': int}, ...]
            self._write_state({
                "best_models": [],
                "total_episodes": 0,
                "global_best_reward": -float("inf")
            })

    def _read_state(self) -> dict:
        """Read shared state from file."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "best_models": [],
                "total_episodes": 0,
                "global_best_reward": -float("inf")
            }

    def _write_state(self, state: dict):
        """Write shared state to file."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def log_episode(
        self,
        instance_id: int,
        episode: int,
        reward: float,
        steps: int,
        boss_hp: float,
        mantis_killed: int,
        loss: float,
        epsilon_or_entropy: float,
    ):
        """Log episode result (thread-safe)."""
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(
                    f"{timestamp},{instance_id},{episode},{reward:.2f},{steps},"
                    f"{boss_hp:.0f},{mantis_killed},{loss:.4f},{epsilon_or_entropy:.4f}\n"
                )

    def update_best_model(
        self, instance_id: int, reward: float, source_model_path: str
    ) -> bool:
        """
        Controlla se il modello merita di entrare nella Hall of Fame (Top K).
        Se sì, lo salva e rimuove il peggiore se la lista è piena.
        """
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            best_models = state.get("best_models", [])

            # Aggiorna il massimo globale per riferimento
            state["global_best_reward"] = max(state.get("global_best_reward", -float('inf')), reward)

            # Logica di inserimento
            inserted = False

            # Se abbiamo meno di K modelli, o se il reward è migliore del peggiore dei nostri Top K
            if len(best_models) < self.keep_top_k or (len(best_models) > 0 and reward > best_models[-1]['reward']):

                # 1. Crea nome file unico
                timestamp_str = str(int(time.time()))
                filename = f"model_rew{int(reward)}_inst{instance_id}_{timestamp_str}.pth"
                dest_path = os.path.join(self.models_dir, filename)

                # 2. Copia il file
                if os.path.exists(source_model_path):
                    shutil.copy(source_model_path, dest_path)

                    # 3. Aggiungi alla lista
                    new_entry = {
                        "reward": reward,
                        "path": dest_path,
                        "instance": instance_id,
                        "timestamp": timestamp_str
                    }
                    best_models.append(new_entry)

                    # 4. Ordina per reward decrescente (Il migliore è index 0)
                    best_models.sort(key=lambda x: x['reward'], reverse=True)

                    # 5. Taglia la lista se troppo lunga e cancella il file vecchio
                    while len(best_models) > self.keep_top_k:
                        removed_entry = best_models.pop()    # Rimuove l'ultimo (il peggiore dei migliori)
                        # Cancella fisicamente il file per non riempire il disco
                        if os.path.exists(removed_entry['path']):
                            try:
                                os.remove(removed_entry['path'])
                            except OSError:
                                pass

                    state["best_models"] = best_models
                    self._write_state(state)

                    rank = best_models.index(new_entry) + 1
                    print(f"\n[Instance {instance_id}] ENTERED HALL OF FAME! Reward: {reward:.2f} (Rank {rank}/{self.keep_top_k})")
                    inserted = True

            return inserted

    def get_any_best_model_path(self) -> Optional[str]:
        """Restituisce un modello a caso dalla Top K per variare l'apprendimento."""
        state = self._read_state()
        best_models = state.get("best_models", [])

        if not best_models:
            return None

        # Pesca a caso uno dei modelli migliori
        # Questo aiuta a evitare minimi locali: a volte proviamo la strategia del 1°, a volte del 2°
        chosen = random.choice(best_models)
        if os.path.exists(chosen['path']):
            return chosen['path']
        return None

    def get_global_best_reward(self) -> float:
        """Restituisce il reward più alto mai visto."""
        state = self._read_state()
        return state.get("global_best_reward", -float("inf"))

    def increment_episodes(self) -> int:
        """Increment global episode counter."""
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            state["total_episodes"] = state.get("total_episodes", 0) + 1
            self._write_state(state)
            return state["total_episodes"]


# ============ PREPROCESSING FUNCTIONS (UPDATED) ============
def preprocess_state_dqn(state_dict: dict) -> np.ndarray:
    """
    DQN State Preprocessing v3 - FIX NORMALIZZAZIONE
    """
    features = []

    # 1. Player Status (5)
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # 2. Player Velocity (2)
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # 3. Terrain (5) - CORRETTO!
    # Il C# invia già valori normalizzati 0-1 (dove 0=vicino, 1=lontano/vuoto).
    # NON dividere di nuovo per 20.0!
    terrain_info = state_dict.get("terrainInfo", [1.0] * 5)
    if not terrain_info or len(terrain_info) < 5:
        terrain_info = [1.0] * 5
    # Usiamo direttamente i valori (clippati per sicurezza tecnica, ma senza divisione)
    features.extend([np.clip(t, 0.0, 1.0) for t in terrain_info[:5]])

    # 4. Boss Position (4)
    boss_rel_x = state_dict.get("bossRelativeX", 0.0)
    boss_rel_y = state_dict.get("bossRelativeY", 0.0)
    distance = state_dict.get("distanceToBoss", 50.0) / 50.0
    facing_boss = float(state_dict.get("isFacingBoss", False))

    features.append(np.clip(boss_rel_x, -1.0, 1.0))        # C# invia già normalizzato -1 a 1? Controlla sotto*
    features.append(np.clip(boss_rel_y, -1.0, 1.0))
    features.append(np.clip(distance, 0.0, 1.0))
    features.append(facing_boss)

    # 5.Boss Status & Kills(4)
    features.append(np.clip(state_dict.get("bossVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossVelocityY", 0.0) / 20.0, -1.0, 1.0))
    features.append(state_dict.get("bossHealth", 100.0) / 100.0)
    features.append(state_dict.get("mantisLordsKilled", 0) / 3.0)

    # 6. BOSS INTENT (4 Features One-Hot)
    boss_action = state_dict.get("bossAction", 0)
    features.append(1.0 if boss_action == 0 else 0.0)
    features.append(1.0 if boss_action == 1 else 0.0)
    features.append(1.0 if boss_action == 2 else 0.0)
    features.append(1.0 if boss_action == 3 else 0.0)

    # 7. HAZARDS (10 Features)
    hazards = state_dict.get("nearbyHazards", [])

    # Helper per hazard
    def add_hazard_features(h):
        # Hazard relative position: C# invia raw distance (dx, dy).
        # Qui dobbiamo normalizzare. 15.0 è un buon range visivo.
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))

    if len(hazards) > 0:
        add_hazard_features(hazards[0])
    else:
        features.extend([0.0] * 5)

    if len(hazards) > 1:
        add_hazard_features(hazards[1])
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)


def preprocess_state_ppo(state_dict: dict) -> np.ndarray:
    """
    PPO State Preprocessing. Aggiornato per coerenza con le nuove feature.
    """
    # Usiamo lo stesso preprocessore del DQN per ora per consistenza,
    # dato che include già tutte le informazioni vitali.
    return preprocess_state_dqn(state_dict)


# ============ WORKER FUNCTIONS ============
def train_dqn_instance(
    instance_id: int,
    port: int,
    checkpoint_dir: str,
    num_episodes: int,
    sync_interval: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 30000,
    max_steps: int = 3000,
):
    """Worker function per training DQN di una singola istanza."""

    print(f"\n[Instance {instance_id}] Starting DQN training on port {port}")

    # Create shared state handler (Hall of Fame enabled)
    shared_state = FileBasedSharedState(checkpoint_dir, "dqn", keep_top_k=3)

    # Instance-specific checkpoint dir
    instance_dir = os.path.join(checkpoint_dir, f"instance_{instance_id}")
    os.makedirs(instance_dir, exist_ok=True)

    # Connect to environment
    try:
        env = HollowKnightEnv(host="localhost", port=port, use_reward_shaping=True)
    except Exception as e:
        print(f"[Instance {instance_id}] Failed to connect: {e}")
        return

    # Get state size
    initial_state = env.reset()
    state_array = preprocess_state_dqn(initial_state)
    state_size = len(state_array)
    action_size = 8  # 8 azioni (IDLE rimosso)

    print(f"[Instance {instance_id}] State size: {state_size} features")

    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,
    )

    # Try to load best model initially (random one from pool)
    best_model_path = shared_state.get_any_best_model_path()
    if best_model_path:
        try:
            agent.load(best_model_path)
            print(f"[Instance {instance_id}] Loaded shared model from pool: {os.path.basename(best_model_path)}")
            agent.steps_done = 0
        except Exception as e:
            print(f"[Instance {instance_id}] Could not load model: {e}")

    global_step = agent.steps_done
    best_local_reward = -float("inf")
    for episode in range(num_episodes):
        state_dict = env.reset()
        state = preprocess_state_dqn(state_dict)

        episode_reward = 0.0
        episode_loss = []

        for step in range(max_steps):
            global_step += 1

            # Action selection
            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            # Environment step
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state_dqn(next_state_dict)

            episode_reward += reward

            # Store in local buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Training
            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size=batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                agent.update_target_network(tau=0.005)

            state = next_state

            if done:
                break

        # Episode complete
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        mantis_killed = next_state_dict.get("mantisLordsKilled", 0)
        boss_hp = next_state_dict.get("bossHealth", 0)

        # Log to shared state
        shared_state.log_episode(
            instance_id,
            episode + 1,
            episode_reward,
            step + 1,
            boss_hp,
            mantis_killed,
            avg_loss,
            epsilon,
        )

        # Save local model
        local_model_path = os.path.join(instance_dir, "latest.pth")
        agent.save(local_model_path)

        # Update global best pool if needed
        if episode_reward > best_local_reward:
            best_local_reward = episode_reward
            # Passa il file, la classe decide se salvarlo nella Hall of Fame
            shared_state.update_best_model(
                instance_id, episode_reward, local_model_path
            )

        # Sync with Hall of Fame periodically
        if (episode + 1) % sync_interval == 0:
            target_model_path = shared_state.get_any_best_model_path()
            global_best_reward = shared_state.get_global_best_reward()

            # Scarica solo se stiamo performando decisamente peggio del migliore assoluto
            if target_model_path and episode_reward < global_best_reward * 0.8:
                try:
                    # 1. Salviamo i contatori attuali per NON resettare l'epsilon
                    current_steps = agent.steps_done
                    current_eps = agent.episodes_done

                    # 2. Carichiamo i pesi di un modello random dalla Hall of Fame
                    agent.load(target_model_path)

                    # 3. Ripristiniamo i contatori
                    agent.steps_done = current_steps
                    agent.episodes_done = current_eps

                    print(
                        f"[Instance {instance_id}] Synced with Hall of Fame model: {os.path.basename(target_model_path)}"
                    )
                except Exception as e:
                    print(
                        f"[Instance {instance_id}] Could not sync with model: {e}"
                    )

        global_ep = shared_state.increment_episodes()

        # Progress log
        if (episode + 1) % 5 == 0:
            print(
                f"[Instance {instance_id}] Ep {episode+1}: R={episode_reward:.1f} | "
                f"Loss={avg_loss:.4f} | Eps={epsilon:.3f} | Mantis={mantis_killed}/3 | "
                f"Global Ep={global_ep}"
            )

    env.close()
    print(f"[Instance {instance_id}] Training complete!")


def train_ppo_instance(
    instance_id: int,
    port: int,
    checkpoint_dir: str,
    num_episodes: int,
    sync_interval: int = 10,
    learning_rate: float = 1e-4,
    gamma: float = 0.995,
    entropy_start: float = 0.05,
    entropy_end: float = 0.01,
    update_interval: int = 1024,
    max_steps: int = 6000,
):
    """Worker function per training PPO di una singola istanza."""

    print(f"\n[Instance {instance_id}] Starting PPO training on port {port}")

    shared_state = FileBasedSharedState(checkpoint_dir, "ppo", keep_top_k=3)

    instance_dir = os.path.join(checkpoint_dir, f"instance_{instance_id}")
    os.makedirs(instance_dir, exist_ok=True)

    try:
        env = HollowKnightEnv(host="localhost", port=port, use_reward_shaping=True)
    except Exception as e:
        print(f"[Instance {instance_id}] Failed to connect: {e}")
        return

    # Get state size (updated)
    initial_state = env.reset()
    state_array = preprocess_state_ppo(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Instance {instance_id}] State size: {state_size} features")

    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=0.97,
        entropy_coef=entropy_start,
        use_lstm=True,
        n_epochs=6,
        batch_size=128,
    )

    # Load from pool
    best_model_path = shared_state.get_any_best_model_path()
    if best_model_path:
        try:
            agent.load(best_model_path)
            print(f"[Instance {instance_id}] Loaded shared model from pool")
        except Exception as e:
            print(f"[Instance {instance_id}] Could not load model: {e}")

    best_local_reward = -float("inf")

    for episode in range(num_episodes):
        # Entropy decay
        progress = episode / num_episodes
        current_entropy = entropy_start - (entropy_start - entropy_end) * progress
        agent.entropy_coef = current_entropy

        state_dict = env.reset()
        state = preprocess_state_ppo(state_dict)
        agent.reset_hidden()

        episode_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        num_updates = 0

        # Exploration bias setup
        if episode < 50:
            explore_rate = 0.20
        elif episode < 150:
            explore_rate = 0.10
        elif episode < 300:
            explore_rate = 0.05
        else:
            explore_rate = 0.02

        for step in range(max_steps):
            if np.random.random() < explore_rate:
                # 60% chance combat, 40% random
                if np.random.random() < 0.6:
                    action = np.random.choice([5, 7])  # ATTACK, SPELL
                else:
                    action = np.random.randint(0, action_size)
                log_prob = 0.0
                value = 0.0
            else:
                action, log_prob, value = agent.select_action(state)

            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state_ppo(next_state_dict)

            episode_reward += reward
            agent.store_transition(state, action, log_prob, value, reward, done)
            state = next_state

            if len(agent.buffer) >= update_interval:
                metrics = agent.learn()
                if metrics:
                    actor_loss_sum += metrics["actor_loss"]
                    critic_loss_sum += metrics["critic_loss"]
                    num_updates += 1

            if done:
                break

        if len(agent.buffer) > 0:
            metrics = agent.learn()
            if metrics:
                actor_loss_sum += metrics["actor_loss"]
                critic_loss_sum += metrics["critic_loss"]
                num_updates += 1

        avg_loss = (actor_loss_sum + critic_loss_sum) / max(num_updates, 1)
        mantis_killed = next_state_dict.get("mantisLordsKilled", 0)
        boss_hp = next_state_dict.get("bossHealth", 0)

        shared_state.log_episode(
            instance_id,
            episode + 1,
            episode_reward,
            step + 1,
            boss_hp,
            mantis_killed,
            avg_loss,
            current_entropy,
        )

        local_model_path = os.path.join(instance_dir, "latest.pth")
        agent.save(local_model_path)

        if episode_reward > best_local_reward:
            best_local_reward = episode_reward
            shared_state.update_best_model(
                instance_id, episode_reward, local_model_path
            )

        # Sync with Hall of Fame
        if (episode + 1) % sync_interval == 0:
            target_model_path = shared_state.get_any_best_model_path()
            global_best_reward = shared_state.get_global_best_reward()

            if target_model_path and episode_reward < global_best_reward * 0.7:
                try:
                    agent.load(target_model_path)
                    print(f"[Instance {instance_id}] Synced with Hall of Fame model")
                except Exception as e:
                    print(f"[Instance {instance_id}] Could not sync: {e}")

        global_ep = shared_state.increment_episodes()

        if (episode + 1) % 5 == 0:
            print(
                f"[Instance {instance_id}] Ep {episode+1}: R={episode_reward:.1f} | "
                f"Entropy={current_entropy:.3f} | Mantis={mantis_killed}/3 | "
                f"Global Ep={global_ep}"
            )

    env.close()
    print(f"[Instance {instance_id}] Training complete!")


# ============ MAIN ORCHESTRATOR ============
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Instance Training for Hollow Knight AI (Hall of Fame Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "ppo"],
        required=True,
        help="Tipo di agente da usare (dqn o ppo)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help="Numero di istanze da avviare",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5555,
        help="Porta base",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Numero di episodi per istanza",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=10,
        help="Intervallo di sincronizzazione modelli",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory per checkpoint",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints_{args.agent}_multi"

    if args.lr is None:
        args.lr = 1e-5 if args.agent == "dqn" else 1e-4

    if args.gamma is None:
        args.gamma = 0.99 if args.agent == "dqn" else 0.995

    checkpoint_dir_full = os.path.join(AI_AGENTS_DIR, args.checkpoint_dir)

    print("=" * 70)
    print(f"MULTI-INSTANCE TRAINING - {args.agent.upper()} (Top 3 Hall of Fame)")
    print("=" * 70)
    print(f"  Instances:       {args.instances}")
    print(f"  Episodes/Inst:   {args.episodes}")
    print(f"  Sync Interval:   {args.sync_interval}")
    print(f"  Checkpoint Dir:  {checkpoint_dir_full}")
    print("=" * 70)

    print("\nPORTE DA CONFIGURARE:")
    for i in range(args.instances):
        port = args.base_port + i
        print(f"  Istanza {i}: set SYNTHETIC_SOUL_PORT={port}")

    print("\n" + "=" * 70)
    print("Avvia ora le istanze di Hollow Knight con le porte corrette!")
    print("Premi ENTER quando tutte le istanze sono pronte...")
    print("=" * 70)

    input()

    os.makedirs(checkpoint_dir_full, exist_ok=True)

    processes = []

    for i in range(args.instances):
        port = args.base_port + i

        if args.agent == "dqn":
            p = mp.Process(
                target=train_dqn_instance,
                args=(i, port, checkpoint_dir_full, args.episodes, args.sync_interval),
                kwargs={"learning_rate": args.lr, "gamma": args.gamma},
            )
        else:  # ppo
            p = mp.Process(
                target=train_ppo_instance,
                args=(i, port, checkpoint_dir_full, args.episodes, args.sync_interval),
                kwargs={"learning_rate": args.lr, "gamma": args.gamma,
                        },
            )

        processes.append(p)
        p.start()
        print(f"[Main] Started instance {i} on port {port}")
        time.sleep(2)

    print("\n[Main] All instances started. Waiting for completion...")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] Stopping all instances...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()