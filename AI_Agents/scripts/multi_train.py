"""
Multi-Instance Training Orchestrator for Hollow Knight AI.

Permette di:
- Scegliere l'agente (DQN o PPO)
- Avviare multiple istanze in parallelo
- Condividere experience buffer e modelli tra istanze
- Sincronizzare i best model automaticamente
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

# Setup paths - Aggiungi la directory AI_Agents al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_AGENTS_DIR = os.path.dirname(SCRIPT_DIR)  # AI_Agents/
sys.path.insert(0, AI_AGENTS_DIR)

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv


# ============ FILE-BASED SHARED STATE ============
class FileBasedSharedState:
    """
    Stato condiviso tra istanze usando file system.
    Thread-safe tramite file locking.
    """

    def __init__(self, checkpoint_dir: str, agent_type: str):
        self.checkpoint_dir = checkpoint_dir
        self.agent_type = agent_type

        # Setup directories
        os.makedirs(checkpoint_dir, exist_ok=True)

        # File paths
        self.log_file = os.path.join(checkpoint_dir, "training_log.txt")
        self.state_file = os.path.join(checkpoint_dir, "shared_state.json")
        self.lock_file = os.path.join(checkpoint_dir, ".lock")
        self.best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        self.best_meta_path = os.path.join(checkpoint_dir, "best_model_meta.json")

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
            self._write_state(
                {"best_reward": -float("inf"), "total_episodes": 0, "best_instance": -1}
            )

    def _read_state(self) -> dict:
        """Read shared state from file."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "best_reward": -float("inf"),
                "total_episodes": 0,
                "best_instance": -1,
            }

    def _write_state(self, state: dict):
        """Write shared state to file."""
        with open(self.state_file, "w") as f:
            json.dump(state, f)

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
        self, instance_id: int, reward: float, model_path: str
    ) -> bool:
        """Update best model if this instance achieved better reward."""
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()

            if reward > state["best_reward"]:
                state["best_reward"] = reward
                state["best_instance"] = instance_id
                self._write_state(state)

                # Copy model
                if os.path.exists(model_path):
                    shutil.copy(model_path, self.best_model_path)

                    # Save metadata
                    with open(self.best_meta_path, "w") as f:
                        json.dump(
                            {
                                "reward": reward,
                                "instance": instance_id,
                                "timestamp": datetime.now().isoformat(),
                                "total_episodes": state["total_episodes"],
                            },
                            f,
                            indent=2,
                        )

                print(
                    f"\n[Instance {instance_id}] NEW GLOBAL BEST! Reward: {reward:.2f}"
                )
                return True

        return False

    def get_best_reward(self) -> float:
        """Get current best reward."""
        state = self._read_state()
        return state.get("best_reward", -float("inf"))

    def get_best_model_path(self) -> Optional[str]:
        """Get path to best model if exists."""
        if os.path.exists(self.best_model_path):
            return self.best_model_path
        return None

    def increment_episodes(self) -> int:
        """Increment global episode counter."""
        lock = filelock.FileLock(self.lock_file, timeout=10)
        with lock:
            state = self._read_state()
            state["total_episodes"] = state.get("total_episodes", 0) + 1
            self._write_state(state)
            return state["total_episodes"]


# ============ PREPROCESSING FUNCTIONS ============
def preprocess_state_dqn(state_dict: dict) -> np.ndarray:
    features = []

    # 1. Player Status
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # 2. Player Velocity
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # 3. Terrain (Normalizzato per vedere meglio i muri/spuntoni)
    terrain_info = state_dict.get("terrainInfo", [10.0] * 5)
    if not terrain_info or len(terrain_info) < 5:
        terrain_info = [10.0] * 5
    # Dividiamo per 20.0 (distanza massima vista) per avere valori 0-1
    features.extend([np.clip(t / 20.0, 0.0, 1.0) for t in terrain_info[:5]])

    # 4. Boss Position
    boss_rel_x = state_dict.get("bossRelativeX", 0.0)
    boss_rel_y = state_dict.get("bossRelativeY", 0.0)
    distance = state_dict.get("distanceToBoss", 50.0) / 50.0
    facing_boss = float(state_dict.get("isFacingBoss", False))

    features.append(np.clip(boss_rel_x / 30.0, -1.0, 1.0))
    features.append(np.clip(boss_rel_y / 30.0, -1.0, 1.0))
    features.append(np.clip(distance, 0.0, 1.0))
    features.append(facing_boss)

    # 5. Boss Velocity & Status
    features.append(np.clip(state_dict.get("bossVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossVelocityY", 0.0) / 20.0, -1.0, 1.0))
    features.append(state_dict.get("bossHealth", 100.0) / 100.0)
    features.append(state_dict.get("mantisLordsKilled", 0) / 3.0)

    # 6. HAZARDS (Doppio Occhio per i Boomerang)
    hazards = state_dict.get("nearbyHazards", [])

    # Hazard 1 (Il più vicino)
    if len(hazards) > 0:
        h = hazards[0]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))
    else:
        features.extend([0.0] * 5)

    # Hazard 2 (Il secondo boomerang - CRUCIALE!)
    if len(hazards) > 1:
        h = hazards[1]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)




def preprocess_state_ppo(state_dict: dict) -> np.ndarray:
    """
    ENHANCED STATE (26 features) - Combat-focused.
    Include informazioni su boss velocity e stato combat.
    """
    features = []

    # Player basics (5) - Status + soul
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)  # Soul per spell
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # Player velocity (2) - Per capire momentum
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # Terrain (5) - raycasts
    terrain_info = state_dict.get("terrainInfo", [1.0] * 5)
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    # Boss position (4) - Direzione e distanza
    boss_rel_x = state_dict.get("bossRelativeX", 0.0)
    boss_rel_y = state_dict.get("bossRelativeY", 0.0)
    distance = state_dict.get("distanceToBoss", 50.0) / 50.0
    facing_boss = float(state_dict.get("isFacingBoss", False))

    features.append(np.clip(boss_rel_x / 30.0, -1.0, 1.0))
    features.append(np.clip(boss_rel_y / 30.0, -1.0, 1.0))
    features.append(np.clip(distance, 0.0, 1.0))
    features.append(facing_boss)

    # Boss velocity (2) - Per prevedere movimento
    features.append(np.clip(state_dict.get("bossVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # Boss health (1) - Per tracking progresso
    features.append(state_dict.get("bossHealth", 100.0) / 100.0)

    # Mantis killed (1) - Fase del fight
    features.append(state_dict.get("mantisLordsKilled", 0) / 3.0)

    # Hazards (5) - Il piÃ¹ vicino
    hazards = state_dict.get("nearbyHazards", [])
    if len(hazards) > 0:
        h = hazards[0]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 1.0])

    if len(hazards) > 1:
        h = hazards[1]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
    else:
        features.extend([0.0] * 2)

    return np.array(features, dtype=np.float32)


# ============ HYPERPARAMETERS INSTANCE WORKER FUNCTIONS ============
def train_dqn_instance(
    instance_id: int,
    port: int,
    checkpoint_dir: str,
    num_episodes: int,
    sync_interval: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 100000,
    max_steps: int = 3000,
):
    """Worker function per training DQN di una singola istanza."""

    print(f"\n[Instance {instance_id}] Starting DQN training on port {port}")

    # Create shared state handler
    shared_state = FileBasedSharedState(checkpoint_dir, "dqn")

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

    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,  # Aumentato
    )

# Load best model if exists
    best_model_path = shared_state.get_best_model_path()
    if best_model_path:
        try:
            agent.load(best_model_path)
            print(f"[Instance {instance_id}] Loaded shared best model")
            agent.steps_done = 0
        except Exception as e:
            print(f"[Instance {instance_id}] Could not load best model: {e}")

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

        # Update global best if needed (Se noi siamo i migliori, aggiorniamo gli altri)
        if episode_reward > best_local_reward:
            best_local_reward = episode_reward
            shared_state.update_best_model(
                instance_id, episode_reward, local_model_path
            )

        # Sync with best model periodically (Se noi siamo scarsi, impariamo dagli altri)
        if (episode + 1) % sync_interval == 0:
            best_model_path = shared_state.get_best_model_path()
            best_reward = shared_state.get_best_reward()

            # Scarica solo se il modello globale è significativamente migliore del nostro risultato attuale
            if best_model_path and episode_reward < best_reward * 0.8:
                try:
                    # === FIX IMPORTANTE INIZIO ===
                    # 1. Salviamo i contatori attuali per NON resettare l'epsilon
                    current_steps = agent.steps_done
                    current_eps = agent.episodes_done

                    # 2. Carichiamo i pesi del cervello migliore
                    agent.load(best_model_path)

                    # 3. Ripristiniamo i nostri contatori (l'esperienza di esplorazione resta la nostra)
                    agent.steps_done = current_steps
                    agent.episodes_done = current_eps
                    # === FIX IMPORTANTE FINE ===

                    print(
                        f"[Instance {instance_id}] Synced with best model (reward: {best_reward:.2f})"
                    )
                except Exception as e:
                    print(
                        f"[Instance {instance_id}] Could not sync with best model: {e}"
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
    learning_rate: float = 1e-4,  # Ridotto per stabilitÃ
    gamma: float = 0.995,  # Aumentato per reward sparse
    entropy_start: float = 0.05,  # Inizio moderato per esplorazione
    entropy_end: float = 0.01,   # Fine basso per sfruttamento
    update_interval: int = 1024,  # Ridotto per update piÃ¹ frequenti
    max_steps: int = 6000,
):
    """Worker function per training PPO di una singola istanza."""

    print(f"\n[Instance {instance_id}] Starting PPO V2 training on port {port}")

    # Create shared state handler
    shared_state = FileBasedSharedState(checkpoint_dir, "ppo")

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
    state_array = preprocess_state_ppo(initial_state)
    state_size = len(state_array)
    action_size = 8  # 8 azioni (IDLE rimosso)

    # Initialize agent with LSTM
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=0.97,  # Aumentato per better credit assignment
        entropy_coef=entropy_start,
        use_lstm=True,  # Abilitato per pattern recognition
        n_epochs=6,  # Aumentato per better learning
        batch_size=128,  # Aumentato
    )

    # Load best model if exists
    best_model_path = shared_state.get_best_model_path()
    if best_model_path:
        try:
            agent.load(best_model_path)
            print(f"[Instance {instance_id}] Loaded shared best model")
        except Exception as e:
            print(f"[Instance {instance_id}] Could not load best model: {e}")

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

        # Exploration rate con decadimento graduale
        if episode < 50:
            explore_rate = 0.20  # 20% warmup
        elif episode < 150:
            explore_rate = 0.10  # 10% fase intermedia
        elif episode < 300:
            explore_rate = 0.05  # 5% raffinamento
        else:
            explore_rate = 0.02  # 2% mantenimento minimo

        for step in range(max_steps):
            # SMART EXPLORATION: bias verso azioni combat
            if np.random.random() < explore_rate:
                # 60% chance di azione combat, 40% movimento
                if np.random.random() < 0.6:
                    action = np.random.choice([5, 7])  # ATTACK, SPELL
                else:
                    action = np.random.randint(0, action_size)
                log_prob = 0.0
                value = 0.0
            else:
                action, log_prob, value = agent.select_action(state)

            # Environment step
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state_ppo(next_state_dict)

            episode_reward += reward

            # Store transition
            agent.store_transition(state, action, log_prob, value, reward, done)

            state = next_state

            # Update policy
            if len(agent.buffer) >= update_interval:
                metrics = agent.learn()
                if metrics:
                    actor_loss_sum += metrics["actor_loss"]
                    critic_loss_sum += metrics["critic_loss"]
                    num_updates += 1

            if done:
                break

        # Final update
        if len(agent.buffer) > 0:
            metrics = agent.learn()
            if metrics:
                actor_loss_sum += metrics["actor_loss"]
                critic_loss_sum += metrics["critic_loss"]
                num_updates += 1

        # Episode complete
        avg_loss = (actor_loss_sum + critic_loss_sum) / max(num_updates, 1)
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
            current_entropy,
        )

        # Save local model
        local_model_path = os.path.join(instance_dir, "latest.pth")
        agent.save(local_model_path)

        # Update global best if needed
        if episode_reward > best_local_reward:
            best_local_reward = episode_reward
            shared_state.update_best_model(
                instance_id, episode_reward, local_model_path
            )

        # Sync with best model periodically
        if (episode + 1) % sync_interval == 0:
            best_model_path = shared_state.get_best_model_path()
            best_reward = shared_state.get_best_reward()
            if best_model_path and episode_reward < best_reward * 0.7:
                try:
                    agent.load(best_model_path)
                    print(f"[Instance {instance_id}] Synced with best model")
                except Exception as e:
                    print(
                        f"[Instance {instance_id}] Could not sync with best model: {e}"
                    )

        global_ep = shared_state.increment_episodes()

        # Progress log
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
        description="Multi-Instance Training for Hollow Knight AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Training DQN con 2 istanze
  python multi_train.py --agent dqn --instances 2 --episodes 500

  # Training PPO con 3 istanze su porte custom
  python multi_train.py --agent ppo --instances 3 --base-port 5560

  # Training singola istanza (equivalente a train_dqn.py)
  python multi_train.py --agent dqn --instances 1

Note:
  - Ogni istanza richiede una copia di Hollow Knight in esecuzione
  - Imposta SYNTHETIC_SOUL_PORT=<port> prima di avviare ogni gioco
  - Le istanze condividono automaticamente il best model e l'experience buffer
        """,
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
        help="Numero di istanze da avviare (default: 1)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5555,
        help="Porta base (istanze useranno base, base+1, base+2, ...)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Numero di episodi per istanza (default: 1000)",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=10,
        help="Intervallo di sincronizzazione modelli (default: 10 episodi)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory per checkpoint (default: checkpoints_<agent>_multi)",
    )

    # Hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-5 per DQN, 5e-4 per PPO)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 0.99 per DQN, 0.98 per PPO)",
    )

    args = parser.parse_args()

    # Set defaults based on agent type
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints_{args.agent}_multi"

    if args.lr is None:
        args.lr = 1e-5 if args.agent == "dqn" else 1e-4  # PPO lr ridotto

    if args.gamma is None:
        args.gamma = 0.99 if args.agent == "dqn" else 0.995  # PPO gamma aumentato

    # Full path for checkpoint dir
    checkpoint_dir_full = os.path.join(AI_AGENTS_DIR, args.checkpoint_dir)

    print("=" * 70)
    print(f"MULTI-INSTANCE TRAINING - {args.agent.upper()}")
    print("=" * 70)
    print(f"  Agent Type:      {args.agent.upper()}")
    print(f"  Instances:       {args.instances}")
    print(f"  Base Port:       {args.base_port}")
    print(f"  Episodes/Inst:   {args.episodes}")
    print(f"  Sync Interval:   {args.sync_interval}")
    print(f"  Checkpoint Dir:  {checkpoint_dir_full}")
    print(f"  Learning Rate:   {args.lr}")
    print(f"  Gamma:           {args.gamma}")
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

    # Create checkpoint directory
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    # Start worker processes
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
                kwargs={"learning_rate": args.lr, "gamma": args.gamma},
            )

        processes.append(p)
        p.start()
        print(f"[Main] Started instance {i} on port {port}")
        time.sleep(2)  # Stagger starts to avoid connection issues

    # Wait for all processes
    print("\n[Main] All instances started. Waiting for completion...")
    print("[Main] Press Ctrl+C to stop all instances\n")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] Stopping all instances...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    # Read final state
    state_file = os.path.join(checkpoint_dir_full, "shared_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            final_state = json.load(f)
        print(f"  Total Episodes:  {final_state.get('total_episodes', 'N/A')}")
        print(f"  Best Reward:     {final_state.get('best_reward', 'N/A'):.2f}")

    print(f"  Best Model:      {os.path.join(checkpoint_dir_full, 'best_model.pth')}")
    print(f"  Training Log:    {os.path.join(checkpoint_dir_full, 'training_log.txt')}")
    print("=" * 70)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()