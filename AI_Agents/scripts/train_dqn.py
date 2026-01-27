import os
import sys
import subprocess
import math
import numpy as np
from datetime import datetime
import torch  # Importato qui per il check GPU

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.dqn_agent import DQNAgent
from src.env.hollow_knight_env import HollowKnightEnv


# ============ AUTO PLOT GENERATOR IMPORT ============
def auto_generate_plots(
    log_file, checkpoint_dir, algorithm="DQN", window=20, current_episode=0
):
    """Genera automaticamente grafici in sottocartelle organizzate."""
    if current_episode == 1000:
        episode_folder = f"episode_{current_episode}_final"
    else:
        episode_folder = f"episode_{current_episode}"

    plots_dir = os.path.join("..", f"plots_{algorithm.lower()}", episode_folder)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_file):
        print(f"[Auto Plot] Warning: Log file non trovato: {log_file}")
        return

    script_path = os.path.join(os.path.dirname(__file__), "generate_plots.py")
    if not os.path.exists(script_path):
        print("[Auto Plot] Warning: Script generate_plots.py non trovato")
        return

    try:
        print("\n[Auto Plot] Generazione grafici in corso...")
        print(f"[Auto Plot] Cartella: {plots_dir}")
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--mode",
                algorithm.lower(),
                f"--{algorithm.lower()}-log",
                log_file,
                "--output",
                plots_dir,
                "--window",
                str(window),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print(f"[Auto Plot] ✓ Grafici generati in: {plots_dir}")
            info_path = os.path.join(plots_dir, "info.txt")
            with open(info_path, "w") as f:
                f.write("Training Snapshot\n")
                f.write("================\n")
                f.write(f"Algorithm: {algorithm}\n")
                f.write(f"Episode: {current_episode}\n")
                f.write(f"Smoothing Window: {window}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print("[Auto Plot] ✗ Errore durante generazione:")
            if result.stderr:
                print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print("[Auto Plot] ✗ Timeout durante generazione grafici")
    except Exception as e:
        print(f"[Auto Plot] ✗ Errore: {e}")


def preprocess_state(state_dict):
    """Optimized state preprocessing - 28 features"""
    features = []

    # PLAYER FEATURES (10)
    player_x = state_dict.get("playerX", 0.0)
    player_y = state_dict.get("playerY", 0.0)
    features.append(player_x / 40.0)
    features.append(player_y / 30.0)
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 10.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 10.0, -1.0, 1.0))
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))
    features.append(float(state_dict.get("hasDoubleJump", False)))

    # TERRAIN INFO (5)
    terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    # BOSS FEATURES (7)
    boss_x = state_dict.get("bossX", 0.0)
    boss_y = state_dict.get("bossY", 0.0)
    boss_relative_x = (boss_x - player_x) / 40.0
    boss_relative_y = (boss_y - player_y) / 30.0
    features.append(boss_relative_x)
    features.append(boss_relative_y)
    features.append(state_dict.get("bossHealth", 0) / 1000.0)
    distance_to_boss = state_dict.get("distanceToBoss", 50.0)
    features.append(np.clip(distance_to_boss / 50.0, 0.0, 1.0))
    angle_to_boss = math.atan2(boss_relative_y, boss_relative_x) / math.pi
    features.append(angle_to_boss)
    features.append(float(state_dict.get("isFacingBoss", False)))
    boss_vel_x = state_dict.get("bossVelocityX", 0.0)
    features.append(np.clip(boss_vel_x / 10.0, -1.0, 1.0))

    state_dict["bossRelativeX"] = boss_relative_x
    state_dict["bossRelativeY"] = boss_relative_y

    # HAZARDS (6 features - top 2)
    hazards = state_dict.get("nearbyHazards", [])
    if hazards:
        hazards_with_dist = []
        for h in hazards:
            rel_x = h.get("relX", 0.0)
            rel_y = h.get("relY", 0.0)
            dist = math.sqrt(rel_x**2 + rel_y**2)
            hazards_with_dist.append((dist, h))
        hazards_with_dist.sort(key=lambda x: x[0])
        sorted_hazards = [h for _, h in hazards_with_dist[:2]]
    else:
        sorted_hazards = []

    for i in range(2):
        if i < len(sorted_hazards):
            h = sorted_hazards[i]
            rel_x = h.get("relX", 0.0) / 30.0
            rel_y = h.get("relY", 0.0) / 30.0
            features.append(np.clip(rel_x, -1.0, 1.0))
            features.append(np.clip(rel_y, -1.0, 1.0))
            hazard_vel_x = h.get("velocityX", 0.0)
            hazard_vel_y = h.get("velocityY", 0.0)
            player_vel_x = state_dict.get("playerVelocityX", 0.0)
            player_vel_y = state_dict.get("playerVelocityY", 0.0)
            rel_vel_magnitude = math.sqrt(
                (hazard_vel_x - player_vel_x) ** 2 + (hazard_vel_y - player_vel_y) ** 2
            )
            features.append(np.clip(rel_vel_magnitude / 20.0, 0.0, 1.0))
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def train_dqn(
    num_episodes=1000,
    max_steps_per_episode=5000,
    batch_size=64,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=200000,
    target_update_freq=2000,
    save_freq=50,
    checkpoint_dir="checkpoints_dqn_mantis",
    host="localhost",
    port=5555,
    plot_freq=100,
):
    """Main training loop with organized auto-plot generation."""
    # Salva i checkpoint nella cartella AI_Agents
    ai_agents_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(ai_agents_root, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print(f"[Train] Connecting to Hollow Knight at {host}:{port}...")

    # MODIFICA: use_reward_shaping=True delega il calcolo reward all'ambiente
    env = HollowKnightEnv(host=host, port=port, use_reward_shaping=True)

    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Train] State size: {state_size}, Action size: {action_size}")
    print("[Train] Using ENV-BASED reward system (Centralized logic)")
    print(f"[Train] Grafici organizzati in sottocartelle ogni {plot_freq} episodi")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,
    )

    # --- GPU CHECK ---
    print(f"\n{'='*40}")
    print(f"🔥 DISPOSITIVO RILEVATO: {agent.device} 🔥")
    if str(agent.device) == "cuda":
        print(f"   Scheda Video: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️ ATTENZIONE: Stai usando la CPU!")
    print(f"{'='*40}\n")
    # -----------------

    latest_checkpoint = os.path.join(checkpoint_dir_full, "latest.pth")
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print("[Train] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train] Could not load checkpoint: {e}")

    # Rimosso RewardCalculator() - ora è tutto nell'env
    episode_rewards = []
    best_reward = -float("inf")

    log_file = os.path.join(checkpoint_dir_full, "training_log.txt")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(
                "episode,total_reward,steps,global_step,mantis_killed,avg_loss,epsilon\n"
            )

    print(f"\n{'='*60}")
    print(f"Starting DQN Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    global_step = 0

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = preprocess_state(state_dict)

        episode_reward = 0.0
        episode_loss = []
        total_damage_taken = 0

        # Recupero kills iniziali per tracciare progresso episodio

        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        for step in range(max_steps_per_episode):
            global_step += 1

            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            # MODIFICA: Ora step restituisce 4 valori
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            # Il reward è già calcolato dall'ambiente, non serve RewardCalculator

            episode_reward += reward
            # Info extra per log
            damage_this_step = (
                info.get("damage_taken", 0) if "damage_taken" in info else 0
            )
            total_damage_taken += damage_this_step

            agent.store_transition(state, action, reward, next_state, done)

            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size=batch_size)
                if loss is not None:
                    episode_loss.append(loss)

            if agent.steps_done > 0 and agent.steps_done % target_update_freq == 0:
                agent.update_target_network()
                print(f"  [Step {agent.steps_done}] Target network updated")

            state = next_state
            # Non serve più prev_state_dict per il calcolo reward locale

            if step % 100 == 0:
                curr_dist = next_state_dict.get("distanceToBoss", 0)
                in_range = "✓" if 2.0 <= curr_dist <= 6.0 else "✗"
                print(
                    f"  [Step {step}] Reward: {episode_reward:.2f}, Dist: {curr_dist:.1f} {in_range}, TotDmgTaken: {total_damage_taken}"
                )

            if done:
                reason = (
                    "Player died"
                    if next_state_dict.get("isDead")
                    else (
                        "Boss defeated"
                        if next_state_dict.get("bossDefeated")
                        else "Unknown"
                    )
                )
                print(f"  [Episode End] Reason: {reason}")
                break

        agent.episodes_done += 1
        episode_rewards.append(episode_reward)

        avg_loss = np.mean(episode_loss) if episode_loss else 0.0

        # Tracking Mantis Killed dall'ultimo stato ricevuto
        current_mantis_killed = next_state_dict.get("mantisLordsKilled", 0)

        avg_reward_last_10 = (
            np.mean(episode_rewards[-10:])
            if len(episode_rewards) >= 10
            else episode_reward
        )

        print(f"\n[Episode {episode + 1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Steps: {step + 1}")
        print(f"  Global Steps: {global_step}")
        print(f"  Epsilon: {epsilon:.4f}")
        print(f"  Mantis Lords Killed: {current_mantis_killed}/3")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        print(f"  Total Damage Taken: {total_damage_taken}")

        # Log CSV
        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{current_mantis_killed},{avg_loss:.4f},{epsilon:.4f}\n"
            )

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir_full, "best_model.pth")
            agent.save(best_path)
            print(f"  [NEW BEST] Saved to {best_path}")

        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir_full, f"episode_{episode + 1}.pth"
            )
            agent.save(checkpoint_path)
            agent.save(latest_checkpoint)
            print(f"  [Checkpoint] Saved to {checkpoint_path}")

        # ============ AUTO GENERATE PLOTS ============
        if (episode + 1) % plot_freq == 0 or (episode + 1) == num_episodes:
            print(f"\n{'='*60}")
            print(f"[PLOTS] Generazione grafici episodio {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            auto_generate_plots(
                log_file=log_file,
                checkpoint_dir=checkpoint_dir_full,
                algorithm="DQN",
                window=min(20, max(10, (episode + 1) // 50)),
                current_episode=episode + 1,
            )
            print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print("DQN Training Completed!")
    print(f"{'='*60}")
    print(f"Best Reward: {best_reward:.2f}")

    final_path = os.path.join(checkpoint_dir_full, "final_model.pth")
    agent.save(final_path)
    env.close()


if __name__ == "__main__":
    HYPERPARAMS = {
        "num_episodes": 1000,
        "max_steps_per_episode": 5000,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 10000,
        "target_update_freq": 1000,
        "save_freq": 50,
        "checkpoint_dir": "checkpoints",
        "host": "localhost",
        "port": 5555,
        "plot_freq": 100,
    }

    print("=" * 60)
    print("DQN Training - Mantis Lords - ENV REWARDS")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ GPU Accelerated")
    print("  ✓ Reward logic moved to Environment")
    print("  ✓ Auto-generated plots every 100 episodes")
    print("  ✓ Mantis Lords progress tracking")
    print("=" * 60)

    try:
        train_dqn(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train] Error during training: {e}")
        import traceback

        traceback.print_exc()
