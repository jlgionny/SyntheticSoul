import os
import sys
import subprocess
import time
import numpy as np
from datetime import datetime

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


class RewardCalculator:
    """Simplified Reward Function - Combat-Focused con Mantis Lords tracking"""

    def __init__(self):
        self.prev_boss_health = None
        self.episode_start_time = None
        self.steps_in_combat_range = 0
        self.current_step = 0
        self.prev_mantis_killed = 0

    def reset(self):
        self.prev_boss_health = None
        self.episode_start_time = time.time()
        self.steps_in_combat_range = 0
        self.current_step = 0
        self.prev_mantis_killed = 0

    def calculate_reward(self, state_dict, prev_state, done, info=None):
        reward = 0.0
        self.current_step += 1

        # DAMAGE TO BOSS
        if (
            prev_state is not None
            and "bossHealth" in state_dict
            and "bossHealth" in prev_state
        ):
            boss_damage = prev_state["bossHealth"] - state_dict["bossHealth"]
            if boss_damage > 0:
                reward += 20.0
                print(f"  [Reward] Boss hit: +20.0 ({boss_damage} HP damage)")

        # DAMAGE TAKEN
        damage_taken = state_dict.get("damageTaken", 0)
        if damage_taken > 0:
            damage_penalty = damage_taken * 15.0
            reward -= damage_penalty
            print(f"  [Reward] Damage taken: -{damage_penalty:.1f}")

        # COMBAT RANGE BONUS
        curr_dist = state_dict.get("distanceToBoss", 100.0)
        if 2.0 <= curr_dist <= 6.0:
            reward += 0.1
            self.steps_in_combat_range += 1
        else:
            self.steps_in_combat_range = 0

        # ATTACK INCENTIVE
        if info is not None and "action_name" in info:
            action_name = info["action_name"]
            if action_name == "ATTACK" and 2.0 <= curr_dist <= 6.0:
                reward += 0.2

        # WALL STUCK PENALTY
        terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
        player_vel_x = state_dict.get("playerVelocityX", 0.0)
        player_vel_y = state_dict.get("playerVelocityY", 0.0)

        if len(terrain_info) >= 3:
            wall_distance = terrain_info[2]
            is_stuck = (
                abs(player_vel_x) < 0.1
                and abs(player_vel_y) < 0.1
                and wall_distance < 0.1
            )
            if is_stuck:
                reward -= 0.5

        # ENCOURAGE CENTER MOVEMENT IF FAR
        if curr_dist > 8.0 and len(terrain_info) >= 3:
            wall_distance = terrain_info[2]
            if wall_distance > 0.4:
                reward += 0.05

        # MANTIS LORDS KILL REWARD
        mantis_killed = state_dict.get("mantisLordsKilled", 0)
        if mantis_killed > self.prev_mantis_killed:
            new_kills = mantis_killed - self.prev_mantis_killed
            mantis_bonus = new_kills * 150.0
            reward += mantis_bonus
            print(
                f"  [MANTIS LORD KILLED] +{new_kills} defeated: +{mantis_bonus:.2f} (Total: {mantis_killed}/3)"
            )
        self.prev_mantis_killed = mantis_killed

        # TERMINAL STATE REWARDS
        if done:
            if state_dict.get("isDead", False):
                reward -= 50.0
                print("  [Reward] Episode end (death): -50.0")
            elif state_dict.get("bossDefeated", False):
                reward += 500.0
                elapsed = (
                    time.time() - self.episode_start_time
                    if self.episode_start_time
                    else 0
                )
                time_bonus = max(0, 100 - elapsed / 10)
                reward += time_bonus
                print(f"  [Reward] BOSS DEFEATED: +{500 + time_bonus:.1f}")

        return reward


def preprocess_state(state_dict: dict) -> np.ndarray:
    """
    MINIMAL STATE (10 features) - NO raycast (broken).
    ONLY boss position + player status.
    """
    features = []

    # Player basics (4)
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # Boss (4) - LA SOLA COSA CHE FUNZIONA
    boss_rel_x = state_dict.get("bossRelativeX", 0.0)
    boss_rel_y = state_dict.get("bossRelativeY", 0.0)
    distance = state_dict.get("distanceToBoss", 50.0) / 50.0
    facing_boss = float(state_dict.get("isFacingBoss", False))

    features.append(boss_rel_x)
    features.append(boss_rel_y)
    features.append(np.clip(distance, 0.0, 1.0))
    features.append(facing_boss)

    # Hazard più vicino (2) - posizione
    hazards = state_dict.get("nearbyHazards", [])
    if len(hazards) > 0:
        h = hazards[0]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def train_dqn(
    num_episodes=1000,
    max_steps_per_episode=5000,
    batch_size=64,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=10000,
    target_update_freq=1000,
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
    env = HollowKnightEnv(host=host, port=port)

    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Train] State size: {state_size}, Action size: {action_size}")
    print("[Train] Using SIMPLIFIED reward system (combat-focused)")
    print(f"[Train] Grafici organizzati in sottocartelle ogni {plot_freq} episodi")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,
    )

    latest_checkpoint = os.path.join(checkpoint_dir_full, "latest.pth")
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print("[Train] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train] Could not load checkpoint: {e}")

    reward_calc = RewardCalculator()
    episode_rewards = []
    best_reward = -float("inf")

    log_file = os.path.join(checkpoint_dir_full, "training_log.txt")

    # FIX: Header aggiornato come PPO
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
        reward_calc.reset()

        episode_reward = 0.0
        episode_loss = []
        prev_state_dict = None
        total_damage_taken = 0

        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        for step in range(max_steps_per_episode):
            global_step += 1

            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            reward = reward_calc.calculate_reward(
                next_state_dict, prev_state_dict, done, info
            )

            episode_reward += reward
            damage_this_step = next_state_dict.get("damageTaken", 0)
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
            prev_state_dict = state_dict
            state_dict = next_state_dict

            if step % 100 == 0:
                curr_dist = next_state_dict.get("distanceToBoss", 0)
                in_range = "✓" if 2.0 <= curr_dist <= 6.0 else "✗"
                print(
                    f"  [Step {step}] Reward: {episode_reward:.2f}, Dist: {curr_dist:.1f} {in_range}, Dmg: {total_damage_taken}"
                )

            if done:
                reason = (
                    "Player died"
                    if state_dict.get("isDead")
                    else (
                        "Boss defeated" if state_dict.get("bossDefeated") else "Unknown"
                    )
                )
                print(f"  [Episode End] Reason: {reason}")
                break

        agent.episodes_done += 1
        episode_rewards.append(episode_reward)

        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        mantis_killed = reward_calc.prev_mantis_killed

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
        print(f"  Mantis Lords Killed: {mantis_killed}/3")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        print(f"  Combat Range Steps: {reward_calc.steps_in_combat_range}")
        print(f"  Total Damage Taken: {total_damage_taken}")

        # FIX: Salva metriche come PPO
        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{mantis_killed},{avg_loss:.4f},{epsilon:.4f}\n"
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

        # ============ AUTO GENERATE PLOTS IN ORGANIZED FOLDERS ============
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
    print("DQN Training - Mantis Lords - ORGANIZED PLOTS")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ Auto-generated plots every 100 episodes")
    print("  ✓ Organized in subfolders (episode_100, episode_200, ...)")
    print("  ✓ Final plots at episode 1000")
    print("  ✓ Mantis Lords progress tracking")
    print("  ✓ Simplified combat-focused reward")
    print("\nStructure:")
    print("  plots_dqn/")
    print("    ├── episode_100/")
    print("    ├── episode_200/")
    print("    ├── ...")
    print("    └── episode_1000_final/")
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    try:
        train_dqn(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train] Error during training: {e}")
        import traceback

        traceback.print_exc()
