import os
import sys
import subprocess
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv

# ============ AUTO PLOT GENERATOR IMPORT ============


def auto_generate_plots(
    log_file, checkpoint_dir, algorithm="PPO", window=20, current_episode=0
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
                "--log",
                log_file,
                "--type",
                algorithm.lower(),
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

            # Info file
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
    """Reward Function ottimizzata per Mantis Lords."""

    def __init__(self):
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        self.prev_mantis_killed = 0
        self.episode_start_time = None

    def reset(self, initial_state=None):
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        # Leggi il valore iniziale dall'environment invece di resettare a 0
        if initial_state is not None:
            self.prev_mantis_killed = initial_state.get("mantisLordsKilled", 0)
        else:
            self.prev_mantis_killed = 0
        self.episode_start_time = time.time()

    def calculate_reward(self, state_dict, prev_state, done, info=None):
        reward = 0.0

        if (
            prev_state is not None
            and "bossHealth" in state_dict
            and "bossHealth" in prev_state
        ):
            boss_damage = prev_state["bossHealth"] - state_dict["bossHealth"]
            if boss_damage > 0:
                reward += boss_damage * 3.0

        curr_dist = state_dict.get("distanceToBoss", 100.0)
        if 5.0 <= curr_dist <= 8.0:
            reward += 0.2
        elif curr_dist < 3.0:
            reward -= 0.3
        elif curr_dist > 12.0:
            reward -= 0.5

        if prev_state is not None and "distanceToBoss" in prev_state:
            prev_dist = prev_state["distanceToBoss"]
            distance_change = prev_dist - curr_dist
            if prev_dist > 10.0 and distance_change > 0:
                reward += distance_change * 0.1
            elif prev_dist < 4.0 and distance_change < 0:
                reward += abs(distance_change) * 0.1

        terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
        wall_distance = terrain_info[2] if len(terrain_info) >= 3 else 1.0

        if wall_distance < 0.1:
            reward -= 50.0
            print(f"  [CRITICAL] Wall collision imminent: -{50.0:.2f}")
        elif wall_distance < 0.2:
            reward -= 20.0
            print(f"  [WARNING] Too close to spikes: -{20.0:.2f}")
        elif wall_distance < 0.3:
            reward -= 5.0

        hazards = state_dict.get("nearbyHazards", [])
        for h in hazards:
            if h.get("type") == "spikes":
                spike_dist = h.get("distance", 100.0)
                if spike_dist < 1.5:
                    penalty = 30.0 * np.exp(-spike_dist)
                    reward -= penalty
                    print(f"  [Spike Hazard] Distance {spike_dist:.2f}: -{penalty:.2f}")

        if hazards and prev_state is not None:
            active_projectiles = [
                h for h in hazards if h.get("type") in ["boomerang", "projectile"]
            ]
            if active_projectiles:
                closest_proj = min(
                    active_projectiles,
                    key=lambda h: (h.get("relX", 0) ** 2 + h.get("relY", 0) ** 2)
                    ** 0.5,
                )
                proj_dist = (
                    closest_proj.get("relX", 0) ** 2 + closest_proj.get("relY", 0) ** 2
                ) ** 0.5

                if proj_dist < 3.0:
                    curr_health = state_dict.get("playerHealth", 0)
                    prev_health = prev_state.get("playerHealth", 0)
                    if curr_health == prev_health:
                        dodge_reward = 2.0 * (3.0 - proj_dist)
                        reward += dodge_reward
                        if info and info.get("action_name") == "DASH":
                            reward += 1.0

        if (
            prev_state is not None
            and "playerHealth" in state_dict
            and "playerHealth" in prev_state
        ):
            health_loss = prev_state["playerHealth"] - state_dict["playerHealth"]
            if health_loss > 0:
                reward -= health_loss * 25.0
                print(f"  [Reward] Health lost: -{health_loss * 25.0:.2f}")

        mantis_killed = state_dict.get("mantisLordsKilled", 0)
        if mantis_killed > self.prev_mantis_killed:
            new_kills = mantis_killed - self.prev_mantis_killed
            mantis_bonus = new_kills * 150.0
            reward += mantis_bonus
            print(
                f"  [MANTIS LORD KILLED] +{new_kills} defeated: +{mantis_bonus:.2f} (Total: {mantis_killed}/3)"
            )

        self.prev_mantis_killed = mantis_killed

        reward -= 0.005

        if done:
            if state_dict.get("isDead", False):
                reward -= 150.0
                print("  [Reward] Player died: -150.0")
            elif state_dict.get("bossDefeated", False):
                health_bonus = state_dict.get("playerHealth", 0) * 50.0
                total_victory = 500.0 + health_bonus
                reward += total_victory
                print(f"  [Reward] Boss defeated: +{total_victory:.2f}")

        floor_distance = terrain_info[0] if len(terrain_info) >= 1 else 1.0
        if not state_dict.get("isGrounded", True) and floor_distance > 0.6:
            reward -= 3.0

        if state_dict.get("isFacingBoss", False) and curr_dist < 10.0:
            reward += 0.05

        return reward


def preprocess_state(state_dict):
    """State size: 39 features"""
    features = []

    features.append(state_dict.get("playerX", 0.0))
    features.append(state_dict.get("playerY", 0.0))
    features.append(state_dict.get("playerVelocityX", 0.0))
    features.append(state_dict.get("playerVelocityY", 0.0))
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))
    features.append(float(state_dict.get("hasDoubleJump", False)))

    terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    features.append(state_dict.get("bossX", 0.0))
    features.append(state_dict.get("bossY", 0.0))
    features.append(state_dict.get("bossHealth", 0) / 1000.0)
    features.append(state_dict.get("distanceToBoss", 100.0) / 20.0)
    features.append(float(state_dict.get("isFacingBoss", False)))

    dist = state_dict.get("distanceToBoss", 100.0)
    optimal_zone = 6.5
    zone_deviation = abs(dist - optimal_zone) / 20.0
    features.append(zone_deviation)

    hazards = state_dict.get("nearbyHazards", [])
    for i in range(6):
        if i < len(hazards):
            h = hazards[i]
            rel_x = h.get("relX", 0.0) / 15.0
            rel_y = h.get("relY", 0.0) / 15.0
            features.append(rel_x)
            features.append(rel_y)
            vel_x = h.get("velocityX", 0.0) / 20.0
            features.append(vel_x)
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def train_ppo(
    num_episodes=1000,
    max_steps_per_episode=6000,
    update_timestep=1800,
    learning_rate=2e-4,
    gamma=0.995,
    gae_lambda=0.97,
    clip_param=0.15,
    n_epochs=8,
    save_freq=25,
    checkpoint_dir="checkpoints_ppo",
    host="localhost",
    port=5555,
    plot_freq=100,
):
    """Training PPO con grafici organizzati in sottocartelle."""

    # Salva i checkpoint nella cartella AI_Agents
    ai_agents_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(ai_agents_root, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print(f"[Train PPO] Connecting to Hollow Knight at {host}:{port}...")
    env = HollowKnightEnv(host=host, port=port)

    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Train PPO] State size: {state_size}, Action size: {action_size}")
    print(f"[Train PPO] Grafici organizzati in sottocartelle ogni {plot_freq} episodi")

    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=clip_param,
        n_epochs=n_epochs,
        device=None,
    )

    latest_checkpoint = os.path.join(checkpoint_dir_full, "latest_ppo.pth")
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print("[Train PPO] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train PPO] Could not load checkpoint: {e}")

    reward_calc = RewardCalculator()
    episode_rewards = []
    best_reward = -float("inf")
    global_step = 0

    log_file = os.path.join(checkpoint_dir_full, "training_log.txt")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("episode,total_reward,steps,global_step,mantis_killed\n")

    print(f"\n{'='*60}")
    print(f"Starting PPO Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset(initial_state=state_dict)
        agent.reset_hidden()

        episode_reward = 0.0
        prev_state_dict = None

        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        for step in range(max_steps_per_episode):
            global_step += 1

            action, log_prob, val = agent.select_action(state)
            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            reward = reward_calc.calculate_reward(
                next_state_dict, prev_state_dict, done, info
            )
            episode_reward += reward

            agent.store_transition(state, action, log_prob, val, reward, done)

            if global_step % update_timestep == 0:
                print(f"  [PPO Update] Updating policy at step {global_step}...")
                agent.learn()

            state = next_state
            prev_state_dict = state_dict
            state_dict = next_state_dict

            if step % 100 == 0:
                print(
                    f"  [Step {step}] Reward: {episode_reward:.2f}, Global Step: {global_step}"
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

        episode_rewards.append(episode_reward)
        # Usa il valore tracciato dalla RewardCalculator invece di state_dict
        mantis_killed = reward_calc.prev_mantis_killed
        print(f"\n[Episode {episode + 1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps in Episode: {step + 1}")
        print(f"  Global Steps: {global_step}")
        print(f"  Mantis Lords Killed: {mantis_killed}/3")

        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{mantis_killed}\n"
            )

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir_full, "best_model_ppo.pth")
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
                algorithm="PPO",
                window=min(20, max(10, (episode + 1) // 50)),
                current_episode=episode + 1,
            )
            print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print("PPO Training Completed!")
    print(f"{'='*60}")
    print(f"Best Reward: {best_reward:.2f}")

    final_path = os.path.join(checkpoint_dir_full, "final_model_ppo.pth")
    agent.save(final_path)
    env.close()


if __name__ == "__main__":
    HYPERPARAMS = {
        "num_episodes": 1000,
        "max_steps_per_episode": 6000,
        "update_timestep": 1800,
        "learning_rate": 2e-4,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "clip_param": 0.15,
        "n_epochs": 8,
        "save_freq": 25,
        "checkpoint_dir": "checkpoints_ppo_mantis",
        "host": "localhost",
        "port": 5555,
        "plot_freq": 100,
    }

    print("=" * 60)
    print("PPO Training - Mantis Lords - ORGANIZED PLOTS")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ Auto-generated plots every 100 episodes")
    print("  ✓ Organized in subfolders (episode_100, episode_200, ...)")
    print("  ✓ Final plots at episode 1000")
    print("  ✓ Mantis Lords progress tracking")
    print("\nStructure:")
    print("  plots_ppo/")
    print("  ├── episode_100/")
    print("  ├── episode_200/")
    print("  ├── ...")
    print("  └── episode_1000_final/")
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    try:
        train_ppo(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train PPO] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train PPO] Error during training: {e}")
        import traceback

        traceback.print_exc()
