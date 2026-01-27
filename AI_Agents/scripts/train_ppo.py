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
    """Reward Function STABILE con FOCUS su attaccare"""

    def __init__(self):
        self.prev_mantis_killed = 0
        self.episode_start_time = None
        self.last_damage_dealt = 0  # Track damage dealt THIS step
        self.consecutive_far_steps = 0
        self.consecutive_near_steps = 0

    def reset(self, initial_state=None):
        self.prev_mantis_killed = 0
        self.episode_start_time = time.time()
        self.last_damage_dealt = 0
        self.consecutive_far_steps = 0
        self.consecutive_near_steps = 0

    def calculate_reward(self, state_dict, prev_state, done, info=None):
        """
        FIX: prev_state è il VERO stato precedente, non state_dict stesso
        """
        reward = 0.0
        curr_dist = state_dict.get("distanceToBoss", 100.0)

        # ===== 1. BOSS DAMAGE (SOLO se prev_state è DIVERSO) =====
        if prev_state is not None:
            curr_boss_hp = state_dict.get("bossHealth", 0)
            prev_boss_hp = prev_state.get("bossHealth", 0)

            boss_damage = prev_boss_hp - curr_boss_hp

            if boss_damage > 0 and boss_damage != self.last_damage_dealt:
                damage_reward = boss_damage * 10.0  # Aumentato da 5.0
                reward += damage_reward
                self.last_damage_dealt = boss_damage
                print(f"  [DAMAGE] Boss hit for {boss_damage} HP: +{damage_reward:.2f}")
            elif boss_damage == 0:
                self.last_damage_dealt = 0  # Reset

        # ===== 2. DISTANCE REWARD (Strong incentive to approach) =====
        if 3.0 <= curr_dist <= 7.0:
            reward += 1.0  # Aumentato da 0.5
            self.consecutive_near_steps += 1
            self.consecutive_far_steps = 0

            # Bonus per rimanere vicino
            if self.consecutive_near_steps > 50:
                reward += 0.5
        elif curr_dist > 10.0:
            penalty = 1.0 + (curr_dist - 10.0) * 0.2  # Scaling penalty
            reward -= penalty
            self.consecutive_far_steps += 1
            self.consecutive_near_steps = 0

            # Penalty crescente per rimanere lontano
            if self.consecutive_far_steps > 100:
                reward -= 0.5
                if self.consecutive_far_steps % 100 == 0:
                    print(f"  [WARNING] Too far for {self.consecutive_far_steps} steps")
        elif curr_dist < 2.0:
            reward -= 0.3
            self.consecutive_near_steps = 0

        # ===== 3. APPROACH BONUS =====
        if prev_state is not None:
            prev_dist = prev_state.get("distanceToBoss", 100.0)
            distance_change = prev_dist - curr_dist

            if prev_dist > 8.0 and distance_change > 0:
                reward += distance_change * 0.8  # Aumentato da 0.5
            elif prev_dist > 8.0 and distance_change < 0:
                reward -= abs(distance_change) * 0.4

        # ===== 4. HEALTH LOSS =====
        if prev_state is not None:
            curr_hp = state_dict.get("playerHealth", 0)
            prev_hp = prev_state.get("playerHealth", 0)
            health_loss = prev_hp - curr_hp

            if health_loss > 0:
                penalty = health_loss * 8.0  # Aumentato da 5.0
                reward -= penalty
                print(f"  [DAMAGE TAKEN] Lost {health_loss} HP: -{penalty:.2f}")

        # ===== 5. MANTIS LORDS =====
        mantis_killed = state_dict.get("mantisLordsKilled", 0)
        if mantis_killed > self.prev_mantis_killed:
            new_kills = mantis_killed - self.prev_mantis_killed
            bonus = new_kills * 100.0  # Aumentato da 50.0
            reward += bonus
            print(f"  [MANTIS KILLED] +{new_kills}: +{bonus:.2f}")
        self.prev_mantis_killed = mantis_killed

        # ===== 6. TIME PENALTY =====
        reward -= 0.001

        # ===== 7. TERMINAL REWARDS =====
        if done:
            if state_dict.get("isDead", False):
                reward -= 30.0  # Aumentato da 20.0
                print("  [DEATH] Player died: -30.0")
            elif state_dict.get("bossDefeated", False):
                health_bonus = state_dict.get("playerHealth", 0) * 10.0
                total = 300.0 + health_bonus  # Aumentato da 200.0
                reward += total
                print(f"  [VICTORY] Boss defeated: +{total:.2f}")

        # ===== 8. MOVEMENT BONUS (Anti-idle) =====
        vel_x = state_dict.get("playerVelocityX", 0.0)
        vel_y = state_dict.get("playerVelocityY", 0.0)
        is_moving = abs(vel_x) > 0.2 or abs(vel_y) > 0.2

        if is_moving:
            reward += 0.05
        else:
            reward -= 0.15  # Aumentato penalty per stare fermo

        # ===== 9. FACING BOSS =====
        if state_dict.get("isFacingBoss", False):
            reward += 0.15  # Aumentato da 0.1

        return reward


def preprocess_state(state_dict):
    """State size: 39 features"""
    features = []

    # Player features (10)
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

    # Terrain info (5)
    terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    # Boss features (6)
    features.append(state_dict.get("bossX", 0.0))
    features.append(state_dict.get("bossY", 0.0))
    features.append(state_dict.get("bossHealth", 0) / 1000.0)
    features.append(state_dict.get("distanceToBoss", 100.0) / 20.0)
    features.append(float(state_dict.get("isFacingBoss", False)))
    dist = state_dict.get("distanceToBoss", 100.0)
    optimal_zone = 5.0
    zone_deviation = abs(dist - optimal_zone) / 20.0
    features.append(zone_deviation)

    # Hazards (18 = 6 hazards × 3 features)
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
    update_timestep=800,
    learning_rate=5e-5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    n_epochs=4,
    save_freq=25,
    checkpoint_dir="checkpoints_ppo_stable",
    host="localhost",
    port=5555,
    plot_freq=100,
):
    """Training PPO con FIX per double-print bug."""
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
            f.write(
                "episode,total_reward,steps,global_step,mantis_killed,actor_loss,critic_loss,entropy\n"
            )

    print(f"\n{'='*60}")
    print(f"Starting PPO Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset(initial_state=state_dict)
        agent.reset_hidden()

        episode_reward = 0.0
        prev_state_dict = None  # FIX: Inizia come None

        episode_actor_loss = 0.0
        episode_critic_loss = 0.0
        episode_entropy = 0.0
        num_updates = 0

        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        for step in range(max_steps_per_episode):
            global_step += 1

            action, log_prob, val = agent.select_action(state)
            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            # FIX: Passa il VERO prev_state_dict, non state_dict
            reward = reward_calc.calculate_reward(
                next_state_dict, prev_state_dict, done, info
            )

            episode_reward += reward

            agent.store_transition(state, action, log_prob, val, reward, done)

            if global_step % update_timestep == 0:
                print(f"  [PPO Update] Updating policy at step {global_step}...")

                metrics = agent.learn()
                if metrics:
                    episode_actor_loss += metrics.get("actor_loss", 0.0)
                    episode_critic_loss += metrics.get("critic_loss", 0.0)
                    episode_entropy += metrics.get("entropy", 0.0)
                    num_updates += 1

            # FIX: Aggiorna prev_state_dict QUI, DOPO il calcolo reward
            state = next_state
            prev_state_dict = next_state_dict  # ← QUESTO è il fix

            if step % 100 == 0:
                print(
                    f"  [Step {step}] Reward: {episode_reward:.2f}, Dist: {next_state_dict.get('distanceToBoss', 0):.1f}"
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

        episode_rewards.append(episode_reward)

        mantis_killed = reward_calc.prev_mantis_killed

        avg_actor_loss = episode_actor_loss / num_updates if num_updates > 0 else 0.0
        avg_critic_loss = episode_critic_loss / num_updates if num_updates > 0 else 0.0
        avg_entropy = episode_entropy / num_updates if num_updates > 0 else 0.0

        print(f"\n[Episode {episode + 1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps in Episode: {step + 1}")
        print(f"  Global Steps: {global_step}")
        print(f"  Mantis Lords Killed: {mantis_killed}/3")
        print(f"  Avg Actor Loss: {avg_actor_loss:.4f}")
        print(f"  Avg Critic Loss: {avg_critic_loss:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.4f}")

        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{mantis_killed},{avg_actor_loss:.4f},{avg_critic_loss:.4f},{avg_entropy:.4f}\n"
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
        "update_timestep": 800,
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "n_epochs": 4,
        "save_freq": 25,
        "checkpoint_dir": "checkpoints_ppo_fixed",  # NUOVA CARTELLA
        "host": "localhost",
        "port": 5555,
        "plot_freq": 100,
    }

    print("=" * 60)
    print("PPO Training - FIXED: Double-print bug + Reward tuning")
    print("=" * 60)
    print("\nKey Fixes:")
    print("  ✓ Fixed double-print bug (prev_state assignment)")
    print("  ✓ Increased boss damage reward: 5.0 → 10.0")
    print("  ✓ Increased distance reward: 0.5 → 1.0")
    print("  ✓ Strong penalty for staying far (scaling)")
    print("  ✓ Increased idle penalty: 0.1 → 0.15")
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
