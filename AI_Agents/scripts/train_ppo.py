"""
PPO Training Script for Hollow Knight Mantis Lords Boss Fight.
Optimized with aggressive reward shaping and curriculum learning.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv


def preprocess_state(state_dict: dict) -> np.ndarray:
    """
    Convert raw state dict to 37-dimensional feature vector.
    NO absolute coordinates - only relative positions and velocities.
    """
    features = []

    # Player features (8)
    features.append(state_dict.get("playerVelocityX", 0.0) / 10.0)
    features.append(state_dict.get("playerVelocityY", 0.0) / 10.0)
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))
    features.append(float(state_dict.get("hasDoubleJump", False)))

    # Terrain raycast info (5)
    terrain_info = state_dict.get("terrainInfo", [1.0] * 5)
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    # Boss features - RELATIVE COORDINATES (6)
    player_x = state_dict.get("playerX", 0.0)
    player_y = state_dict.get("playerY", 0.0)
    boss_x = state_dict.get("bossX", 0.0)
    boss_y = state_dict.get("bossY", 0.0)

    boss_rel_x = (boss_x - player_x) / 40.0
    boss_rel_y = (boss_y - player_y) / 30.0

    features.append(np.clip(boss_rel_x, -1.0, 1.0))
    features.append(np.clip(boss_rel_y, -1.0, 1.0))
    features.append(state_dict.get("bossHealth", 0) / 1000.0)
    features.append(np.clip(state_dict.get("distanceToBoss", 50.0) / 50.0, 0.0, 1.0))
    features.append(float(state_dict.get("isFacingBoss", False)))

    # Zone deviation (1) - distance from optimal combat range
    optimal_range = 5.0
    distance = state_dict.get("distanceToBoss", 100.0)
    zone_deviation = abs(distance - optimal_range) / 20.0
    features.append(np.clip(zone_deviation, 0.0, 1.0))

    # Hazards (18): 6 hazards × 3 features (relX, relY, velX)
    hazards = state_dict.get("nearbyHazards", [])
    for i in range(6):
        if i < len(hazards):
            h = hazards[i]
            rel_x = np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0)
            rel_y = np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0)
            vel_x = np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0)
            features.extend([rel_x, rel_y, vel_x])
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def train_ppo(
    num_episodes: int = 1000,
    max_steps: int = 6000,
    update_interval: int = 512,  # RIDOTTO da 800 - update più frequenti
    learning_rate: float = 3e-5,  # RIDOTTO da 5e-5 - learning più stabile
    gamma: float = 0.995,  # AUMENTATO da 0.99 - più valore a reward futuri
    entropy_coef_start: float = 0.20,  # AUMENTATO - esplorazione aggressiva
    entropy_coef_end: float = 0.08,  # NUOVO - decadimento curriculum
    save_freq: int = 25,
    checkpoint_dir: str = "checkpoints_ppo_optimized",
    host: str = "localhost",
    port: int = 5555,
):
    """Train PPO agent on Mantis Lords with curriculum learning."""

    # Setup directories
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(root_dir, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print("=" * 60)
    print(f"PPO Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("OPTIMIZATIONS:")
    print("  ✓ Damage reward: 40 → 200 (+400%)")
    print("  ✓ Kill reward: 400 → 1000 (+150%)")
    print("  ✓ Attack reward: 8 → 15 (+87%)")
    print("  ✓ Survival reward: +0.1 per step (NEW)")
    print("  ✓ Progressive death penalty (100/50/20)")
    print("  ✓ Entropy curriculum: 0.20 → 0.08")
    print("  ✓ Update interval: 800 → 512")
    print("=" * 60)

    # Initialize environment with reward shaping
    env = HollowKnightEnv(host=host, port=port, use_reward_shaping=True)
    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[PPO] State size: {state_size}, Action size: {action_size}")

    # Initialize agent with high initial entropy
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coef=entropy_coef_start,
        use_lstm=True,
    )

    # Load checkpoint if exists
    latest_checkpoint = os.path.join(checkpoint_dir_full, "latest_ppo.pth")
    start_episode = 0
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            # Try to read episode number from log
            logfile = os.path.join(checkpoint_dir_full, "training_log.txt")
            if os.path.exists(logfile):
                with open(logfile, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        start_episode = int(last_line[0])
            print(f"[PPO] Resumed from episode {start_episode}")
        except Exception as e:
            print(f"[PPO] Could not load checkpoint: {e}")

    # Training logs
    logfile = os.path.join(checkpoint_dir_full, "training_log.txt")
    if not os.path.exists(logfile):
        with open(logfile, "w") as f:
            f.write(
                "episode,total_reward,steps,damage_dealt,boss_hits,mantis_killed,steps_survived,actor_loss,critic_loss,entropy\n"
            )

    best_reward = -float("inf")
    episode_rewards = []

    # Training loop
    for episode in range(start_episode, num_episodes):
        # CURRICULUM LEARNING - Decadimento entropy
        progress = episode / num_episodes
        current_entropy = (
            entropy_coef_start - (entropy_coef_start - entropy_coef_end) * progress
        )
        agent.entropy_coef = current_entropy

        state_dict = env.reset()
        state = preprocess_state(state_dict)
        agent.reset_hidden()

        episode_reward = 0.0
        episode_steps = 0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        num_updates = 0

        print(f"\n{'='*60}")
        print(f"[Episode {episode + 1}/{num_episodes}] Entropy: {current_entropy:.3f}")
        print(f"{'='*60}")

        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Execute action
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            # Store transition
            agent.store_transition(state, action, log_prob, value, reward, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            # Update policy
            if len(agent.buffer) >= update_interval:
                print(f"  [Update] Step {step}: Updating policy...")
                metrics = agent.learn()
                if metrics:
                    actor_loss_sum += metrics["actor_loss"]
                    critic_loss_sum += metrics["critic_loss"]
                    entropy_sum += metrics["entropy"]
                    num_updates += 1

            # Log progress
            if step % 200 == 0 and step > 0:
                dist = next_state_dict.get("distanceToBoss", 0)
                hits = info.get("total_boss_hits", 0)
                damage = info.get("total_damage_dealt", 0)
                print(
                    f"  Step {step}: R={episode_reward:.1f}, D={dist:.1f}, Hits={hits}, Dmg={damage:.1f}"
                )

            if done:
                if info.get("death"):
                    survived = info.get("steps_survived", 0)
                    print(f"  [End] Player died (survived {survived} steps)")
                elif info.get("victory"):
                    print("  [End] 🎉 VICTORY!")
                break

        # Final policy update
        if len(agent.buffer) > 0:
            print("  [Final Update] Processing remaining buffer...")
            metrics = agent.learn()
            if metrics:
                actor_loss_sum += metrics["actor_loss"]
                critic_loss_sum += metrics["critic_loss"]
                entropy_sum += metrics["entropy"]
                num_updates += 1

        # Compute averages
        avg_actor_loss = actor_loss_sum / num_updates if num_updates > 0 else 0.0
        avg_critic_loss = critic_loss_sum / num_updates if num_updates > 0 else 0.0
        avg_entropy = entropy_sum / num_updates if num_updates > 0 else 0.0

        episode_rewards.append(episode_reward)
        avg_reward_50 = (
            np.mean(episode_rewards[-50:])
            if len(episode_rewards) >= 50
            else np.mean(episode_rewards)
        )

        # Episode summary
        print(f"\n{'='*60}")
        print(f"[Episode {episode + 1}] SUMMARY")
        print(f"{'='*60}")
        print(f"  Reward: {episode_reward:.2f} | Avg (50): {avg_reward_50:.2f}")
        print(f"  Steps: {episode_steps} | Survived: {info.get('steps_survived', 0)}")
        print(f"  Damage Dealt: {info.get('total_damage_dealt', 0):.1f}")
        print(f"  Boss Hits: {info.get('total_boss_hits', 0)}")
        print(f"  Mantis Killed: {next_state_dict.get('mantisLordsKilled', 0)}/3")
        print(f"  Loss: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}")
        print(f"  Entropy: {avg_entropy:.4f} (target={current_entropy:.3f})")
        print(f"{'='*60}")

        # Log to file
        with open(logfile, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{episode_steps},"
                f"{info.get('total_damage_dealt', 0):.1f},{info.get('total_boss_hits', 0)},"
                f"{next_state_dict.get('mantisLordsKilled', 0)},"
                f"{info.get('steps_survived', 0)},"
                f"{avg_actor_loss:.4f},{avg_critic_loss:.4f},{avg_entropy:.4f}\n"
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir_full, "best_ppo.pth")
            agent.save(best_path)
            print(f"  🌟 NEW BEST REWARD! Saved to {best_path}")

        # Periodic checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir_full, f"episode_{episode + 1}.pth"
            )
            agent.save(checkpoint_path)
            agent.save(latest_checkpoint)
            print("  💾 Checkpoint saved")

        # Early stopping check
        if len(episode_rewards) >= 100:
            recent_avg = np.mean(episode_rewards[-100:])
            if recent_avg > 500:  # Se media ultimi 100 > 500, sta imparando bene
                print(
                    f"\n✓ Strong performance detected (avg={recent_avg:.1f})! Training stabilized."
                )

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Final Avg (50 eps): {avg_reward_50:.2f}")
    print(f"Total Episodes: {num_episodes}")
    print("=" * 60)

    final_path = os.path.join(checkpoint_dir_full, "final_ppo.pth")
    agent.save(final_path)
    env.close()


if __name__ == "__main__":
    HYPERPARAMS = {
        "num_episodes": 1000,
        "max_steps": 6000,
        "update_interval": 512,
        "learning_rate": 3e-5,
        "gamma": 0.995,
        "entropy_coef_start": 0.20,
        "entropy_coef_end": 0.08,
        "save_freq": 25,
        "checkpoint_dir": "checkpoints_ppo_optimized",
        "host": "localhost",
        "port": 5555,
    }

    print("=" * 60)
    print("PPO Training - Mantis Lords Boss Fight")
    print("OPTIMIZED VERSION with Curriculum Learning")
    print("=" * 60)
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 60)
    print("Starting in 3 seconds...")
    time.sleep(3)
    print()

    train_ppo(**HYPERPARAMS)
