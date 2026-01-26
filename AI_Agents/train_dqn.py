import os
import sys
import time
import json
import math
import numpy as np
import torch
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.dqn_agent import DQNAgent
from src.env.hollow_knight_env import HollowKnightEnv


class RewardCalculator:
    """
    Simplified Reward Function - Combat-Focused

    Design Philosophy:
    - Boss engagement is the PRIMARY objective
    - Simple, interpretable reward components
    - Sparse rewards for outcomes, Dense for positioning
    - No conflicting incentives
    """

    def __init__(self):
        self.prev_boss_health = None
        self.episode_start_time = None
        self.steps_in_combat_range = 0
        self.current_step = 0

    def reset(self):
        """Reset for new episode."""
        self.prev_boss_health = None
        self.episode_start_time = time.time()
        self.steps_in_combat_range = 0
        self.current_step = 0

    def calculate_reward(self, state_dict, prev_state, done, info=None):
        """
        Simplified reward focusing on boss combat.

        Components (in priority order):
        1. Damage to boss: +20 per HP
        2. Damage taken: -15 per HP
        3. Combat range: +0.1 per step in range [2, 6]
        4. Attack bonus: +0.2 if attacking in optimal range
        5. Wall stuck penalty: Only if velocity = 0 AND against wall
        """
        reward = 0.0
        self.current_step += 1

        # ============ PRIORITY 1: DAMAGE TO BOSS ============
        if prev_state is not None and 'bossHealth' in state_dict and 'bossHealth' in prev_state:
            boss_damage = prev_state['bossHealth'] - state_dict['bossHealth']
            if boss_damage > 0:
                reward += 20.0 
                print(f"  [Reward] Boss hit: +20.0 ({boss_damage} HP damage)")

        # ============ PRIORITY 2: DAMAGE TAKEN ============
        damage_taken = state_dict.get('damageTaken', 0)
        if damage_taken > 0:
            damage_penalty = damage_taken * 15.0
            reward -= damage_penalty
            print(f"  [Reward] Damage taken: -{damage_penalty:.1f}")

        # ============ PRIORITY 3: COMBAT RANGE BONUS ============
        curr_dist = state_dict.get('distanceToBoss', 100.0)

        # Dense reward for staying in optimal attack range
        if 2.0 <= curr_dist <= 6.0:
            reward += 0.1
            self.steps_in_combat_range += 1
        else:
            self.steps_in_combat_range = 0

        # ============ PRIORITY 4: ATTACK INCENTIVE ============
        if info is not None and 'action_name' in info:
            action_name = info['action_name']
            # Bonus for attacking when in optimal range
            if action_name == 'ATTACK' and 2.0 <= curr_dist <= 6.0:
                reward += 0.2

        # ============ PRIORITY 5: WALL STUCK PENALTY ============
        # Only penalize if truly stuck (velocity = 0 AND against wall)
        terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
        player_vel_x = state_dict.get('playerVelocityX', 0.0)
        player_vel_y = state_dict.get('playerVelocityY', 0.0)

        if len(terrain_info) >= 3:
            wall_distance = terrain_info[2]  # Distance to nearest wall

            # Check if stuck: no velocity AND very close to wall
            is_stuck = (abs(player_vel_x) < 0.1 and abs(player_vel_y) < 0.1 and 
                       wall_distance < 0.1)

            if is_stuck:
                reward -= 0.5

        # ============ PRIORITY 6: ENCOURAGE CENTER MOVEMENT IF FAR ============
        # If far from boss, incentivize moving toward center (away from walls)
        if curr_dist > 8.0 and len(terrain_info) >= 3:
            wall_distance = terrain_info[2]
            # Reward being in open space when far from boss
            if wall_distance > 0.4:
                reward += 0.05

        # ============ TERMINAL STATE REWARDS ============
        if done:
            if state_dict.get('isDead', False):
                reward -= 50.0
                print(f"  [Reward] Episode end (death): -50.0")
            elif state_dict.get('bossDefeated', False):
                reward += 500.0
                # Time bonus for fast victories
                elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
                time_bonus = max(0, 100 - elapsed / 10)
                reward += time_bonus
                print(f"  [Reward] BOSS DEFEATED: +{500 + time_bonus:.1f}")

        return reward


def preprocess_state(state_dict):
    """
    Optimized state preprocessing with better normalization.

    Design changes:
    1. All distances normalized to [0, 1]
    2. Boss relative position is the dominant signal
    3. Only 2 nearest hazards included
    4. Hazard velocity included for dodging learning

    State size: 28 features
    - Player: 10 features
    - Terrain: 5 features  
    - Boss: 7 features
    - Hazards: 6 features (2 hazards × 3 features each)
    """
    features = []

    # ========== PLAYER FEATURES (10 features) ==========
    # Normalize positions to typical arena size (assume ~40x30 units)
    player_x = state_dict.get('playerX', 0.0)
    player_y = state_dict.get('playerY', 0.0)

    features.append(player_x / 40.0)  # Normalized X position
    features.append(player_y / 30.0)  # Normalized Y position

    # Velocity (normalized to typical max velocity ~10 units/s)
    features.append(np.clip(state_dict.get('playerVelocityX', 0.0) / 10.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get('playerVelocityY', 0.0) / 10.0, -1.0, 1.0))

    # Health and soul (already 0-1 range)
    features.append(state_dict.get('playerHealth', 0) / 10.0)
    features.append(state_dict.get('playerSoul', 0) / 100.0)

    # Boolean capabilities
    features.append(float(state_dict.get('canDash', False)))
    features.append(float(state_dict.get('canAttack', False)))
    features.append(float(state_dict.get('isGrounded', False)))
    features.append(float(state_dict.get('hasDoubleJump', False)))

    # ========== TERRAIN INFO (5 features) ==========
    # Already normalized by C# mod (distances to ground/ceiling/walls)
    terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])

    # ========== BOSS FEATURES (7 features) ==========
    boss_x = state_dict.get('bossX', 0.0)
    boss_y = state_dict.get('bossY', 0.0)

    # Boss relative position (DOMINANT SIGNALS)
    boss_relative_x = (boss_x - player_x) / 40.0  # Normalized
    boss_relative_y = (boss_y - player_y) / 30.0  # Normalized

    features.append(boss_relative_x)
    features.append(boss_relative_y)

    # Boss health (normalized)
    features.append(state_dict.get('bossHealth', 0) / 1000.0)

    # Distance to boss (normalized to arena diagonal ~50 units)
    distance_to_boss = state_dict.get('distanceToBoss', 50.0)
    features.append(np.clip(distance_to_boss / 50.0, 0.0, 1.0))

    # Angle-related features
    angle_to_boss = math.atan2(boss_relative_y, boss_relative_x) / math.pi  # [-1, 1]
    features.append(angle_to_boss)

    # Is facing boss
    features.append(float(state_dict.get('isFacingBoss', False)))

    # Boss velocity (if available, for prediction)
    boss_vel_x = state_dict.get('bossVelocityX', 0.0)
    features.append(np.clip(boss_vel_x / 10.0, -1.0, 1.0))

    # Save key values back to state_dict for reward calculation
    state_dict['bossRelativeX'] = boss_relative_x
    state_dict['bossRelativeY'] = boss_relative_y

    # ========== HAZARDS: TOP 2 NEAREST (6 features) ==========
    # Each hazard: relative position (2) + relative velocity (1) = 3 features
    hazards = state_dict.get('nearbyHazards', [])

    # Sort by distance and take closest 2
    if hazards:
        # Calculate distance for each hazard
        hazards_with_dist = []
        for h in hazards:
            rel_x = h.get('relX', 0.0)
            rel_y = h.get('relY', 0.0)
            dist = math.sqrt(rel_x**2 + rel_y**2)
            hazards_with_dist.append((dist, h))

        # Sort by distance
        hazards_with_dist.sort(key=lambda x: x[0])
        sorted_hazards = [h for _, h in hazards_with_dist[:2]]
    else:
        sorted_hazards = []

    # Process top 2 hazards
    for i in range(2):
        if i < len(sorted_hazards):
            h = sorted_hazards[i]

            # Relative position (normalized to ~30 unit detection range)
            rel_x = h.get('relX', 0.0) / 30.0
            rel_y = h.get('relY', 0.0) / 30.0
            features.append(np.clip(rel_x, -1.0, 1.0))
            features.append(np.clip(rel_y, -1.0, 1.0))

            # Relative velocity (for dodging learning)
            # Calculated as: hazard_velocity - player_velocity
            hazard_vel_x = h.get('velocityX', 0.0)
            hazard_vel_y = h.get('velocityY', 0.0)
            player_vel_x = state_dict.get('playerVelocityX', 0.0)
            player_vel_y = state_dict.get('playerVelocityY', 0.0)

            rel_vel_magnitude = math.sqrt(
                (hazard_vel_x - player_vel_x)**2 + 
                (hazard_vel_y - player_vel_y)**2
            )
            features.append(np.clip(rel_vel_magnitude / 20.0, 0.0, 1.0))
        else:
            # No hazard at this slot
            features.extend([0.0, 0.0, 0.0])

    # TOTAL: 10 + 5 + 7 + 6 = 28 features
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
    checkpoint_dir='checkpoints_dqn',
    host='localhost',
    port=5555
):
    """Main training loop with optimized reward and state processing."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[Train] Connecting to Hollow Knight at {host}:{port}...")
    env = HollowKnightEnv(host=host, port=port)

    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Train] State size: {state_size}, Action size: {action_size}")
    print(f"[Train] Using SIMPLIFIED reward system (combat-focused)")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000
    )

    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print(f"[Train] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train] Could not load checkpoint: {e}")

    reward_calc = RewardCalculator()
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    log_file = os.path.join(checkpoint_dir, 'training_log.txt')

    print(f"\n{'='*60}")
    print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset()

        episode_reward = 0.0
        episode_loss = []
        prev_state_dict = None
        total_damage_taken = 0

        print(f"\n[Episode {episode+1}/{num_episodes}] Starting...")

        for step in range(max_steps_per_episode):
            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            reward = reward_calc.calculate_reward(
                next_state_dict, 
                prev_state_dict, 
                done, 
                info
            )

            episode_reward += reward

            # Track total damage taken this episode
            damage_this_step = next_state_dict.get('damageTaken', 0)
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
                curr_dist = next_state_dict.get('distanceToBoss', 0)
                in_range = "✓" if 2.0 <= curr_dist <= 6.0 else "✗"
                print(f"  [Step {step}] Reward: {episode_reward:.2f}, "
                      f"Dist: {curr_dist:.1f} {in_range}, "
                      f"Dmg Taken: {total_damage_taken}")

            if done:
                reason = ('Player died' if state_dict.get('isDead') else 
                         'Boss defeated' if state_dict.get('bossDefeated') else 
                         'Unknown')
                print(f"  [Episode End] Reason: {reason}")
                break

        agent.episodes_done += 1
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)

        avg_reward_last_10 = (np.mean(episode_rewards[-10:]) 
                             if len(episode_rewards) >= 10 else episode_reward)

        print(f"\n[Episode {episode+1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Steps: {step+1}")
        print(f"  Epsilon: {epsilon:.4f}")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        print(f"  Combat Range Steps: {reward_calc.steps_in_combat_range}")
        print(f"  Total Damage Taken: {total_damage_taken}")

        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{episode_reward:.2f},{avg_loss:.4f},"
                   f"{step+1},{epsilon:.4f},{total_damage_taken}\n")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            agent.save(best_path)
            print(f"  [NEW BEST] Saved to {best_path}")

        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'episode_{episode+1}.pth')
            agent.save(checkpoint_path)
            agent.save(latest_checkpoint)
            print(f"  [Checkpoint] Saved to {checkpoint_path}")

    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best Reward: {best_reward:.2f}")

    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    agent.save(final_path)
    env.close()


if __name__ == "__main__":
    HYPERPARAMS = {
        'num_episodes': 1000,
        'max_steps_per_episode': 5000,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 10000,
        'target_update_freq': 1000,
        'save_freq': 50,
        'checkpoint_dir': 'checkpoints',
        'host': 'localhost',
        'port': 5555
    }

    print("="*60)
    print("HOLLOW KNIGHT DQN TRAINING - OPTIMIZED VERSION")
    print("="*60)
    print("\nKey Optimizations:")
    print("  ✓ Simplified reward (5 components instead of 10)")
    print("  ✓ Boss engagement is PRIMARY objective")
    print("  ✓ Dense reward for combat range [2, 6]")
    print("  ✓ Wall penalty only when stuck (velocity = 0)")
    print("  ✓ Normalized state features [0, 1]")
    print("  ✓ Only 2 nearest hazards with velocity")
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("="*60)
    print()

    try:
        train_dqn(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train] Error during training: {e}")
        import traceback
        traceback.print_exc()