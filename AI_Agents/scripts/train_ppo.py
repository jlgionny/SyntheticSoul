"""
PPO Training Script - MINIMAL STATE (18 features).
Debugging wall-banging behavior.

Uso:
    python train_ppo.py                     # Default settings
    python train_ppo.py --port 5556         # Custom port
    python train_ppo.py --instance 1        # Separate checkpoints
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv


def preprocess_state(state_dict: dict) -> np.ndarray:
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

    # Hazards (5) - Il più vicino
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

    return np.array(features, dtype=np.float32)


def train_ppo(
    num_episodes: int = 1000,
    max_steps: int = 6000,
    update_interval: int = 1024,  # Ridotto per update più frequenti
    learning_rate: float = 1e-4,  # Ridotto per stabilità
    gamma: float = 0.995,  # Aumentato per reward sparse
    entropy_coef_start: float = 0.30,  # Ridotto - meno random
    entropy_coef_end: float = 0.05,
    save_freq: int = 25,
    checkpoint_dir: str = "checkpoints_ppo_v2",
    host: str = "localhost",
    port: int = 5555,
):
    """Train PPO agent with enhanced state and combat-focused rewards."""

    # Setup directories
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(root_dir, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print("=" * 60)
    print(f"PPO Training V2 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("CONFIGURATION:")
    print("  ✓ State: ENHANCED (26 features)")
    print("  ✓ Actions: 9 (base actions)")
    print(f"  ✓ Learning rate: {learning_rate}")
    print(f"  ✓ Gamma: {gamma}")
    print(f"  ✓ Entropy: {entropy_coef_start} → {entropy_coef_end}")
    print(f"  ✓ Update interval: {update_interval}")
    print("  ✓ Combat-focused reward shaping")
    print("=" * 60)

    # Initialize environment with combat-focused reward shaping
    env = HollowKnightEnv(host=host, port=port, use_reward_shaping=True)
    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 9  # Azioni base

    print(f"[PPO] State size: {state_size}, Action size: {action_size}")

    # Initialize agent with LSTM for temporal patterns
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=0.97,  # Aumentato per better credit assignment
        entropy_coef=entropy_coef_start,
        use_lstm=True,  # Abilitato per pattern recognition
        n_epochs=6,  # Aumentato per better learning
        batch_size=128,  # Aumentato
    )

    # Training logs
    logfile = os.path.join(checkpoint_dir_full, "training_log.txt")
    if not os.path.exists(logfile):
        with open(logfile, "w") as f:
            f.write(
                "episode,total_reward,steps,boss_hp,player_hp,mantis_killed,wall_violations,attacks,hits,hit_rate,actor_loss,critic_loss,entropy\n"
            )

    best_reward = -float("inf")
    episode_rewards = []

    # Training loop
    for episode in range(num_episodes):
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
        wall_violations = 0

        print(f"\n{'='*60}")
        print(f"[Episode {episode + 1}/{num_episodes}] Entropy: {current_entropy:.3f}")

        # Exploration rate decrescente
        if episode < 50:
            explore_rate = 0.3  # 30% random per primi 50 episodi
            print(f"  [EXPLORATION] {explore_rate*100:.0f}% random (warmup)")
        elif episode < 150:
            explore_rate = 0.15  # 15% per episodi 50-150
            print(f"  [EXPLORATION] {explore_rate*100:.0f}% random")
        else:
            explore_rate = 0.05  # 5% per resto del training
        print(f"{'='*60}")

        # Track combat stats per episode
        attacks_attempted = 0
        boss_hits = 0

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

            # Execute action
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            # Track combat stats (ora info non contiene più attack_attempted)
            # Lo determiniamo dall'azione eseguita
            if action in [5, 7]:  # ATTACK, SPELL
                attacks_attempted += 1

            # Track wall violations
            terrain = next_state_dict.get("terrainInfo", [1.0] * 5)
            wall_ahead = terrain[2] if len(terrain) > 2 else 1.0
            distance_to_boss = next_state_dict.get("distanceToBoss", 100.0)
            if wall_ahead < 0.2 and distance_to_boss > 15.0:
                wall_violations += 1

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

            # DEBUG LOG - Detailed state info
            if step % 200 == 0 and step > 0:
                boss_x = next_state_dict.get("bossRelativeX", 0)
                boss_y = next_state_dict.get("bossRelativeY", 0)
                dist = next_state_dict.get("distanceToBoss", 0)
                boss_hp = info.get("boss_damage", 0)
                player_hp = info.get("player_hp", 0)
                action_name = env.ACTIONS.get(action, "UNKNOWN")

                print(f"\n  Step {step}: R={episode_reward:.1f}")
                print(
                    f"    Boss: X={boss_x:.2f}, Y={boss_y:.2f}, Dist={dist:.1f}, HP={boss_hp:.0f}"
                )
                print(f"    Player: HP={player_hp}, Action={action_name}")
                print(f"    Terrain: Wall={wall_ahead:.2f} (0=close, 1=far)")
                print(f"    Wall violations: {wall_violations}")

            if done:
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
        hit_rate = (boss_hits / attacks_attempted * 100) if attacks_attempted > 0 else 0

        print(f"\n{'='*60}")
        print(f"[Episode {episode + 1}] SUMMARY")
        print(f"{'='*60}")
        print(f"  Reward: {episode_reward:.2f} | Avg (50): {avg_reward_50:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  Boss HP: {next_state_dict.get('bossHealth', 100):.0f}")
        print(f"  Player HP: {next_state_dict.get('playerHealth', 0)}")
        print(f"  Mantis Killed: {next_state_dict.get('mantisLordsKilled', 0)}/3")
        print(
            f"  Combat: {attacks_attempted} attacks, {boss_hits} hits ({hit_rate:.1f}%)"
        )
        print(f"  Wall Violations: {wall_violations}")
        print(f"  Loss: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}")
        print(f"  Entropy: {avg_entropy:.4f} (target={current_entropy:.3f})")
        print(f"{'='*60}")

        # Log to file
        with open(logfile, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{episode_steps},"
                f"{next_state_dict.get('bossHealth', 100):.0f},{next_state_dict.get('playerHealth', 0)},"
                f"{next_state_dict.get('mantisLordsKilled', 0)},{wall_violations},"
                f"{attacks_attempted},{boss_hits},{hit_rate:.1f},"
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
            print("  💾 Checkpoint saved")

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
    # ============ ARGOMENTI CLI PER MULTI-ISTANZA ============
    parser = argparse.ArgumentParser(description="Train PPO Agent for Hollow Knight")
    parser.add_argument(
        "--port", type=int, default=5555, help="Port to connect to (default: 5555)"
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="Instance ID for checkpoint naming (default: 0)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to train (default: 1000)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--entropy-start",
        type=float,
        default=0.30,
        help="Starting entropy coefficient (default: 0.30)",
    )
    parser.add_argument(
        "--entropy-end",
        type=float,
        default=0.05,
        help="Ending entropy coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help="Discount factor (default: 0.995)",
    )
    args = parser.parse_args()

    # Modifica il nome della cartella checkpoint per istanza
    checkpoint_dir = "checkpoints_ppo_v2"
    if args.instance > 0:
        checkpoint_dir = f"checkpoints_ppo_minimal_instance{args.instance}"

    HYPERPARAMS = {
        "num_episodes": args.episodes,
        "max_steps": 6000,
        "update_interval": 1024,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "entropy_coef_start": args.entropy_start,
        "entropy_coef_end": args.entropy_end,
        "save_freq": 25,
        "checkpoint_dir": checkpoint_dir,
        "host": "localhost",
        "port": args.port,
    }

    print("=" * 60)
    print(f"PPO Training - Instance {args.instance}")
    print(f"Port: {args.port}")
    print("MINIMAL STATE + MAX EXPLORATION")
    print("=" * 60)
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 60)
    print("Starting in 3 seconds...")
    time.sleep(3)
    print()

    try:
        train_ppo(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user.")
    except Exception as e:
        print(f"\n[Train] Critical Error: {e}")
        import traceback

        traceback.print_exc()
