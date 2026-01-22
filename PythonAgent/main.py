"""Main training loop for DQN Hollow Knight agent"""
import time
import os
from datetime import datetime
from game_communicator import GameCommunicator
from state_processor import StateProcessor
from dqn_agent import DQNAgent
from config import *

# FIX: Reconnection configuration
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 5
MAX_CONSECUTIVE_FAILURES = 5

def attempt_reconnection(communicator, attempt_number):
    """FIX: Attempt to reconnect to the game"""
    print(f"\n{'='*60}")
    print(f"🔄 RECONNECTION ATTEMPT {attempt_number}/{MAX_RECONNECT_ATTEMPTS}")
    print(f"{'='*60}")
    print(f"Waiting {RECONNECT_DELAY}s before reconnecting...")
    time.sleep(RECONNECT_DELAY)
    
    success = communicator.connect(max_retries=5, retry_delay=3)
    
    if success:
        print("✓ Reconnection successful! Resuming training...")
        return True
    else:
        print("✗ Reconnection failed")
        return False

def main():
    print("=== Hollow Knight DQN Training ===")
    print(f"Starting at: {datetime.now()}")

    # Initialize components
    communicator = GameCommunicator()
    processor = StateProcessor()
    agent = DQNAgent()

    # Try to load existing model
    agent.load()

    # Connect to game
    if not communicator.connect():
        print("Failed to connect to game. Make sure Hollow Knight is running with the mod!")
        return

    # Training statistics
    episode_rewards = []
    episode_losses = []
    
    # FIX: Failure tracking
    consecutive_failures = 0

    print("\n=== Starting Training ===")
    print(f"Episodes: {MAX_EPISODES}")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Epsilon: {agent.epsilon:.3f}\n")

    try:
        for episode in range(MAX_EPISODES):
            episode_start_time = time.time()

            # FIX: Enhanced connection check with reconnection logic
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{MAX_EPISODES}")
            print(f"{'='*60}")
            print("Waiting for game state...")
            
            raw_state = communicator.receive_state()
            
            # FIX: If no state, attempt reconnection instead of skipping
            if raw_state is None:
                consecutive_failures += 1
                print(f"⚠ No state received (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\n" + "!"*60)
                    print("CRITICAL: Too many consecutive failures!")
                    print("!"*60)
                    
                    # Attempt reconnection
                    reconnection_success = False
                    for reconnect_attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                        if attempt_reconnection(communicator, reconnect_attempt):
                            reconnection_success = True
                            consecutive_failures = 0
                            break
                    
                    if not reconnection_success:
                        print("\n" + "="*60)
                        print("ABORTING TRAINING - Cannot reconnect to game")
                        print("="*60)
                        print("Saving model before exit...")
                        agent.save()
                        print("Model saved. Check game/mod status and restart training.")
                        return
                    
                    # After successful reconnect, retry getting state
                    print("Retrying state fetch after reconnect...")
                    raw_state = communicator.receive_state()
                    if raw_state is None:
                        print("Still no state after reconnect. Skipping episode.")
                        continue
                else:
                    # FIX: Wait before retrying instead of instant spam
                    print(f"Waiting 3s before retrying episode {episode + 1}...")
                    time.sleep(3)
                    continue
            
            # FIX: Reset failure counter on successful state receive
            consecutive_failures = 0
            
            state = processor.process(raw_state)
            prev_raw_state = raw_state

            episode_reward = 0
            episode_loss = 0
            steps = 0
            action_counts = {action: 0 for action in ACTIONS}

            # Episode loop
            for step in range(MAX_STEPS_PER_EPISODE):
                # Choose action
                action_idx = agent.act(state, training=True)
                action = ACTIONS[action_idx]
                action_counts[action] += 1

                # Print action distribution every 50 steps
                if step % 50 == 0 and step > 0:
                    print(f"  Step {step} - Action distribution: {action_counts}")

                # Send action to game
                if not communicator.send_action(action):
                    print("✗ Lost connection to game during action send")
                    break

                # Receive next state
                next_raw_state = communicator.receive_state()
                if next_raw_state is None:
                    print("✗ Lost connection to game during state receive")
                    consecutive_failures += 1
                    break

                next_state = processor.process(next_raw_state)

                # Calculate reward
                reward = processor.calculate_reward(prev_raw_state, next_raw_state)
                episode_reward += reward

                # Check if terminal
                done = processor.is_terminal(next_raw_state)

                # Store experience
                agent.remember(state, action_idx, reward, next_state, done)

                # Train
                loss = agent.replay()
                episode_loss += loss

                # Update state
                state = next_state
                prev_raw_state = next_raw_state
                steps += 1

                # Print step info occasionally
                if step % 100 == 0:
                    print(f"  Step {step}: Action={action}, Reward={reward:.1f}, "
                          f"Boss HP={next_raw_state['bossHealth']}, "
                          f"Player HP={next_raw_state['playerHealth']}, "
                          f"Epsilon={agent.epsilon:.3f}")

                if done:
                    break

            # FIX: Only process episode results if we had actual steps
            if steps == 0:
                print(f"⚠ Episode {episode + 1} had 0 steps. Skipping statistics.")
                continue

            # Episode finished
            episode_duration = time.time() - episode_start_time
            avg_loss = episode_loss / max(steps, 1)
            episode_rewards.append(episode_reward)
            episode_losses.append(avg_loss)

            # Update target network periodically
            if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print("  >>> Target network updated")

            # Decay epsilon
            agent.decay_epsilon()

            # Print episode summary
            print(f"\n--- Episode {episode + 1} Summary ---")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Duration: {episode_duration:.1f}s")
            print(f"  Action distribution: {action_counts}")
            
            if len(episode_rewards) >= 10:
                avg_reward_10 = sum(episode_rewards[-10:]) / 10
                print(f"  Avg Reward (last 10): {avg_reward_10:.2f}")

            # Save model periodically
            if (episode + 1) % SAVE_FREQUENCY == 0:
                agent.save()
                print(f"  >>> Model saved!")

            # Small delay between episodes
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user!")
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        # Training complete
        print("\n=== Training Complete ===")
        agent.save()
        communicator.close()
        print(f"Final epsilon: {agent.epsilon:.3f}")
        print(f"Total episodes: {len(episode_rewards)}")
        if episode_rewards:
            print(f"Best reward: {max(episode_rewards):.2f}")
            print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")

if __name__ == "__main__":
    main()
