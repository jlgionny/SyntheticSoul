import os
import sys
import time
import json
import numpy as np
import torch
from datetime import datetime

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importa l'agente PPO invece del DQN
from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv

class RewardCalculator:
    """
    Reward Function per Hollow Knight AI (Identica alla versione DQN).
    """
    
    def __init__(self):
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        self.episode_start_time = None
    
    def reset(self):
        """Reset per nuovo episodio."""
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        self.episode_start_time = time.time()
    
    def calculate_reward(self, state_dict, prev_state, done, info=None):
        reward = 0.0
        
        # ============ REWARD PER DANNO AL BOSS ============
        if prev_state is not None and 'bossHealth' in state_dict and 'bossHealth' in prev_state:
            boss_damage = prev_state['bossHealth'] - state_dict['bossHealth']
            if boss_damage > 0:
                reward += boss_damage * 2.0
                # print(f"  [Reward] Boss damage: +{boss_damage * 2.0:.2f}")
        
        # ============ REWARD PER AVVICINAMENTO AL BOSS ============
        if prev_state is not None and 'distanceToBoss' in state_dict and 'distanceToBoss' in prev_state:
            prev_dist = prev_state['distanceToBoss']
            curr_dist = state_dict['distanceToBoss']
            
            distance_change = prev_dist - curr_dist
            if distance_change > 0:
                reward += distance_change * 0.05
            
            if curr_dist > 15.0:
                reward -= 0.1
        
        # ============ PENALITÀ PER PERDITA VITA ============
        if prev_state is not None and 'playerHealth' in state_dict and 'playerHealth' in prev_state:
            health_loss = prev_state['playerHealth'] - state_dict['playerHealth']
            if health_loss > 0:
                reward -= health_loss * 20.0
                print(f"  [Reward] Health lost: -{health_loss * 20.0:.2f}")
        
        # ============ PENALITÀ TEMPORALE ============
        reward -= 0.01
        
        # ============ PENALITÀ PER COLLISIONE CON MURO ============
        if info is not None and 'action_name' in info:
            action_name = info['action_name']
            terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
            
            if len(terrain_info) >= 3:
                wall_distance = terrain_info[2]
                if wall_distance < 0.15:  # Muro molto vicino
                    if action_name in ['MOVE_LEFT', 'MOVE_RIGHT']:
                        reward -= 20.0
        
        # ============ REWARD/PENALITÀ PER MORTE O VITTORIA ============
        if done:
            if state_dict.get('isDead', False):
                reward -= 100.0
                print(f"  [Reward] Player died: -100.0")
            elif state_dict.get('bossDefeated', False):
                reward += 500.0
                print(f"  [Reward] Boss defeated: +500.0")
        
        # ============ BONUS PER SCHIVATA ============
        hazards = state_dict.get('nearbyHazards', [])
        if hazards:
            distances = []
            for h in hazards:
                rx = h.get('relX', 100.0)
                ry = h.get('relY', 100.0)
                dist = (rx**2 + ry**2)**0.5
                distances.append(dist)
            
            if distances:
                closest_dist = min(distances)
                if closest_dist < 3.0:
                    if prev_state is not None and state_dict['playerHealth'] == prev_state.get('playerHealth', 0):
                        reward += 0.5
        
        # ============ PENALITÀ PER CADUTA ============
        terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
        if len(terrain_info) >= 1:
            floor_distance = terrain_info[0]
            if not state_dict.get('isGrounded', True) and floor_distance > 0.6:
                reward -= 5.0
        
        return reward

def preprocess_state(state_dict):
    """
    Converte lo stato del gioco in un array numpy per la rete.
    Identico alla versione DQN per compatibilità.
    """
    features = []
    
    # ========== PLAYER FEATURES (10 features) ==========
    features.append(state_dict.get('playerX', 0.0))
    features.append(state_dict.get('playerY', 0.0))
    features.append(state_dict.get('playerVelocityX', 0.0))
    features.append(state_dict.get('playerVelocityY', 0.0))
    features.append(state_dict.get('playerHealth', 0) / 10.0)
    features.append(state_dict.get('playerSoul', 0) / 100.0)
    features.append(float(state_dict.get('canDash', False)))
    features.append(float(state_dict.get('canAttack', False)))
    features.append(float(state_dict.get('isGrounded', False)))
    features.append(float(state_dict.get('hasDoubleJump', False)))
    
    # ========== TERRAIN INFO (5 features) ==========
    terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])
    
    # ========== BOSS FEATURES (4 features) ==========
    features.append(state_dict.get('bossX', 0.0))
    features.append(state_dict.get('bossY', 0.0))
    features.append(state_dict.get('bossHealth', 0) / 1000.0)
    features.append(state_dict.get('distanceToBoss', 100.0) / 20.0)
    
    # ========== NEARBY HAZARDS (6 features) ==========
    hazards = state_dict.get('nearbyHazards', [])
    for i in range(3):
        if i < len(hazards):
            h = hazards[i]
            features.append(h.get('relX', 0.0) / 15.0)
            features.append(h.get('relY', 0.0) / 15.0)
        else:
            features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)

def train_ppo(
    num_episodes=1000,
    max_steps_per_episode=5000,
    update_timestep=2048, # PPO Specific: Ogni quanto fare l'update
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    n_epochs=10,
    save_freq=50,
    checkpoint_dir='checkpoints_ppo', # Directory diversa per PPO
    host='localhost',
    port=5555
):
    """
    Loop principale di training per PPO.
    """
    
    # Crea directory per checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Inizializza environment
    print(f"[Train PPO] Connecting to Hollow Knight at {host}:{port}...")
    env = HollowKnightEnv(host=host, port=port)
    
    # Determina state size dal primo reset
    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8 
    
    print(f"[Train PPO] State size: {state_size}, Action size: {action_size}")
    
    # Inizializza PPO agent
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=clip_param,
        n_epochs=n_epochs,
        device=None # Auto-detect
    )
    
    # Carica checkpoint se esiste
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_ppo.pth')
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print(f"[Train PPO] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train PPO] Could not load checkpoint: {e}")
    
    # Reward calculator
    reward_calc = RewardCalculator()
    
    # Training stats
    episode_rewards = []
    best_reward = -float('inf')
    
    # Global step counter per triggerare l'update PPO
    global_step = 0
    
    # Log file
    log_file = os.path.join(checkpoint_dir, 'training_log_ppo.txt')
    
    print(f"\n{'='*60}")
    print(f"Starting PPO Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # ============ TRAINING LOOP ============
    for episode in range(num_episodes):
        # Reset environment
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset()
        
        episode_reward = 0.0
        prev_state_dict = None
        
        print(f"\n[Episode {episode+1}/{num_episodes}] Starting...")
        
        for step in range(max_steps_per_episode):
            global_step += 1
            
            # 1. Seleziona azione (PPO restituisce anche probabilità e valore)
            action, log_prob, val = agent.select_action(state)
            
            # 2. Esegui azione
            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)
            
            # 3. Calcola reward
            reward = reward_calc.calculate_reward(
                next_state_dict,
                prev_state_dict,
                done,
                info
            )
            
            episode_reward += reward
            
            # 4. Salva transizione nella memoria PPO (Rollout Buffer)
            # PPO ha bisogno di: state, action, log_prob, value, reward, done
            agent.store_transition(state, action, log_prob, val, reward, done)
            
            # 5. PPO Update: Si aggiorna ogni 'update_timestep' passi totali
            if global_step % update_timestep == 0:
                print(f"  [PPO Update] Updating policy at step {global_step}...")
                agent.learn()
            
            # Aggiorna stato
            state = next_state
            prev_state_dict = state_dict
            state_dict = next_state_dict
            
            # Log periodico
            if step % 100 == 0:
                print(f"  [Step {step}] Reward: {episode_reward:.2f}, Global Step: {global_step}")
            
            if done:
                reason = 'Player died' if state_dict.get('isDead') else 'Boss defeated' if state_dict.get('bossDefeated') else 'Unknown'
                print(f"  [Episode End] Reason: {reason}")
                break
        
        # ============ FINE EPISODIO ============
        episode_rewards.append(episode_reward)
        
        # Statistiche
        avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
        
        print(f"\n[Episode {episode+1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps in Episode: {step+1}")
        print(f"  Global Steps: {global_step}")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        
        # Salva log
        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{episode_reward:.2f},{step+1},{global_step}\n")
        
        # Salva best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir, 'best_model_ppo.pth')
            agent.save(best_path)
            print(f"  [NEW BEST] Saved to {best_path}")
        
        # Salva periodicamente
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'episode_{episode+1}.pth')
            agent.save(checkpoint_path)
            agent.save(latest_checkpoint)
            print(f"  [Checkpoint] Saved to {checkpoint_path}")
    
    # ============ FINE TRAINING ============
    print(f"\n{'='*60}")
    print(f"PPO Training Completed!")
    print(f"{'='*60}")
    print(f"Best Reward: {best_reward:.2f}")
    
    final_path = os.path.join(checkpoint_dir, 'final_model_ppo.pth')
    agent.save(final_path)
    
    env.close()

if __name__ == "__main__":
    # Hyperparameters per PPO
    HYPERPARAMS = {
        'num_episodes': 1000,
        'max_steps_per_episode': 5000,
        'update_timestep': 2048,   # Quanto spesso aggiornare la rete
        'learning_rate': 3e-4,     # LR tipico per PPO
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.2,
        'n_epochs': 10,
        'save_freq': 50,
        'checkpoint_dir': 'checkpoints_ppo',
        'host': 'localhost',
        'port': 5555
    }
    
    print("PPO Hyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    
    try:
        train_ppo(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train PPO] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train PPO] Error during training: {e}")
        import traceback
        traceback.print_exc()