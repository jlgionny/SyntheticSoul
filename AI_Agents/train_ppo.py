import os
import sys
import time
import json
import numpy as np
import torch
from datetime import datetime

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.ppo_agent import PPOAgent
from src.env.hollow_knight_env import HollowKnightEnv

class RewardCalculator:
    """
    Reward Function ottimizzata per Mantis Lords.
    """
    def __init__(self):
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        self.prev_mantis_killed = 0  # NUOVO: Traccia numero precedente
        self.episode_start_time = None

    def reset(self):
        """Reset per nuovo episodio."""
        self.prev_boss_health = None
        self.prev_player_health = None
        self.prev_distance_to_boss = None
        self.prev_mantis_killed = 0  # NUOVO: Reset a 0 ad ogni episodio
        self.episode_start_time = time.time()

    def calculate_reward(self, state_dict, prev_state, done, info=None):
        reward = 0.0
        
        # ============ REWARD PER DANNO AL BOSS ============
        if prev_state is not None and 'bossHealth' in state_dict and 'bossHealth' in prev_state:
            boss_damage = prev_state['bossHealth'] - state_dict['bossHealth']
            if boss_damage > 0:
                reward += boss_damage * 3.0
                # print(f" [Reward] Boss damage: +{boss_damage * 3.0:.2f}")
        
        # ============ REWARD AGGRESSIVITÀ: ZONA OTTIMALE (5-8 unità) ============
        curr_dist = state_dict.get('distanceToBoss', 100.0)
        
        if 5.0 <= curr_dist <= 8.0:
            reward += 0.2
        elif curr_dist < 3.0:
            reward -= 0.3
        elif curr_dist > 12.0:
            reward -= 0.5
        
        # Reward per avvicinamento/allontanamento intelligente
        if prev_state is not None and 'distanceToBoss' in prev_state:
            prev_dist = prev_state['distanceToBoss']
            distance_change = prev_dist - curr_dist
            
            if prev_dist > 10.0 and distance_change > 0:
                reward += distance_change * 0.1
            elif prev_dist < 4.0 and distance_change < 0:
                reward += abs(distance_change) * 0.1
        
        # ============ PENALITÀ SPUNTONI ESPONENZIALE ============
        terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
        wall_distance = terrain_info[2] if len(terrain_info) >= 3 else 1.0
        
        if wall_distance < 0.1:
            reward -= 50.0
            print(f" [CRITICAL] Wall collision imminent: -{50.0:.2f}")
        elif wall_distance < 0.2:
            reward -= 20.0
            print(f" [WARNING] Too close to spikes: -{20.0:.2f}")
        elif wall_distance < 0.3:
            reward -= 5.0
        
        # Penalità spike hazard rilevati
        hazards = state_dict.get('nearbyHazards', [])
        for h in hazards:
            if h.get('type') == 'spikes':
                spike_dist = h.get('distance', 100.0)
                if spike_dist < 1.5:
                    penalty = 30.0 * np.exp(-spike_dist)
                    reward -= penalty
                    print(f" [Spike Hazard] Distance {spike_dist:.2f}: -{penalty:.2f}")
        
        # ============ REWARD SCHIVATA DINAMICA (BOOMERANG) ============
        if hazards and prev_state is not None:
            active_projectiles = [h for h in hazards if h.get('type') in ['boomerang', 'projectile']]
            
            if active_projectiles:
                closest_proj = min(active_projectiles, key=lambda h: (h.get('relX', 0)**2 + h.get('relY', 0)**2)**0.5)
                proj_dist = (closest_proj.get('relX', 0)**2 + closest_proj.get('relY', 0)**2)**0.5
                
                if proj_dist < 3.0:
                    curr_health = state_dict.get('playerHealth', 0)
                    prev_health = prev_state.get('playerHealth', 0)
                    
                    if curr_health == prev_health:
                        dodge_reward = 2.0 * (3.0 - proj_dist)
                        reward += dodge_reward
                        # print(f" [Dodge Success] Projectile at {proj_dist:.2f}: +{dodge_reward:.2f}")
                        
                        if info and info.get('action_name') == 'DASH':
                            reward += 1.0
        
        # ============ PENALITÀ PER PERDITA VITA ============
        if prev_state is not None and 'playerHealth' in state_dict and 'playerHealth' in prev_state:
            health_loss = prev_state['playerHealth'] - state_dict['playerHealth']
            if health_loss > 0:
                reward -= health_loss * 25.0
                print(f" [Reward] Health lost: -{health_loss * 25.0:.2f}")
        
        # ============ BONUS MANTIS LORDS KILLED (SOLO QUANDO AUMENTA!) ============
        mantis_killed = state_dict.get('mantisLordsKilled', 0)
        
        # Dai il bonus SOLO se il numero è aumentato rispetto al frame precedente
        if mantis_killed > self.prev_mantis_killed:
            # Bonus per OGNI nuova mantide uccisa
            new_kills = mantis_killed - self.prev_mantis_killed
            mantis_bonus = new_kills * 150.0  # 150 per mantide uccisa
            reward += mantis_bonus
            print(f" [MANTIS LORD KILLED] +{new_kills} defeated: +{mantis_bonus:.2f} (Total: {mantis_killed}/3)")
        
        # Aggiorna il tracker per il prossimo frame
        self.prev_mantis_killed = mantis_killed
        
        # ============ PENALITÀ TEMPORALE (ridotta) ============
        reward -= 0.005
        
        # ============ REWARD/PENALITÀ PER MORTE O VITTORIA ============
        if done:
            if state_dict.get('isDead', False):
                reward -= 150.0
                print(f" [Reward] Player died: -150.0")
            elif state_dict.get('bossDefeated', False):
                health_bonus = state_dict.get('playerHealth', 0) * 50.0
                total_victory = 500.0 + health_bonus
                reward += total_victory
                print(f" [Reward] Boss defeated: +{total_victory:.2f} (base 500 + {health_bonus} health bonus)")
        
        # ============ PENALITÀ PER CADUTA ============
        floor_distance = terrain_info[0] if len(terrain_info) >= 1 else 1.0
        if not state_dict.get('isGrounded', True) and floor_distance > 0.6:
            reward -= 3.0
        
        # ============ REWARD PER FACING BOSS ============
        if state_dict.get('isFacingBoss', False) and curr_dist < 10.0:
            reward += 0.05
        
        return reward



def preprocess_state(state_dict):
    """
    Preprocessing ottimizzato per Mantis Lords (6 hazard + features avanzate).
    State size: 10 (player) + 5 (terrain) + 6 (boss) + 18 (6 hazard) = 39 features
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
    
    # ========== TERRAIN INFO (5 features - GIÀ NORMALIZZATI) ==========
    terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    features.extend(terrain_info[:5])
    
    # ========== BOSS FEATURES AVANZATE (6 features) ==========
    features.append(state_dict.get('bossX', 0.0))
    features.append(state_dict.get('bossY', 0.0))
    features.append(state_dict.get('bossHealth', 0) / 1000.0)
    features.append(state_dict.get('distanceToBoss', 100.0) / 20.0)
    
    # NUOVI: Facing Boss + Distanza da Zona Ottimale
    features.append(float(state_dict.get('isFacingBoss', False)))
    
    # Distanza normalizzata da zona ottimale (5-8 unità)
    dist = state_dict.get('distanceToBoss', 100.0)
    optimal_zone = 6.5  # Centro zona ottimale
    zone_deviation = abs(dist - optimal_zone) / 20.0  # Quanto si è lontani dalla zona ideale
    features.append(zone_deviation)
    
    # ========== NEARBY HAZARDS (18 features per 6 hazard) ==========
    hazards = state_dict.get('nearbyHazards', [])
    
    for i in range(6):  # AUMENTATO DA 3 A 6
        if i < len(hazards):
            h = hazards[i]
            
            # Posizione relativa
            rel_x = h.get('relX', 0.0) / 15.0
            rel_y = h.get('relY', 0.0) / 15.0
            features.append(rel_x)
            features.append(rel_y)
            
            # Velocità (CRITICO PER BOOMERANG CHE TORNANO INDIETRO)
            vel_x = h.get('velocityX', 0.0) / 20.0
            features.append(vel_x)
            
        else:
            # Padding per hazard assenti
            features.extend([0.0, 0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def train_ppo(
    num_episodes=2000,
    max_steps_per_episode=6000,
    update_timestep=1800,
    learning_rate=2e-4,
    gamma=0.995,
    gae_lambda=0.97,
    clip_param=0.15,
    n_epochs=8,
    save_freq=25,
    checkpoint_dir='checkpoints_ppo',
    host='localhost',
    port=5555
):
    """
    Loop principale di training per PPO ottimizzato per Mantis Lords.
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
        device=None  # Auto-detect
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
        # Reset environment E LSTM hidden state
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset()
        agent.reset_hidden()  # IMPORTANTE per LSTM
        
        episode_reward = 0.0
        prev_state_dict = None
        
        print(f"\n[Episode {episode+1}/{num_episodes}] Starting...")
        
        for step in range(max_steps_per_episode):
            global_step += 1
            
            # 1. Seleziona azione
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
            
            # 4. Salva transizione
            agent.store_transition(state, action, log_prob, val, reward, done)
            
            # 5. PPO Update
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
    # Hyperparameters ottimizzati per Mantis Lords (20Hz environment)
    HYPERPARAMS = {
        'num_episodes': 2000,
        'max_steps_per_episode': 6000,
        'update_timestep': 1800,
        'learning_rate': 2e-4,
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'clip_param': 0.15,
        'n_epochs': 8,
        'save_freq': 25,
        'checkpoint_dir': 'checkpoints_ppo_mantis',
        'host': 'localhost',
        'port': 5555
    }
    
    print("="*60)
    print("PPO Hyperparameters - Mantis Lords Configuration")
    print("="*60)
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    try:
        train_ppo(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train PPO] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train PPO] Error during training: {e}")
        import traceback
        traceback.print_exc()
