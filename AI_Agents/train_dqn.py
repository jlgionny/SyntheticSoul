import os
import sys
import time
import json
import numpy as np
import torch
from datetime import datetime

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.dqn_agent import DQNAgent
from src.env.hollow_knight_env import HollowKnightEnv


class RewardCalculator:
    """
    Reward Function per Hollow Knight AI.
    
    Incentiva:
    - Danneggiare il boss
    - Avvicinarsi al boss
    - Sopravvivenza
    - Efficienza temporale
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
        """
        Calcola il reward per una transizione.
        
        Args:
            state_dict (dict): Stato corrente
            prev_state (dict): Stato precedente
            done (bool): Se l'episodio è terminato
            info (dict): Informazioni aggiuntive (contiene action_name)
        
        Returns:
            float: Reward totale
        """
        reward = 0.0
        
        # ============ REWARD PER DANNO AL BOSS ============
        if prev_state is not None and 'bossHealth' in state_dict and 'bossHealth' in prev_state:
            boss_damage = prev_state['bossHealth'] - state_dict['bossHealth']
            if boss_damage > 0:
                reward += boss_damage * 2.0
                print(f"  [Reward] Boss damage: +{boss_damage * 2.0:.2f}")
        
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
        # FIX: Penalizza se si muove contro un muro
        if info is not None and 'action_name' in info:
            action_name = info['action_name']
            terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
            
            # terrainInfo[2] = Wall Ahead (distanza dal muro frontale)
            # Se < 0.15 (molto vicino) e stiamo provando a muoverci, penalizza
            if len(terrain_info) >= 3:
                wall_distance = terrain_info[2]
                
                if wall_distance < 0.15:  # Muro molto vicino
                    if action_name in ['MOVE_LEFT', 'MOVE_RIGHT']:
                        reward -= 20.0
                        # print(f"  [Reward] Wall collision penalty: -2.0 (Wall dist: {wall_distance:.2f})")
        
        # ============ REWARD/PENALITÀ PER MORTE ============
        if done:
            if state_dict.get('isDead', False):
                reward -= 100.0
                print(f"  [Reward] Player died: -100.0")
            elif state_dict.get('bossDefeated', False):
                reward += 500.0
                print(f"  [Reward] Boss defeated: +500.0")
        
        # ============ BONUS PER SCHIVATA ============
        # FIX: Calcola distanza manualmente da relX e relY
        hazards = state_dict.get('nearbyHazards', [])
        if hazards:
            # Calcola le distanze usando relX e relY
            distances = []
            for h in hazards:
                # Usa get con default alto per sicurezza
                rx = h.get('relX', 100.0)
                ry = h.get('relY', 100.0)
                dist = (rx**2 + ry**2)**0.5
                distances.append(dist)
            
            if distances:
                closest_dist = min(distances)
                # Se siamo vicini a un pericolo (< 3.0 unità) ma non abbiamo perso vita
                if closest_dist < 3.0:
                    if prev_state is not None and state_dict['playerHealth'] == prev_state.get('playerHealth', 0):
                        reward += 0.5
                        # print(f"  [Reward] Dodge bonus: +0.5 (Dist: {closest_dist:.2f})")
        
        # ============ PENALITÀ PER CADUTA ============
        # Usa terrainInfo[0] (Floor Distance) invece di floorDistance separato
        terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
        if len(terrain_info) >= 1:
            floor_distance = terrain_info[0]
            # Se non siamo a terra E il pavimento è lontano (alto valore = lontano)
            if not state_dict.get('isGrounded', True) and floor_distance > 0.6:
                reward -= 5.0
                # print(f"  [Reward] Falling penalty: -5.0 (Floor dist: {floor_distance:.2f})")
        
        return reward





def preprocess_state(state_dict):
    """
    Converte lo stato del gioco in un array numpy per la rete.
    
    Args:
        state_dict (dict): Stato JSON dal gioco
    
    Returns:
        np.ndarray: Array di features
    """
    features = []
    
    # ========== PLAYER FEATURES (10 features) ==========
    features.append(state_dict.get('playerX', 0.0))
    features.append(state_dict.get('playerY', 0.0))
    features.append(state_dict.get('playerVelocityX', 0.0))
    features.append(state_dict.get('playerVelocityY', 0.0))
    features.append(state_dict.get('playerHealth', 0) / 10.0)  # Normalizza
    features.append(state_dict.get('playerSoul', 0) / 100.0)  # Normalizza
    features.append(float(state_dict.get('canDash', False)))
    features.append(float(state_dict.get('canAttack', False)))
    features.append(float(state_dict.get('isGrounded', False)))
    features.append(float(state_dict.get('hasDoubleJump', False)))
    
    # ========== TERRAIN INFO (5 features) ==========
    # FIX: Usa terrainInfo array invece di campi separati
    terrain_info = state_dict.get('terrainInfo', [1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Assicurati che abbiamo 5 valori (fallback se mancanti)
    if len(terrain_info) < 5:
        terrain_info = list(terrain_info) + [1.0] * (5 - len(terrain_info))
    
    # terrainInfo[0] = Floor Distance (sotto)
    # terrainInfo[1] = Gap Ahead (avanti-basso, diagonale)
    # terrainInfo[2] = Wall Ahead (avanti, orizzontale)
    # terrainInfo[3] = Ceiling Ahead (avanti-alto, diagonale)
    # terrainInfo[4] = Ceiling Distance (sopra)
    features.extend(terrain_info[:5])  # Già normalizzati (0-1) dal C#
    
    # ========== BOSS FEATURES (4 features) ==========
    features.append(state_dict.get('bossX', 0.0))
    features.append(state_dict.get('bossY', 0.0))
    features.append(state_dict.get('bossHealth', 0) / 1000.0)  # Normalizza
    features.append(state_dict.get('distanceToBoss', 100.0) / 20.0)  # Normalizza
    
    # ========== NEARBY HAZARDS (top 3, 2 features each = 6 features) ==========
    # FIX: Usa relX e relY invece di distance (che non esiste nel JSON C#)
    hazards = state_dict.get('nearbyHazards', [])
    for i in range(3):
        if i < len(hazards):
            h = hazards[i]
            # Usa solo relX e relY (normalizzati)
            features.append(h.get('relX', 0.0) / 15.0)  # Normalizza ±15 unità
            features.append(h.get('relY', 0.0) / 15.0)
        else:
            # Padding se meno di 3 hazards
            features.extend([0.0, 0.0])
    
    # TOTALE: 10 (player) + 5 (terrain) + 4 (boss) + 6 (hazards) = 25 features
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
    checkpoint_dir='checkpoints',
    host='localhost',
    port=5555
):
    """
    Loop principale di training.
    
    Args:
        num_episodes (int): Numero di episodi da eseguire
        max_steps_per_episode (int): Step massimi per episodio
        batch_size (int): Batch size per training
        learning_rate (float): Learning rate
        gamma (float): Discount factor
        epsilon_start (float): Epsilon iniziale
        epsilon_end (float): Epsilon finale
        epsilon_decay (int): Decay steps per epsilon
        target_update_freq (int): Frequenza aggiornamento target network
        save_freq (int): Frequenza salvataggio model
        checkpoint_dir (str): Directory per salvare i modelli
        host (str): Host del socket server
        port (int): Porta del socket server
    """
    
    # Crea directory per checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Inizializza environment
    print(f"[Train] Connecting to Hollow Knight at {host}:{port}...")
    env = HollowKnightEnv(host=host, port=port)
    
    # Determina state size dal primo reset
    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8  # Left, Right, Up, Down, Jump, Attack, Dash, Cast
    
    print(f"[Train] State size: {state_size}, Action size: {action_size}")
    
    # Inizializza agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000
    )
    
    # Carica checkpoint se esiste
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print(f"[Train] Resumed from checkpoint")
        except Exception as e:
            print(f"[Train] Could not load checkpoint: {e}")
    
    # Reward calculator
    reward_calc = RewardCalculator()
    
    # Training stats
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    
    # Log file
    log_file = os.path.join(checkpoint_dir, 'training_log.txt')
    
    print(f"\n{'='*60}")
    print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # ============ TRAINING LOOP ============
    for episode in range(num_episodes):
        # Reset environment e reward calculator
        state_dict = env.reset()
        state = preprocess_state(state_dict)
        reward_calc.reset()
        
        # FIX: Inizializza variabili episodio
        episode_reward = 0.0
        episode_loss = []
        prev_state_dict = None
        
        print(f"\n[Episode {episode+1}/{num_episodes}] Starting...")
        
        for step in range(max_steps_per_episode):
            # Calcola epsilon decrescente
            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            
            # Seleziona azione
            action = agent.select_action(state, epsilon=epsilon)
            
            # Esegui azione nell'environment
            next_state_dict, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)
            
            # Calcola reward - FIX: argomenti posizionali corretti
            reward = reward_calc.calculate_reward(
                next_state_dict,    # Primo argomento: state_dict
                prev_state_dict,    # Secondo argomento: prev_state
                done,               # Terzo argomento: done
                info                # Quarto argomento: info
            )
            
            episode_reward += reward
            
            # Salva transizione in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Ottimizza model SOLO se il buffer ha abbastanza samples
            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size=batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                
                # FIX: Aggiorna target network SOLO se steps > 0 e ogni N steps
                if agent.steps_done > 0 and agent.steps_done % target_update_freq == 0:
                    agent.update_target_network()
                    print(f"  [Step {agent.steps_done}] Target network updated")
            
            # Aggiorna stato
            state = next_state
            prev_state_dict = state_dict
            state_dict = next_state_dict
            
            # Log periodico
            if step % 100 == 0:
                print(f"  [Step {step}] Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Buffer: {len(agent.memory)}")
            
            # Fine episodio
            if done:
                reason = 'Player died' if state_dict.get('isDead') else 'Boss defeated' if state_dict.get('bossDefeated') else 'Unknown'
                print(f"  [Episode End] Reason: {reason}")
                break
        
        # ============ FINE EPISODIO ============
        agent.episodes_done += 1
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)
        
        # Statistiche
        avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
        
        print(f"\n[Episode {episode+1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Steps: {step+1}")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        print(f"  Buffer size: {len(agent.memory)}")
        print(f"  Total steps: {agent.steps_done}")
        
        # Salva log
        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{episode_reward:.2f},{avg_loss:.4f},{step+1},{epsilon:.4f}\n")
        
        # Salva best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            agent.save(best_path)
            print(f"  [NEW BEST] Saved to {best_path}")
        
        # Salva periodicamente
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'episode_{episode+1}.pth')
            agent.save(checkpoint_path)
            agent.save(latest_checkpoint)
            print(f"  [Checkpoint] Saved to {checkpoint_path}")


        
        # ============ FINE EPISODIO ============
        agent.episodes_done += 1
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)
        
        # Statistiche
        avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
        
        print(f"\n[Episode {episode+1}] Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Steps: {step+1}")
        print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
        print(f"  Buffer size: {len(agent.memory)}")
        print(f"  Total steps: {agent.steps_done}")
        
        # Salva log
        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{episode_reward:.2f},{avg_loss:.4f},{step+1},{epsilon:.4f}\n")
        
        # Salva best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
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
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    # Salva final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    agent.save(final_path)
    
    env.close()


if __name__ == "__main__":
    # Hyperparameters
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
    
    print("Hyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    
    # Start training
    try:
        train_dqn(**HYPERPARAMS)
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
    except Exception as e:
        print(f"\n[Train] Error during training: {e}")
        import traceback
        traceback.print_exc()
