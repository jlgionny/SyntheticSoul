"""Processes game state into normalized features"""

import numpy as np


class StateProcessor:
    def __init__(self):
        self.prev_boss_health = None
        self.prev_player_health = None
        
    def process(self, raw_state):
        """Convert raw game state to normalized feature vector"""
        if raw_state is None:
            return None
        
        # Normalize positions (assuming arena is roughly -50 to 50 units)
        player_x_norm = raw_state['playerX'] / 50.0
        player_y_norm = raw_state['playerY'] / 50.0
        boss_x_norm = raw_state['bossX'] / 50.0
        boss_y_norm = raw_state['bossY'] / 50.0
        
        # Normalize velocities (assuming max velocity ~20)
        vel_x_norm = np.clip(raw_state['playerVelocityX'] / 20.0, -1, 1)
        vel_y_norm = np.clip(raw_state['playerVelocityY'] / 20.0, -1, 1)
        
        # Normalize health and soul
        player_health_norm = raw_state['playerHealth'] / max(raw_state['playerMaxHealth'], 1)
        boss_health_norm = raw_state['bossHealth'] / max(raw_state['bossMaxHealth'], 1)
        soul_norm = raw_state['playerSoul'] / 99.0  # Max soul is 99
        
        # Normalize distance (assuming max relevant distance is 100)
        distance_norm = min(raw_state['distanceToBoss'] / 100.0, 1.0)
        
        # Relative position to boss
        rel_x = (raw_state['bossX'] - raw_state['playerX']) / 50.0
        rel_y = (raw_state['bossY'] - raw_state['playerY']) / 50.0
        
        # Create feature vector
        features = np.array([
            player_x_norm,
            player_y_norm,
            vel_x_norm,
            vel_y_norm,
            player_health_norm,
            soul_norm,
            boss_x_norm,
            boss_y_norm,
            boss_health_norm,
            distance_norm,
            rel_x,
            rel_y,
            float(raw_state['canDash']),
            float(raw_state['canAttack']),
            float(raw_state['isGrounded']),
            float(raw_state['isDead']),
            float(raw_state['bossDefeated']),
            float(raw_state.get('hasDoubleJump', False)),  # Add hasDoubleJump
        ], dtype=np.float32)
        
        return features
    
    def calculate_reward(self, prev_state, current_state):
        """Calculate reward based on state transition"""
        if prev_state is None or current_state is None:
            return 0.0
        
        reward = 0.0
        
        # Boss damage dealt
        boss_damage = prev_state['bossHealth'] - current_state['bossHealth']
        if boss_damage > 0:
            reward += boss_damage * 20.0
        
        # Player damage taken
        player_damage = prev_state['playerHealth'] - current_state['playerHealth']
        if player_damage > 0:
            reward -= player_damage * 10.0
        
        # Distance reward (encourage staying in optimal range)
        optimal_distance = 5.0  # Optimal attack range
        prev_dist_diff = abs(prev_state['distanceToBoss'] - optimal_distance)
        curr_dist_diff = abs(current_state['distanceToBoss'] - optimal_distance)
        
        if curr_dist_diff < prev_dist_diff:
            reward += 0.5
        
        # Boss defeated
        if current_state['bossDefeated'] and not prev_state['bossDefeated']:
            reward += 1000.0
        
        # Player died
        if current_state['isDead'] and not prev_state['isDead']:
            reward -= 500.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def is_terminal(self, state):
        """Check if state is terminal (episode should end)"""
        if state is None:
            return True
        return state['isDead'] or state['bossDefeated']
