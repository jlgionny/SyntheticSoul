"""Deep Q-Network agent implementation"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from config import *

class DQNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        # Replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Q-Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def replay(self, batch_size=BATCH_SIZE):
        """
        FIX: Train on a batch of experiences with optimized tensor conversion.
        Converts lists to numpy arrays before creating tensors to avoid performance warning.
        """
        if len(self.memory) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # FIX: Extract batch components and convert to numpy arrays first
        # This is MUCH faster than torch.FloatTensor([list of arrays])
        states_list = [exp[0] for exp in batch]
        actions_list = [exp[1] for exp in batch]
        rewards_list = [exp[2] for exp in batch]
        next_states_list = [exp[3] for exp in batch]
        dones_list = [exp[4] for exp in batch]
        
        # FIX: Use numpy to stack arrays efficiently before converting to tensors
        states_np = np.array(states_list, dtype=np.float32)
        next_states_np = np.array(next_states_list, dtype=np.float32)
        
        # Convert to tensors (now fast since input is numpy array, not list of arrays)
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        rewards = torch.FloatTensor(rewards_list).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_list).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath=MODEL_SAVE_PATH):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath=MODEL_SAVE_PATH):
        """Load model weights"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
            return True
        return False
