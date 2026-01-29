import os
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim


from src.models.dqn_net import DuelingDQN


# Struttura dati per le transizioni
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Replay Buffer per salvare e campionare esperienze.
    Usa una deque per efficienza con limite automatico.
    """

    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


class DQNAgent:
    """
    Deep Q-Network Agent per Hollow Knight.
    """

    def __init__(
        self,
        state_size,
        action_size=8,
        hidden_sizes=[128, 256, 128],
        learning_rate=1e-4,
        gamma=0.99,
        buffer_capacity=100000,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # ==============================================================================
        # 2. QUI USIAMO "DuelingDQN" INVECE DI "DQN" (La modifica fondamentale)
        # ==============================================================================
        self.policy_net = DuelingDQN(state_size, action_size, hidden_sizes).to(
            self.device
        )
        self.target_net = DuelingDQN(state_size, action_size, hidden_sizes).to(
            self.device
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function (Smooth L1 Loss per stabilità)
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Training stats
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state, epsilon=0.0):
        # Epsilon-greedy: exploration vs exploitation
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).to(self.device)
                elif not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)

                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # policy_net ora è una DuelingDQN
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()

                return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Calcolo Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Calcolo V(s') con Double DQN per stabilità
        with torch.no_grad():
            # Usa Policy Net per scegliere l'azione migliore nel next state
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            # Usa Target Net per calcolare il valore di quell'azione
            next_state_values = (
                self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            )

            target_values = reward_batch + (
                self.gamma * next_state_values * (1 - done_batch)
            )

        # Huber Loss (SmoothL1) è meglio di MSE per evitare esplosioni
        loss = self.criterion(state_action_values.squeeze(), target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # CLIPPING MOLTO PIÙ AGGRESSIVO (Era 10.0, ora 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.steps_done += 1
        return loss.item()

    def update_target_network(self, tau=1.0):
        """
        Soft update dei parametri:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, filepath)
        # print(f"[DQNAgent] Model saved to {filepath}")

    def load(self, filepath, load_optimizer=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.steps_done = checkpoint.get("steps_done", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)

        print(f"[DQNAgent] Model loaded from {filepath}")
        print(f"  Steps: {self.steps_done}, Episodes: {self.episodes_done}")

    def get_epsilon(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
            -1.0 * self.steps_done / epsilon_decay
        )
        return epsilon
