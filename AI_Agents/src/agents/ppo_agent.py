"""
PPO Agent with LSTM for Hollow Knight Mantis Lords boss fight.
Optimized implementation with proper LSTM handling and GAE.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network with LSTM for temporal dependencies."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 384,
        use_lstm: bool = True,
    ):
        super(ActorCritic, self).__init__()
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim

        # Feature extractor with LayerNorm for stability
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # LSTM for temporal memory (boomerang projectiles)
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.hidden_state = None

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with LSTM support."""
        features = self.feature_extractor(x)

        if self.use_lstm:
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add time dim

            features, self.hidden_state = self.lstm(features, self.hidden_state)

            # CRITICAL: Detach hidden state to prevent gradient backprop issues
            if self.hidden_state is not None:
                self.hidden_state = (
                    self.hidden_state[0].detach(),
                    self.hidden_state[1].detach(),
                )

            if features.shape[1] == 1:
                features = features.squeeze(1)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def reset_hidden(self):
        """Reset LSTM hidden state (call at episode start)."""
        if self.use_lstm:
            self.hidden_state = None


class ReplayBuffer:
    """Buffer for storing trajectory data (temporal order preserved)."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """Store transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self) -> Dict[str, torch.Tensor]:
        """Retrieve all data as tensors (temporal order preserved)."""
        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions": torch.LongTensor(self.actions),
            "log_probs": torch.FloatTensor(self.log_probs),
            "values": torch.FloatTensor(self.values),
            "rewards": torch.FloatTensor(self.rewards),
            "dones": torch.FloatTensor(self.dones),
        }

    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """Proximal Policy Optimization agent with LSTM."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 384,
        use_lstm: bool = True,
        learning_rate: float = 3e-5,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.05,  # Inizio moderato per esplorazione
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """Initialize PPO agent."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Network
        self.policy = ActorCritic(state_size, action_size, hidden_size, use_lstm).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer
        self.buffer = ReplayBuffer()

        print(f"[PPO] Initialized on {self.device}")
        print(f"[PPO] LSTM: {use_lstm} | LR: {learning_rate} | Entropy: {entropy_coef}")

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, value, reward, done)

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_done = 0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - next_done)
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO."""
        if len(self.buffer) == 0:
            return None

        # Get data
        data = self.buffer.get()
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        old_values = data["values"].to(self.device)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, old_values, dones)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        # PPO epochs
        for epoch in range(self.n_epochs):
            # Process in mini-batches (SEQUENTIAL ORDER for LSTM)
            num_samples = len(states)
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                batch_old_log_probs = old_log_probs[start_idx:end_idx]
                batch_advantages = advantages[start_idx:end_idx]
                batch_returns = returns[start_idx:end_idx]

                # Reset hidden state for each batch
                self.policy.reset_hidden()

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # PPO clipped loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss - ensure same shape to avoid broadcasting issues
                values_squeezed = values.squeeze(-1)  # Remove last dim only
                if values_squeezed.dim() == 0:
                    values_squeezed = values_squeezed.unsqueeze(0)
                critic_loss = nn.MSELoss()(values_squeezed, batch_returns)

                # Total loss
                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Clear buffer
        self.buffer.clear()

        # CRITICAL FIX: Reset hidden state after training to avoid batch size mismatch
        # After training with batch_size=64, hidden state has shape [1, 64, 384]
        # But select_action() expects [1, 1, 384] for single sample inference
        self.policy.reset_hidden()

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def reset_hidden(self):
        """Reset LSTM hidden state."""
        self.policy.reset_hidden()

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[PPO] Loaded from {filepath}")
