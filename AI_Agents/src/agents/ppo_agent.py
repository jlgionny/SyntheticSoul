"""
PPO Agent with LSTM for Hollow Knight Mantis Lords boss fight.
V3 - BALANCED AGGRO:
1. Batch Size Safety Check in forward() to prevent crashes.
2. LSTM state reset removed from batch loop to preserve memory flow.
3. SmoothL1Loss instead of MSE for stability.
4. NEW: Learning Rate Scheduler support.
5. NEW: Dynamic entropy coefficient (set externally by training loop).
"""

from typing import Tuple, Dict, Optional
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

from actor_critic import ActorCritic


class ReplayBuffer:
    """Buffer for storing trajectory data."""

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
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions": torch.LongTensor(self.actions),
            "log_probs": torch.FloatTensor(self.log_probs),
            "values": torch.FloatTensor(self.values),
            "rewards": torch.FloatTensor(self.rewards),
            "dones": torch.FloatTensor(self.dones),
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class KillBuffer:
    """
    Stores complete trajectories from kill episodes for replay.
    Only stores states, actions, rewards, dones — log_probs and values
    are re-evaluated with current policy before each learn pass.
    """

    def __init__(self, max_episodes: int = 30):
        self.episodes = deque(maxlen=max_episodes)

    def add_episode(self, states: list, actions: list, rewards: list, dones: list):
        self.episodes.append(
            {
                "states": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards, dtype=np.float32),
                "dones": np.array(dones, dtype=np.float32),
            }
        )

    def sample(self) -> dict:
        return random.choice(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)


class PPOAgent:
    """
    Proximal Policy Optimization agent with LSTM.
    V3: LR Scheduling + Dynamic Entropy.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 384,
        use_lstm: bool = True,
        learning_rate: float = 1e-4,
        lr_end_factor: float = 0.3,  # LR finale = lr * lr_end_factor
        total_episodes: int = 1000,  # Per calcolare il decay del LR
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.05,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 128,
        device: Optional[str] = None,
    ):
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

        self.policy = ActorCritic(state_size, action_size, hidden_size, use_lstm).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # === NEW: LR Scheduler (linear decay) ===
        # Decadimento lineare: lr_start -> lr_start * lr_end_factor
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda ep: max(
                lr_end_factor,
                1.0 - (1.0 - lr_end_factor) * (ep / max(total_episodes, 1)),
            ),
        )

        self.buffer = ReplayBuffer()
        self.kill_buffer = KillBuffer(max_episodes=30)

        print(f"[PPO] Initialized on {self.device}")
        print(
            f"[PPO] LSTM: {use_lstm} | LR: {learning_rate} -> {learning_rate * lr_end_factor}"
        )
        print(f"[PPO] Entropy: {entropy_coef} (dynamic, set by training loop)")
        print(f"[PPO] Kill Buffer: max 30 episodes")

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
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
        self.buffer.add(state, action, log_prob, value, reward, done)

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Protezione: std() su 1 solo elemento = NaN, skip normalizzazione
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def learn(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < 2:
            self.buffer.clear()
            return None

        # Save online hidden state before batch processing
        saved_hidden = None
        if self.policy.use_lstm and self.policy.hidden_state is not None:
            saved_hidden = (
                self.policy.hidden_state[0].clone(),
                self.policy.hidden_state[1].clone(),
            )

        data = self.buffer.get()
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        old_values = data["values"].to(self.device)

        advantages, returns = self.compute_gae(rewards, old_values, dones)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.n_epochs):
            self.policy.reset_hidden()

            num_samples = len(states)
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                batch_old_log_probs = old_log_probs[start_idx:end_idx]
                batch_advantages = advantages[start_idx:end_idx]
                batch_returns = returns[start_idx:end_idx]

                log_probs, values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                values_squeezed = values.squeeze(-1)
                if values_squeezed.dim() == 0:
                    values_squeezed = values_squeezed.unsqueeze(0)

                critic_loss = nn.SmoothL1Loss()(values_squeezed, batch_returns)

                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.buffer.clear()

        # Restore online hidden state (don't lobotomize mid-episode)
        if self.policy.use_lstm:
            self.policy.hidden_state = saved_hidden

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def learn_from_kills(self) -> Optional[Dict[str, float]]:
        """
        Extra PPO update on a sampled kill episode.
        Re-evaluates with current policy for fresh log_probs/values,
        then does 2 PPO epochs. This gives consistent gradient towards
        kill behavior even when current episodes are non-kills.
        """
        if len(self.kill_buffer) == 0:
            return None

        # Save online hidden state
        saved_hidden = None
        if self.policy.use_lstm and self.policy.hidden_state is not None:
            saved_hidden = (
                self.policy.hidden_state[0].clone(),
                self.policy.hidden_state[1].clone(),
            )

        episode = self.kill_buffer.sample()
        states = torch.FloatTensor(episode["states"]).to(self.device)
        actions = torch.LongTensor(episode["actions"]).to(self.device)
        rewards = torch.FloatTensor(episode["rewards"]).to(self.device)
        dones = torch.FloatTensor(episode["dones"]).to(self.device)

        if len(states) < 2:
            return None

        # Re-evaluate with current policy to get fresh baselines
        self.policy.reset_hidden()
        with torch.no_grad():
            old_log_probs, old_values, _ = self.policy.evaluate(states, actions)
            old_values = old_values.squeeze(-1)
            if old_values.dim() == 0:
                old_values = old_values.unsqueeze(0)

        advantages, returns = self.compute_gae(rewards, old_values, dones)

        # PPO update — fewer epochs since slightly off-policy
        for epoch in range(2):
            self.policy.reset_hidden()

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))

                b_states = states[start_idx:end_idx]
                b_actions = actions[start_idx:end_idx]
                b_old_lp = old_log_probs[start_idx:end_idx]
                b_adv = advantages[start_idx:end_idx]
                b_ret = returns[start_idx:end_idx]

                log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)

                ratio = torch.exp(log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * b_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                values_sq = values.squeeze(-1)
                if values_sq.dim() == 0:
                    values_sq = values_sq.unsqueeze(0)
                critic_loss = nn.SmoothL1Loss()(values_sq, b_ret)

                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Restore online hidden state
        if self.policy.use_lstm:
            self.policy.hidden_state = saved_hidden
        return {"kill_replay": True}

    def step_scheduler(self):
        """Avanza il LR scheduler di uno step. Chiamare una volta per episodio."""
        self.scheduler.step()

    def get_current_lr(self) -> float:
        """Ritorna il LR attuale."""
        return self.optimizer.param_groups[0]["lr"]

    def reset_hidden(self):
        self.policy.reset_hidden()

    def save(self, filepath: str):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        # NON carichiamo l'optimizer: vogliamo un optimizer fresco per la Fase 2
        # così il LR scheduler parte pulito e i momenti Adam non portano bias
        # dalla fase precedente (che aveva reward function diversa).
        print(f"[PPO] Loaded weights from {filepath} (optimizer reset for new phase)")
