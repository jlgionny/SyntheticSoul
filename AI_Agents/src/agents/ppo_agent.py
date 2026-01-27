import torch
import torch.optim as optim
import numpy as np
import os
from src.models.actor_critic import ActorCritic


class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, 32)  # Batch size 32
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + 32] for i in batch_start]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=5e-5,  # Ridotto da 2e-4
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,  # Aumentato da 0.15
        n_epochs=4,
        device=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"[PPOAgent] Using device: {self.device}")

        self.policy = ActorCritic(
            state_size,
            action_size,
            hidden_size=256,  # Ridotto da 384 (meno capacity = più stabile)
            use_lstm=True,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.memory = PPOMemory()
        print(
            f"[PPOAgent] Initialized with state_size={state_size}, action_size={action_size}"
        )
        print(f"[PPOAgent] Hidden size: 256, LSTM: Enabled, LR: {learning_rate}")

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.memory.store_memory(state, action, log_prob, value, reward, done)

    def learn(self):
        if len(self.memory.states) == 0:
            return None

        (
            state_arr,
            action_arr,
            old_prob_arr,
            vals_arr,
            reward_arr,
            dones_arr,
            batches,
        ) = self.memory.generate_batches()

        values = vals_arr

        # GAE calculation
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        last_advantage = 0

        for t in reversed(range(len(reward_arr))):
            mask = 1.0 - dones_arr[t]
            last_value = values[t + 1] if (t + 1) < len(reward_arr) else 0.0
            delta = reward_arr[t] + self.gamma * last_value * mask - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * mask * last_advantage
            last_advantage = advantage[t]

        # Normalize advantages (STABILITÀ)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        advantage = torch.tensor(advantage, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        old_probs_tensor = torch.tensor(old_prob_arr, dtype=torch.float32).to(
            self.device
        )

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.n_epochs):
            for batch_idx, batch in enumerate(batches):
                self.policy.reset_hidden()

                states = torch.tensor(state_arr[batch], dtype=torch.float).to(
                    self.device
                )
                old_probs = old_probs_tensor[batch]
                actions = torch.tensor(action_arr[batch]).to(self.device)

                new_probs, critic_value, dist_entropy = self.policy.evaluate(
                    states, actions
                )
                critic_value = critic_value.squeeze()

                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()
                entropy_loss = -dist_entropy.mean()

                # CRITICAMENTE IMPORTANTE: entropy bonus più forte
                total_loss = (
                    actor_loss + 0.5 * critic_loss + 0.05 * entropy_loss
                )  # 0.01 → 0.05

                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping più aggressivo
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.2
                )  # 0.5 → 0.2
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
                num_batches += 1

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(
                    f"    [Epoch {epoch+1}/{self.n_epochs}] Loss: {total_loss.item():.4f}"
                )

        avg_metrics = {
            "actor_loss": total_actor_loss / num_batches if num_batches > 0 else 0.0,
            "critic_loss": total_critic_loss / num_batches if num_batches > 0 else 0.0,
            "entropy": total_entropy / num_batches if num_batches > 0 else 0.0,
        }

        self.policy.reset_hidden()
        self.memory.clear_memory()

        return avg_metrics

    def reset_hidden(self):
        self.policy.reset_hidden()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.policy.load_state_dict(torch.load(filename, map_location=self.device))
            print(f"[PPOAgent] Model loaded from {filename}")
        else:
            print(f"[PPOAgent] Warning: No checkpoint found at {filename}")
