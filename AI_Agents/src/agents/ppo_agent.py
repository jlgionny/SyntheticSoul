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
        batch_start = np.arange(0, n_states, 64)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+64] for i in batch_start]
        
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

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
    def __init__(self, state_size, action_size, learning_rate=2e-4, gamma=0.995, 
                 gae_lambda=0.97, policy_clip=0.15, n_epochs=8, device=None):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"[PPOAgent] Using device: {self.device}")
        
        # Inizializza Actor-Critic Network con LSTM e hidden_size aumentato
        self.policy = ActorCritic(
            state_size, 
            action_size, 
            hidden_size=384,  # Aumentato da 256
            use_lstm=True     # Abilita LSTM
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Buffer
        self.memory = PPOMemory()
        
        print(f"[PPOAgent] Initialized with state_size={state_size}, action_size={action_size}")
        print(f"[PPOAgent] Hidden size: 384, LSTM: Enabled")

    def select_action(self, state):
        """
        Seleziona azione per interagire con l'environment.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        """Memorizza lo step nel buffer temporaneo"""
        self.memory.store_memory(state, action, log_prob, value, reward, done)

    def learn(self):
        """
        Cuore del PPO: Calcola advantages e aggiorna la rete per N epoche.
        """
        # Recupera i dati dal buffer
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()
        
        values = vals_arr
        
        # --- Calcolo GAE (Generalized Advantage Estimation) ---
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        last_advantage = 0
        
        for t in reversed(range(len(reward_arr))):
            # Se è done, il valore futuro è 0
            mask = 1.0 - dones_arr[t]
            last_value = values[t+1] if (t + 1) < len(reward_arr) else 0.0
            delta = reward_arr[t] + self.gamma * last_value * mask - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * mask * last_advantage
            last_advantage = advantage[t]
        
        # Converti in tensori
        advantage = torch.tensor(advantage).to(self.device)
        values = torch.tensor(values).to(self.device)
        
        # Reset LSTM hidden state prima di batch training
        self.policy.reset_hidden()
        
        # Loop di ottimizzazione (Epochs)
        for epoch in range(self.n_epochs):
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                
                # Valuta i nuovi log_probs e values per gli stati nel batch
                new_probs, critic_value, dist_entropy = self.policy.evaluate(states, actions)
                
                # Critic value shape fix
                critic_value = critic_value.squeeze()
                
                # Ratio per il PPO (pi_new / pi_old)
                prob_ratio = torch.exp(new_probs - old_probs)
                
                # Surrogate Loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 
                    1 - self.policy_clip, 
                    1 + self.policy_clip
                ) * advantage[batch]
                
                # Loss Totale
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Critic Loss
                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value)**2).mean()
                
                # Entropy Loss
                entropy_loss = -dist_entropy.mean()
                
                # Combine losses
                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Svuota la memoria dopo l'update!
        self.memory.clear_memory()

    def reset_hidden(self):
        """Reset LSTM hidden state (chiamare a inizio episodio)."""
        self.policy.reset_hidden()

    def save(self, filename):
        """Salva i pesi del modello."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        """Carica i pesi del modello."""
        if os.path.exists(filename):
            self.policy.load_state_dict(torch.load(filename, map_location=self.device))
            print(f"[PPOAgent] Model loaded from {filename}")
        else:
            print(f"[PPOAgent] Warning: No checkpoint found at {filename}")