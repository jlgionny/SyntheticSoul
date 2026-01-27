import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=384, use_lstm=True):
        """
        Actor-Critic con LSTM per gestire temporalità dei boomerang.

        Args:
            state_size: Dimensione input (39 con 6 hazard)
            action_size: Numero azioni (8)
            hidden_size: Hidden layer size (aumentato a 384)
            use_lstm: Se True, usa LSTM per memoria temporale
        """
        super(ActorCritic, self).__init__()
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size

        # Feature Extractor (più profondo)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),  # Stabilizza training
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        # LSTM per memoria temporale (boomerang che tornano indietro)
        if self.use_lstm:
            self.lstm = nn.LSTM(
                hidden_size, hidden_size, num_layers=1, batch_first=True
            )
            self.hidden_state = None

        # Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

        # Critic Head (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        """Feature extraction con supporto LSTM."""
        features = self.feature_layer(x)

        if self.use_lstm:
            # Se x ha batch_size, aggiungi dimensione temporale
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # [batch, 1, hidden]

            features, self.hidden_state = self.lstm(features, self.hidden_state)

            # FIX CRITICO: Detach l'hidden state per prevenire il backward attraverso il grafo
            # Questo permette all'LSTM di mantenere la memoria temporale senza mantenere
            # il grafo computazionale che causa l'errore
            if self.hidden_state is not None:
                self.hidden_state = (
                    self.hidden_state[0].detach(),
                    self.hidden_state[1].detach(),
                )

            # Rimuovi dimensione temporale
            if features.shape[1] == 1:
                features = features.squeeze(1)

        return features

    def get_action(self, state):
        """Campiona azione dalla policy."""
        features = self.forward(state)
        action_logits = self.actor(features)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        value = self.critic(features)
        return action, dist.log_prob(action), value

    def evaluate(self, state, action):
        """Valuta azioni per PPO update."""
        features = self.forward(state)
        action_logits = self.actor(features)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(features)
        return action_logprobs, value, dist_entropy

    def reset_hidden(self):
        """Reset LSTM hidden state (chiamare ad ogni episodio)."""
        if self.use_lstm:
            self.hidden_state = None
