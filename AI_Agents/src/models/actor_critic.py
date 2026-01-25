import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        # Layer condivisi (Feature Extractor)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Actor Head (Probabilità azioni)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic Head (Valore dello stato)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.feature_layer(x)

    def get_action(self, state):
        features = self.forward(state)
        action_logits = self.actor(features)
        dist = Categorical(logits=action_logits)
        
        action = dist.sample()
        value = self.critic(features)
        
        return action, dist.log_prob(action), value

    def evaluate(self, state, action):
        features = self.forward(state)
        action_logits = self.actor(features)
        dist = Categorical(logits=action_logits)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(features)
        
        return action_logprobs, value, dist_entropy