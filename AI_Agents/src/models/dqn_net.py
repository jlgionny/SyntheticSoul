import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network per l'agente di Hollow Knight.

    Architettura:
    - Input: stato del gioco (player, boss, hazards, environment) ~20-25 features
    - Hidden layers: 128 -> 256 -> 128 neuroni con ReLU
    - Output: 8 Q-values (una per ogni azione)

    Azioni:
    0: Left, 1: Right, 2: Up, 3: Down, 4: Jump, 5: Attack, 6: Dash, 7: Cast
    """

    def __init__(self, state_size, action_size=8, hidden_sizes=[128, 256, 128]):
        """
        Inizializza la rete DQN.

        Args:
            state_size (int): Dimensione dello stato di input
            action_size (int): Numero di azioni possibili (default: 8)
            hidden_sizes (list): Lista con il numero di neuroni per ogni hidden layer
        """
        super(DQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Layer 1: Input -> Hidden1
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])

        # Layer 2: Hidden1 -> Hidden2
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Layer 3: Hidden2 -> Hidden3
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        # Output layer: Hidden3 -> Q-values
        self.fc4 = nn.Linear(hidden_sizes[2], action_size)

        # Inizializzazione pesi (Xavier uniform)
        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializza i pesi della rete con Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        """
        Forward pass della rete.

        Args:
            state (torch.Tensor): Stato del gioco [batch_size, state_size]

        Returns:
            torch.Tensor: Q-values per ogni azione [batch_size, action_size]
        """
        # Layer 1 con ReLU
        x = F.relu(self.fc1(state))

        # Layer 2 con ReLU
        x = F.relu(self.fc2(x))

        # Layer 3 con ReLU
        x = F.relu(self.fc3(x))

        # Output layer (nessuna attivazione, Q-values possono essere negativi)
        q_values = self.fc4(x)

        return q_values

    def act(self, state, epsilon=0.0):
        """
        Seleziona un'azione usando epsilon-greedy policy.

        Args:
            state (torch.Tensor): Stato corrente [state_size] o [batch_size, state_size]
            epsilon (float): Probabilità di azione random (exploration)

        Returns:
            int: Indice dell'azione selezionata
        """
        # Esplorazione random
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()

        # Exploitation: scegli l'azione con Q-value più alto
        with torch.no_grad():
            # Assicurati che lo stato abbia dimensione batch
            if state.dim() == 1:
                state = state.unsqueeze(0)

            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action

    def get_q_values(self, state):
        """
        Ottiene i Q-values per uno stato senza gradient tracking.

        Args:
            state (torch.Tensor): Stato del gioco

        Returns:
            torch.Tensor: Q-values per tutte le azioni
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN - Architettura avanzata che separa Value e Advantage streams.

    Opzionale: Usa questa versione per migliorare le performance dell'agente.
    Migliora la learning stability separando il valore dello stato (V)
    dal vantaggio di ogni azione (A).
    """

    def __init__(self, state_size, action_size=8, hidden_sizes=[128, 256, 128]):
        """Inizializza Dueling DQN."""
        super(DuelingDQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Value stream
        self.value_fc = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.value_out = nn.Linear(hidden_sizes[2], 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.advantage_out = nn.Linear(hidden_sizes[2], action_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializza i pesi."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        """
        Forward pass con Dueling architecture.

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        # Shared layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)  # [batch_size, 1]

        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)  # [batch_size, action_size]

        # Combina: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def act(self, state, epsilon=0.0):
        """Seleziona azione con epsilon-greedy."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action

    def get_q_values(self, state):
        """Ottiene Q-values senza gradient."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state)


# Esempio di utilizzo
if __name__ == "__main__":
    # Parametri
    STATE_SIZE = 25  # player(13) + boss(5) + hazards(5*3=15) + environment(4) = ~25-30
    ACTION_SIZE = 8  # Left, Right, Up, Down, Jump, Attack, Dash, Cast

    # Crea la rete
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(STATE_SIZE, ACTION_SIZE).to(device)

    print(f"DQN Network Architecture:")
    print(dqn)
    print(f"\nTotal parameters: {sum(p.numel() for p in dqn.parameters())}")

    # Test forward pass
    dummy_state = torch.randn(1, STATE_SIZE).to(device)
    q_values = dqn(dummy_state)
    print(f"\nInput shape: {dummy_state.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")

    # Test action selection
    action = dqn.act(dummy_state.squeeze(), epsilon=0.1)
    print(f"\nSelected action (epsilon=0.1): {action}")
