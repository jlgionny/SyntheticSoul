import os
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.dqn_net import DQN


# Struttura dati per le transizioni
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Replay Buffer per salvare e campionare esperienze.
    Usa una deque per efficienza con limite automatico.
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity (int): Dimensione massima del buffer
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Aggiunge una transizione al buffer.
        
        Args:
            state: Stato corrente
            action: Azione eseguita
            reward: Reward ricevuto
            next_state: Prossimo stato
            done: Se l'episodio è terminato
        """
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Campiona un batch random di transizioni.
        
        Args:
            batch_size (int): Numero di transizioni da campionare
        
        Returns:
            tuple: Batch di (states, actions, rewards, next_states, dones)
        """
        # Campiona batch_size transizioni random
        transitions = random.sample(self.memory, batch_size)
        
        # Converte list of Transitions in Transition of lists
        batch = Transition(*zip(*transitions))
        
        return batch
    
    def __len__(self):
        """Restituisce il numero di transizioni nel buffer."""
        return len(self.memory)
    
    def clear(self):
        """Svuota il buffer."""
        self.memory.clear()


class DQNAgent:
    """
    Deep Q-Network Agent per Hollow Knight.
    
    Implementa:
    - Policy network e Target network
    - Experience Replay
    - Epsilon-greedy action selection
    - Model optimization con TD-learning
    """
    
    def __init__(
        self,
        state_size,
        action_size=8,
        hidden_sizes=[128, 256, 128],
        learning_rate=1e-4,
        gamma=0.99,
        buffer_capacity=100000,
        device=None
    ):
        """
        Inizializza il DQN Agent.
        
        Args:
            state_size (int): Dimensione dello stato
            action_size (int): Numero di azioni possibili
            hidden_sizes (list): Architettura della rete
            learning_rate (float): Learning rate per l'optimizer
            gamma (float): Discount factor per future rewards
            buffer_capacity (int): Capacità del replay buffer
            device: Device PyTorch (cpu/cuda)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Policy network (quella che viene aggiornata)
        self.policy_net = DQN(state_size, action_size, hidden_sizes).to(self.device)
        
        # Target network (quella usata per calcolare i target Q-values)
        self.target_net = DQN(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net sempre in eval mode
        
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
        """
        Seleziona un'azione usando epsilon-greedy strategy.
        
        Args:
            state (np.ndarray or torch.Tensor): Stato corrente
            epsilon (float): Probabilità di azione random
        
        Returns:
            int: Indice dell'azione selezionata
        """
        # Epsilon-greedy: exploration vs exploitation
        if random.random() < epsilon:
            # Exploration: azione random
            return random.randrange(self.action_size)
        else:
            # Exploitation: migliore azione secondo policy_net
            with torch.no_grad():
                # Converti stato in tensor se necessario
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).to(self.device)
                elif not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                
                # Assicurati che abbia dimensione batch
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                # Calcola Q-values e scegli la migliore azione
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                
                return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Salva una transizione nel replay buffer.
        
        Args:
            state: Stato corrente
            action: Azione eseguita
            reward: Reward ricevuto
            next_state: Prossimo stato
            done: Se l'episodio è terminato
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self, batch_size=64):
        """
        Esegue un optimization step usando un batch dal replay buffer.
        
        Args:
            batch_size (int): Dimensione del batch
        
        Returns:
            float: Loss value (None se il buffer è troppo piccolo)
        """
        # Controlla se c'è abbastanza memoria
        if len(self.memory) < batch_size:
            return None
        
        # Campiona un batch random
        batch = self.memory.sample(batch_size)
        
        # Converti batch in tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Calcola Q(s_t, a) - Q-value dell'azione presa
        # policy_net(state_batch) -> [batch_size, action_size]
        # gather seleziona il Q-value dell'azione specifica
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Calcola V(s_{t+1}) per tutti i next states usando target network
        with torch.no_grad():
            # Calcola max Q-value per il prossimo stato
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            # Se done=True, next_state_value = 0 (episodio terminato)
            next_state_values = next_state_values * (1 - done_batch)
            
            # Calcola target Q-value: r + gamma * max_a' Q(s', a')
            target_values = reward_batch + (self.gamma * next_state_values)
        
        # Calcola loss (Smooth L1 Loss / Huber Loss)
        loss = self.criterion(state_action_values.squeeze(), target_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        self.steps_done += 1
        
        return loss.item()
    
    def update_target_network(self, tau=1.0):
        """
        Aggiorna la target network.
        
        Args:
            tau (float): Soft update parameter
                - tau=1.0: Hard update (copia completa)
                - tau<1.0: Soft update (aggiornamento graduale)
        """
        if tau == 1.0:
            # Hard update: copia completa dei pesi
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # Soft update: aggiornamento graduale
            # θ_target = τ * θ_policy + (1 - τ) * θ_target
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    tau * policy_param.data + (1.0 - tau) * target_param.data
                )
    
    def save(self, filepath):
        """
        Salva i pesi della policy network.
        
        Args:
            filepath (str): Path del file .pth
        """
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salva stato completo
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma
        }
        
        torch.save(checkpoint, filepath)
        print(f"[DQNAgent] Model saved to {filepath}")
    
    def load(self, filepath, load_optimizer=True):
        """
        Carica i pesi della policy network.
        
        Args:
            filepath (str): Path del file .pth
            load_optimizer (bool): Se caricare anche l'optimizer state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Carica network states
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        
        # Carica optimizer se richiesto
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Carica stats
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        
        print(f"[DQNAgent] Model loaded from {filepath}")
        print(f"  Steps: {self.steps_done}, Episodes: {self.episodes_done}")
    
    def set_train_mode(self):
        """Imposta la policy network in training mode."""
        self.policy_net.train()
    
    def set_eval_mode(self):
        """Imposta la policy network in evaluation mode."""
        self.policy_net.eval()
    
    def get_epsilon(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000):
        """
        Calcola epsilon decrescente per epsilon-greedy.
        
        Args:
            epsilon_start (float): Epsilon iniziale
            epsilon_end (float): Epsilon finale
            epsilon_decay (int): Numero di step per il decay
        
        Returns:
            float: Epsilon corrente
        """
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1.0 * self.steps_done / epsilon_decay)
        return epsilon


# Esempio di utilizzo
if __name__ == "__main__":
    # Parametri
    STATE_SIZE = 25
    ACTION_SIZE = 8
    BATCH_SIZE = 64
    
    # Crea agent
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_capacity=100000
    )
    
    print(f"DQN Agent initialized on device: {agent.device}")
    print(f"Policy Network: {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    
    # Simula alcune transizioni
    for i in range(200):
        state = np.random.randn(STATE_SIZE)
        action = agent.select_action(state, epsilon=0.5)
        reward = np.random.randn()
        next_state = np.random.randn(STATE_SIZE)
        done = random.random() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"\nReplay buffer size: {len(agent.memory)}")
    
    # Training step
    loss = agent.optimize_model(batch_size=BATCH_SIZE)
    print(f"Training loss: {loss:.4f}")
    
    # Update target network
    agent.update_target_network()
    print("Target network updated")
    
    # Salva e carica
    agent.save("checkpoints/test_agent.pth")
    agent.load("checkpoints/test_agent.pth")
