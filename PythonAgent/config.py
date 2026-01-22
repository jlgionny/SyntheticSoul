"""Configuration for DQN training"""

# Network settings
HOST = '127.0.0.1'
PORT = 5555

# DQN hyperparameters
STATE_SIZE = 18  # Updated to include hasDoubleJump
ACTION_SIZE = 9  # Number of possible actions
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128

# Training parameters
LEARNING_RATE = 0.0001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Memory and batch
MEMORY_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10  # Update target network every N episodes

# Training
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 2000
SAVE_FREQUENCY = 50  # Save model every N episodes

# Rewards
REWARD_BOSS_DAMAGE = 20.0
REWARD_PLAYER_DAMAGE = -25.0
REWARD_BOSS_DEFEATED = 1000.0
REWARD_PLAYER_DIED = -500.0
REWARD_DISTANCE_CLOSE = 0.5
REWARD_TIME_PENALTY = -0.1

# Actions mapping
ACTIONS = [
    'IDLE',
    'MOVE_LEFT',
    'MOVE_RIGHT',
    'JUMP',
    'ATTACK',
    'DASH',
    'SPELL_UP',
    'SPELL_DOWN',
    'SPELL_SIDE'
]

# Paths
MODEL_SAVE_PATH = 'models/dqn_model.pth'
LOG_PATH = 'logs/training.log'
