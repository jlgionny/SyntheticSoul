# Synthetic Soul — Hollow Knight Mantis Lords RL Bot

AI agent che impara a battere i **Mantis Lords** in Hollow Knight tramite Reinforcement Learning. Supporta due algoritmi (**PPO** e **DQN**) con reward shaping dedicato, training multi-istanza parallelo e curriculum progressivo a 5 fasi.

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hollow Knight (Unity)                        │
│  GameStateExtractor ──► SocketCommunicator ◄── ActionExecutor   │
│  MantisAttackDetector        │ TCP                              │
│  VictoryTracker              │ localhost:5555                   │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Python RL Pipeline                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐   Environment Layer         │
│  │  env_ppo.py  │  │  env_dqn.py  │   (reward shaping,          │
│  │  (dense)     │  │  (sparse)    │    socket I/O, game logic)  │
│  └──────┬───────┘  └──────┬───────┘                             │
│  ┌──────▼───────┐  ┌──────▼───────┐   Training Layer            │
│  │ train_ppo.py │  │ train_dqn.py │   (loop, HoF, multi-inst)   │
│  └──────┬───────┘  └──────┬───────┘                             │
│  ┌──────▼───────┐  ┌──────▼───────┐   Agent Layer               │
│  │  ppo_agent   │  │  dqn_agent   │   (reti neurali,            │
│  │  ActorCritic │  │  DuelingDQN  │    ottimizzazione)          │
│  └──────────────┘  └──────────────┘                             │
│                                                                 │
│  preprocess.py (51 features)  │  generate_plots.py (grafici)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## File

### Python

| File | Ruolo |
|---|---|
| `train_ppo.py` | Training PPO (multi-istanza, multi-fase, Hall of Fame) |
| `train_dqn.py` | Training DQN (stessa infrastruttura, logica DQN) |
| `env_ppo.py` | Environment con reward **denso** ottimizzato per PPO/GAE |
| `env_dqn.py` | Environment con reward **sparso** ottimizzato per DQN/replay |
| `ppo_agent.py` | Agente PPO con LSTM, GAE(λ), kill buffer replay |
| `dqn_agent.py` | Agente DQN con Dueling architecture, experience replay |
| `actor_critic.py` | Rete Actor-Critic (usata da PPO) |
| `dqn_net.py` | Rete Dueling DQN |
| `preprocess.py` | Preprocessing stato (v1: 34 features, v2: 51 features) |
| `generate_plots.py` | Visualizzazione training per presentazioni |

### C# — Mod Unity

| File | Ruolo |
|---|---|
| `SyntheticSoulMod.cs` | Entry point del mod |
| `GameStateExtractor.cs` | Estrae stato completo come JSON |
| `MantisAttackDetector.cs` | Rileva pattern attacco mantidi (dash, drop, throw) |
| `ActionExecutor.cs` | Esegue azioni AI come input controller |
| `SocketCommunicator.cs` | Bridge TCP tra Unity e Python |
| `VictoryTracker.cs` | Statistiche vittorie e streak |
| `DesktopLogger.cs` | Utility logging |

---

## PPO vs DQN — Differenze

### Reward Shaping

| | PPO (`env_ppo.py`) — Denso | DQN (`env_dqn.py`) — Sparso |
|---|---|---|
| Segnale per-step | +0.02 tick, distance tracking, movement quality | +0.01 tick minimo |
| Shaping spaziale | Approccio in recovery, ritirata sotto attacco | Nessuno (DQN va in loop) |
| Dodge tracking | Bonus streak cumulativo (log-scaled) | Solo evento binario |
| Kill reward | +60 (denso accumula) | +100 (deve dominare Q-values) |
| Vittoria | +30 | +50 |
| Morte | da -2 a -5 | da -3 a -8 |
| No-hit bonus | Continuo per-step in mastery | Solo terminale |

### Training Loop

| | PPO (`train_ppo.py`) | DQN (`train_dqn.py`) |
|---|---|---|
| Policy | On-policy | Off-policy |
| Esplorazione | Entropy coefficient (cosine decay) | ε-greedy (exponential decay) |
| Learning rate | 3e-4 → 5e-5 | 1e-4 → 1e-5 |
| Update | Ogni 256-step rollout, 4-6 epoch | Ogni step da replay buffer |
| Memoria | LSTM + kill buffer | Frame stacking + replay 100K-200K |
| Stabilità | Clipped surrogate | Soft target network (τ=0.002-0.005) |

### Identico tra PPO e DQN

- **Observation space:** 51 features × 4 frame = 204 dimensioni
- **Action space:** 8 azioni discrete (left, right, up, down, jump, attack, dash, spell)
- **5 fasi curriculum:** Survival → First Hits → Aggression → Dual Mantis → Mastery
- **Hall of Fame:** Top-3 modelli condivisi tra istanze parallele
- **Promozione automatica:** Avanzamento fase basato su metriche rolling

---

## Quick Start

### Prerequisiti

- Hollow Knight con mod Synthetic Soul installato
- Python 3.8+ con PyTorch, NumPy, filelock
- Per i grafici: matplotlib, pandas, scipy

### Training

```bash
# PPO — singola istanza, pipeline completa fasi 1→5
python train_ppo.py --ports 5555

# PPO — 3 istanze parallele
python train_ppo.py --instances 3 --ports 5555 5556 5557

# DQN — singola istanza, solo fase 1
python train_dqn.py --phase 1 --ports 5555

# DQN — riprendi da fase 3 con modello pre-trainato
python train_dqn.py --start-phase 3 --pretrained best.pth

# Override numero episodi per fase
python train_ppo.py --episodes 200 --ports 5555
```

### Visualizzazione

```bash
# Singolo algoritmo
python generate_plots.py --mode ppo \
    --ppo-log training_output_ppo/phase_1/training_log_ppo.csv

python generate_plots.py --mode dqn \
    --dqn-log training_output_dqn/phase_1/training_log_dqn.csv

# Confronto PPO vs DQN
python generate_plots.py --mode compare \
    --dqn-log training_output_dqn/phase_3/training_log_dqn.csv \
    --ppo-log training_output_ppo/phase_3/training_log_ppo.csv

# Analisi multi-istanza
python generate_plots.py --mode multi \
    --multi-log training_output_ppo/phase_1/training_log_ppo.csv

# Dashboard per presentazione
python generate_plots.py --mode presentation \
    --dqn-log dqn.csv --ppo-log ppo.csv --output slides/
```

---

## Fasi di Training

| Fase | Nome | Obiettivo | Criterio Promozione |
|---|---|---|---|
| 1 | Survival | Schivare attacchi, non morire | Avg survival ≥ 700 step |
| 2 | First Hits | Colpire durante recovery windows | Avg danno inflitto ≥ 200 |
| 3 | Aggression | Uccidere la prima mantide | Avg kills ≥ 0.5 |
| 4 | Dual Mantis | Gestire due mantidi insieme | Avg kills ≥ 2.0 |
| 5 | Mastery | Vittoria completa, speed + no-hit | Win rate ≥ 50% |

---

## Struttura Output

```
training_output_ppo/
├── phase_1/
│   ├── shared_state.json
│   ├── shared_state.lock
│   ├── training_log_ppo.csv
│   ├── best_pool/
│   │   ├── hof_ppo_inst0.pth
│   │   └── hof_ppo_inst1.pth
│   ├── instance_0/
│   │   ├── best.pth
│   │   ├── latest.pth
│   │   └── checkpoint_ep100.pth
│   └── instance_1/
│       └── ...
├── phase_2/
│   └── ...
└── ...
```

## CSV Log Format

**PPO** (`training_log_ppo.csv`):
```
timestamp, instance_id, phase, episode, reward, steps,
mantis_killed, boss_hp, boss_defeated, entropy, learning_rate, num_updates
```

**DQN** (`training_log_dqn.csv`):
```
timestamp, instance_id, phase, episode, reward, steps,
mantis_killed, boss_hp, boss_defeated, epsilon, learning_rate, avg_loss
```