<p align="center">
  <img src="assets/logo_grande.png" alt="Synthetic Soul" width="400"/>
</p>

<h1 align="center">Synthetic Soul</h1>

<p align="center">
  <b>Un agente AI che impara a battere i Mantis Lords in Hollow Knight tramite Reinforcement Learning</b>
</p>

<p align="center">
  <code>PPO</code> · <code>DQN</code> · <code>Curriculum Learning</code> · <code>Multi-Instance Training</code> · <code>Hall of Fame</code>
</p>

---

## Panoramica

**Synthetic Soul** e' un sistema di Reinforcement Learning che allena agenti AI a sconfiggere il boss fight dei **Mantis Lords** in Hollow Knight. Il progetto si compone di due parti:

- **Mod C# per Unity** — si aggancia al gioco, estrae lo stato in tempo reale (51 features) e invia le azioni dell'AI al controller tramite TCP
- **Pipeline Python di RL** — allena agenti con due algoritmi (PPO e DQN), ciascuno con reward shaping dedicato, curriculum progressivo a 5 fasi e selezione automatica dei campioni

L'agente osserva HP, posizione, velocita', pattern di attacco delle mantidi, ostacoli e terreno, e impara a schivare, attaccare nelle finestre di recovery e gestire fino a 3 mantidi simultaneamente.

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
┌──────────────────────────────▼───────────────────────────────────┐
│                     Python RL Pipeline                           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐   Environment Layer          │
│  │  env_ppo.py  │  │  env_dqn.py  │   (reward shaping,           │
│  │  (denso)     │  │  (sparso)    │    socket I/O, game logic)   │
│  └──────┬───────┘  └──────┬───────┘                              │
│  ┌──────▼───────┐  ┌──────▼───────┐   Training Layer             │
│  │ train_ppo.py │  │ train_dqn.py │   (loop, HoF, multi-inst)    │
│  └──────┬───────┘  └──────┬───────┘                              │
│  ┌──────▼───────┐  ┌──────▼───────┐   Agent Layer                │
│  │  ppo_agent   │  │  dqn_agent   │   (reti neurali,             │
│  │  ActorCritic │  │  DuelingDQN  │    ottimizzazione)           │
│  └──────────────┘  └──────────────┘                              │
│                                                                  │
│  preprocess.py (51 features)  │  generate_plots.py (grafici)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Struttura del Progetto

```
SyntetichSoul/
├── AI_Agents/
│   ├── scripts/
│   │   ├── train_ppo.py            # Orchestratore training PPO
│   │   ├── train_dqn.py            # Orchestratore training DQN
│   │   ├── play.py                 # Script di inferenza
│   │   └── preprocess.py           # Preprocessing stato (v1/v2)
│   └── src/
│       ├── agents/
│       │   ├── ppo_agent.py        # Agente PPO con LSTM e kill buffer
│       │   └── dqn_agent.py        # Agente DQN con Dueling architecture
│       ├── env/
│       │   ├── env_ppo.py          # Environment reward denso
│       │   └── env_dqn.py          # Environment reward sparso
│       ├── models/
│       │   ├── actor_critic.py     # Rete Actor-Critic + LSTM
│       │   └── dqn_net.py          # Rete Dueling DQN
│       ├── utils/
│       │   └── generate_plots.py   # Suite di visualizzazione
│       └── requirements/
│           └── requirements.txt
│
├── HollowKnightMod/
│   ├── SyntheticSoulMod.cs         # Entry point del mod
│   ├── GameStateExtractor.cs       # Estrazione 51 features come JSON
│   ├── MantisAttackDetector.cs     # Riconoscimento pattern attacco
│   ├── ActionExecutor.cs           # Traduzione azioni AI → input controller
│   ├── SocketCommunicator.cs       # Bridge TCP Unity ↔ Python
│   ├── VictoryTracker.cs           # Tracking vittorie e statistiche
│   └── DesktopLogger.cs            # Utility di logging
│
├── assets/                          # Risorse grafiche
└── SyntheticSoul.sln               # Solution Visual Studio
```

---

## Prerequisiti

- **Hollow Knight** con la mod Synthetic Soul installata
- **Python 3.8+**
- **PyTorch**, NumPy, Gymnasium, filelock
- Per i grafici: matplotlib, pandas, scipy, seaborn

```bash
pip install -r AI_Agents/src/requirements/requirements.txt
```

---

## Quick Start

### 1. Training

Avvia Hollow Knight con la mod attiva, poi lancia il training dal terminale:

```bash
# PPO — singola istanza, tutte le 5 fasi
python AI_Agents/scripts/train_ppo.py --ports 5555

# PPO — 3 istanze parallele su porte diverse
python AI_Agents/scripts/train_ppo.py --instances 3 --ports 5555 5556 5557

# DQN — singola istanza
python AI_Agents/scripts/train_dqn.py --ports 5555

# Riprendi da una fase specifica con modello pre-trainato
python AI_Agents/scripts/train_ppo.py --start-phase 3 --pretrained best.pth --ports 5555
```

### 2. Inferenza (Play)

```bash
# Lancia il campione PPO della fase 5
python AI_Agents/scripts/play.py --agent ppo \
    --model training_output_ppo/champion/phase_5_champion.pth

# DQN con logging su CSV
python AI_Agents/scripts/play.py --agent dqn \
    --model training_output_dqn/champion/phase_4_champion.pth \
    --runs 50 --log results.csv

# Modalita' infinita, output ridotto
python AI_Agents/scripts/play.py --agent ppo --model best.pth --runs 0 --quiet
```

### 3. Visualizzazione

```bash
# Singolo algoritmo
python AI_Agents/src/utils/generate_plots.py --mode ppo \
    --ppo-log training_output_ppo/phase_1/training_log_ppo.csv

# Confronto PPO vs DQN
python AI_Agents/src/utils/generate_plots.py --mode compare \
    --ppo-log ppo.csv --dqn-log dqn.csv

# Dashboard per presentazione
python AI_Agents/src/utils/generate_plots.py --mode presentation \
    --ppo-log ppo.csv --dqn-log dqn.csv --output slides/
```

---

## Curriculum Learning — 5 Fasi

Il training segue un curriculum progressivo. L'agente viene promosso automaticamente alla fase successiva quando raggiunge i criteri richiesti su una finestra rolling di episodi.

| Fase | Nome | Obiettivo | Criterio di Promozione |
|:----:|------|-----------|------------------------|
| 1 | **Survival** | Schivare, non morire | Avg survival steps ≥ 850 |
| 2 | **First Hits** | Colpire durante le finestre di recovery | Avg danno inflitto ≥ 250 |
| 3 | **Aggression** | Uccidere la prima mantide | Avg mantis killed ≥ 0.8 |
| 4 | **Dual Mantis** | Gestire due mantidi simultaneamente | Avg mantis killed ≥ 1.8 |
| 5 | **Mastery** | Vittoria completa, ottimizzare tempo e danni subiti | Win rate ≥ 50% |

Ogni fase ha iperparametri dedicati (learning rate, entropy, episodi) che vengono automaticamente configurati.

---

## PPO vs DQN

I due algoritmi condividono la stessa infrastruttura (multi-istanza, curriculum, Hall of Fame) ma differiscono nel reward shaping e nella strategia di training.

### Reward Shaping

| Aspetto | PPO — Denso | DQN — Sparso |
|---------|-------------|--------------|
| Segnale per-step | +0.02 tick + distance tracking + movement quality | +0.01 tick minimo |
| Shaping spaziale | Bonus approccio in recovery, penalita' ritirata sotto attacco | Nessuno |
| Dodge | Bonus streak cumulativo (log-scaled) | Solo evento binario |
| Kill reward | +60 | +100 (deve dominare i Q-values) |
| Vittoria | +30 | +50 |
| Morte | da -2 a -5 | da -3 a -8 |
| No-hit bonus | Continuo per-step in fase 5 | Solo terminale |

### Training Loop

| Aspetto | PPO | DQN |
|---------|-----|-----|
| Policy | On-policy | Off-policy |
| Esplorazione | Entropy coefficient (cosine decay) | ε-greedy (exponential decay) |
| Learning rate | 3e-4 → 5e-5 | 1e-4 → 1e-5 |
| Update | Ogni rollout (256 step), 4-6 epoch | Ogni step da replay buffer |
| Memoria | LSTM + kill buffer | Frame stacking + replay 100K-200K |
| Stabilita' | Clipped surrogate loss | Soft target network (τ = 0.002–0.005) |

### Componenti Condivisi

- **Observation space:** 51 features × 4 frame = 204 dimensioni
- **Action space:** 8 azioni discrete — left, right, up, down, jump, attack, dash, spell
- **Preprocessing v2:** normalizzazione + pattern di attacco one-hot encoded

---

## Spazio delle Osservazioni (51 Features)

Lo stato del gioco viene estratto in tempo reale dalla mod C# e preprocessato in 51 features:

| Gruppo | Features | Descrizione |
|--------|:--------:|-------------|
| Player | 5 | HP, soul, can dash, can attack, grounded |
| Velocita' giocatore | 2 | Velocity X/Y (clipped [-1, 1]) |
| Terreno | 5 | Collisioni terreno vicine |
| Boss posizione | 4 | Posizione relativa X/Y, distanza, direzione |
| Boss stato | 4 | Velocity boss X/Y, HP boss, mantidi uccise |
| Boss intent | 4 | One-hot: idle, dash, drop, throw |
| Ostacoli | 10 | Fino a 2 hazard: posizione relativa, velocita', distanza |
| Pattern attacco | 17 | Pattern primario (8-way one-hot), stato temporale, velocita', mantide secondaria |

---

## Spazio delle Azioni

| ID | Azione | Input |
|:--:|--------|-------|
| 0 | MOVE_LEFT | Sinistra |
| 1 | MOVE_RIGHT | Destra |
| 2 | UP | Su |
| 3 | DOWN | Giu' |
| 4 | JUMP | Salto |
| 5 | ATTACK | Attacco (slash) |
| 6 | DASH | Dash direzionale |
| 7 | SPELL | Spell (es. fireball) |

---

## Sistema Champion e Hall of Fame

Il progetto include un sistema di selezione automatica dei migliori modelli.

### Hall of Fame

- Mantiene i **top-3 modelli per fase** condivisi tra tutte le istanze parallele
- Accesso thread-safe tramite file lock su `shared_state.json`
- Ogni istanza puo' occupare al massimo 1 slot (diversificazione)
- Soglia minima di miglioramento di 0.5 reward per rimpiazzare un modello esistente

### Selezione Champion

1. A fine fase, tutti i candidati (best/latest di ogni istanza + Hall of Fame) vengono valutati
2. Il modello con la reward piu' alta viene selezionato come **champion** della fase
3. I champion vengono salvati in `training_output_ppo/champion/` con metadati JSON

```
training_output_ppo/champion/
├── phase_1_champion.pth        # Modello campione fase 1
├── phase_1_champion.json       # Metadati (reward, instance, timestamp)
├── phase_1_history.json        # Storico selezioni
├── phase_2_champion.pth
├── ...
└── phase_5_champion.pth
```

---

## Struttura Output del Training

```
training_output_ppo/
├── phase_1/
│   ├── shared_state.json           # Stato Hall of Fame
│   ├── training_log_ppo.csv        # Log episodi (tutte le istanze)
│   ├── best_pool/                  # Top-3 modelli condivisi
│   │   ├── hof_ppo_inst0.pth
│   │   └── hof_ppo_inst1.pth
│   └── instance_0/
│       ├── best.pth                # Miglior modello dell'istanza
│       ├── latest.pth              # Ultimo checkpoint
│       └── checkpoint_ep100.pth    # Checkpoint periodici
├── phase_2/ ... phase_5/
└── champion/                       # Campioni selezionati per fase
```

### Formato CSV Log

**PPO:**
```
timestamp, instance_id, phase, episode, reward, steps,
mantis_killed, boss_hp, boss_defeated, entropy, learning_rate, num_updates
```

**DQN:**
```
timestamp, instance_id, phase, episode, reward, steps,
mantis_killed, boss_hp, boss_defeated, epsilon, learning_rate, avg_loss
```

---

## Mod C# — Hollow Knight

La mod gira all'interno di Unity e si interfaccia con il gioco tramite le API di modding di Hollow Knight.

| Componente | Responsabilita' |
|------------|-----------------|
| **SyntheticSoulMod** | Entry point, lifecycle del mod, gestione scene, reset episodi |
| **GameStateExtractor** | Estrae 51+ features interrogando HeroController e PlayMaker FSM |
| **MantisAttackDetector** | Monitora 3 Mantis Lords via FSM; rileva 8 pattern (idle, dash, diagonal, throw, wall, land, wind-up, recovering) |
| **ActionExecutor** | Riceve action ID via TCP e li mappa agli input di HeroController |
| **SocketCommunicator** | Connessione TCP persistente sulla porta 5555 (configurabile) |
| **VictoryTracker** | Traccia vittorie, sconfitte, streak, win rate per sessione |

La mod supporta sia modalita' **training** (invio continuo di stati e ricezione azioni) che **inference** (demo senza aggiornamento pesi).

---

## Tech Stack

| Componente | Tecnologia |
|------------|------------|
| RL Framework | PyTorch |
| Environment | Gymnasium |
| Comunicazione | TCP Sockets (JSON) |
| Game | Hollow Knight (Unity, C#) |
| Mod Framework | Hollow Knight Modding API (.NET 4.7.2) |
| Concorrenza | filelock (multi-istanza) |
| Visualizzazione | matplotlib, seaborn, scipy |

---

## Autori

Progetto sviluppato come parte del corso di **Fondamenti di Intelligenza Artificiale**.
