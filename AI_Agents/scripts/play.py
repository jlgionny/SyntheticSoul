"""
═══════════════════════════════════════════════════════════════════════
  INFERENCE — Mantis Lords (PPO + DQN)
  Esegui l'agente allenato senza training

  USAGE:
    # PPO
    python play.py --agent ppo --model training_output_ppo/champion/phase_4_champion.pth
    python play.py --agent ppo --model best.pth --runs 50 --log results_ppo.csv

    # DQN
    python play.py --agent dqn --model training_output_dqn/champion/phase_3_champion.pth
    python play.py --agent dqn --model best.pth --runs 0   # infinite

    # Opzioni comuni
    python play.py --agent ppo --model best.pth --quiet           # solo riepilogo
    python play.py --agent dqn --model best.pth --log test.csv    # salva CSV
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
from collections import deque
from datetime import datetime

# ═══ Setup paths ═══
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(SRC_DIR, "agents"))
sys.path.insert(0, os.path.join(SRC_DIR, "env"))
sys.path.insert(0, os.path.join(SRC_DIR, "models"))
sys.path.insert(0, SCRIPT_DIR)

from preprocess import preprocess_state_v2, STATE_DIM_V2


# ═══════════════════════════════════════════════════════════════
# FRAME STACKER
# ═══════════════════════════════════════════════════════════════

class FrameStacker:
    def __init__(self, stack_size: int, state_dim: int):
        self.stack_size = stack_size
        self.state_dim = state_dim
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_state)
        return np.concatenate(list(self.frames))

    def step(self, state: np.ndarray) -> np.ndarray:
        self.frames.append(state)
        return np.concatenate(list(self.frames))


# ═══════════════════════════════════════════════════════════════
# AGENT LOADERS
# ═══════════════════════════════════════════════════════════════

def load_ppo_agent(model_path: str, stacked_dim: int):
    """Carica un agente PPO in modalità inference."""
    from ppo_agent import PPOAgent
    agent = PPOAgent(
        state_size=stacked_dim,
        action_size=8,
        learning_rate=1e-5,
        gamma=0.995,
        gae_lambda=0.95,
        entropy_coef=0.0,
        use_lstm=False,
        n_epochs=1,
        batch_size=64,
    )
    agent.load(model_path)
    agent.policy.eval()
    return agent


def load_dqn_agent(model_path: str, stacked_dim: int):
    """Carica un agente DQN in modalità inference."""
    from dqn_agent import DQNAgent
    agent = DQNAgent(
        state_size=stacked_dim,
        action_size=8,
        hidden_sizes=[128, 256, 128],
        learning_rate=1e-5,
        gamma=0.995,
        buffer_capacity=1000,  # Minimo, non verrà usato
    )
    agent.load(model_path)
    agent.policy_net.eval()
    return agent


# ═══════════════════════════════════════════════════════════════
# GREEDY ACTION SELECTION
# ═══════════════════════════════════════════════════════════════

def select_action_ppo_greedy(agent, state: np.ndarray) -> int:
    """PPO: prendi l'azione con probabilità massima dalla policy."""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        if agent.use_lstm:
            action_probs, _, _ = agent.policy(state_tensor, agent.hidden_state)
        else:
            action_probs, _ = agent.policy(state_tensor)
        return torch.argmax(action_probs, dim=-1).item()


def select_action_dqn_greedy(agent, state: np.ndarray) -> int:
    """DQN: prendi l'azione con Q-value massimo."""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor)
        return torch.argmax(q_values, dim=-1).item()


# ═══════════════════════════════════════════════════════════════
# ENV LOADER
# ═══════════════════════════════════════════════════════════════

def load_env(agent_type: str, port: int):
    """Carica l'environment corretto per il tipo di agente."""
    if agent_type == "ppo":
        from env_ppo import HollowKnightEnvPPO
        return HollowKnightEnvPPO(host="localhost", port=port, phase=4, reward_scale=5.0)
    else:
        from env_dqn import HollowKnightEnvDQN
        return HollowKnightEnvDQN(host="localhost", port=port, phase=4, reward_scale=5.0)


# ═══════════════════════════════════════════════════════════════
# PLAY LOOP
# ═══════════════════════════════════════════════════════════════

def play(agent_type: str, model_path: str, port: int, num_runs: int,
         max_steps: int, quiet: bool, log_path: str):

    STACK_SIZE = 4
    raw_dim = STATE_DIM_V2
    stacked_dim = raw_dim * STACK_SIZE
    stacker = FrameStacker(STACK_SIZE, raw_dim)

    # ═══ Carica agente ═══
    if not os.path.exists(model_path):
        print(f"[ERRORE] Modello non trovato: {model_path}")
        return

    if agent_type == "ppo":
        agent = load_ppo_agent(model_path, stacked_dim)
        select_action = select_action_ppo_greedy
        reset_hidden = lambda: agent.reset_hidden()
    else:
        agent = load_dqn_agent(model_path, stacked_dim)
        select_action = select_action_dqn_greedy
        reset_hidden = lambda: None  # DQN non ha hidden state

    agent_label = agent_type.upper()
    print(f"\n{'═'*60}")
    print(f"  {agent_label} INFERENCE — Mantis Lords")
    print(f"  Modello: {model_path}")
    print(f"  Porta: {port}")
    print(f"  Run: {'infinite (Ctrl+C)' if num_runs == 0 else num_runs}")
    print(f"  Log CSV: {log_path}")
    print(f"{'═'*60}\n")

    # ═══ Connetti al gioco ═══
    env = load_env(agent_type, port)

    # ═══ CSV Log ═══
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow([
            'run', 'timestamp', 'agent', 'result', 'mantis_killed',
            'boss_hp', 'player_hp', 'steps', 'duration_sec', 'model'
        ])

    # ═══ Statistiche ═══
    wins = 0
    losses = 0
    total_steps_list = []
    kills_list = []
    win_steps = []

    run = 0
    infinite = (num_runs == 0)

    try:
        while infinite or run < num_runs:
            run += 1

            raw_state_dict = env.reset()
            raw_state = preprocess_state_v2(raw_state_dict)
            state = stacker.reset(raw_state)
            reset_hidden()

            ep_start = time.time()

            for step in range(max_steps):
                action = select_action(agent, state)
                next_state_dict, _, done, info = env.step(action)

                next_raw = preprocess_state_v2(next_state_dict)
                state = stacker.step(next_raw)

                if done:
                    break

            # ═══ Risultato ═══
            elapsed = time.time() - ep_start
            mantis_killed = next_state_dict.get("mantisLordsKilled", 0)
            boss_defeated = next_state_dict.get("bossDefeated", False)
            boss_hp = next_state_dict.get("bossHealth", 0)
            player_hp = next_state_dict.get("playerHealth", 0)

            kills_list.append(mantis_killed)
            total_steps_list.append(step + 1)

            if boss_defeated:
                wins += 1
                win_steps.append(step + 1)
                marker = "★ VITTORIA ★"
                result = "WIN"
            else:
                losses += 1
                marker = "✗ Sconfitta"
                result = "LOSS"

            # ═══ Scrivi CSV ═══
            csv_writer.writerow([
                run,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                agent_label,
                result,
                mantis_killed,
                f"{boss_hp:.0f}",
                player_hp,
                step + 1,
                f"{elapsed:.2f}",
                os.path.basename(model_path),
            ])
            log_file.flush()

            if not quiet:
                print(
                    f"  Run {run:>4} | {marker} | "
                    f"Kill={mantis_killed} | Boss HP={boss_hp:.0f} | "
                    f"Player HP={player_hp} | Steps={step+1} | "
                    f"Tempo={elapsed:.1f}s"
                )

            # Riepilogo periodico ogni 10 run
            if run % 10 == 0:
                wr = wins / run * 100
                avg_kills = np.mean(kills_list)
                avg_steps = np.mean(total_steps_list)
                print(f"\n  {'─'*50}")
                print(f"  [{agent_label}] Dopo {run} run: {wins}W / {losses}L ({wr:.1f}% win rate)")
                print(f"  Kill medi: {avg_kills:.2f} | Steps medi: {avg_steps:.0f}")
                if win_steps:
                    print(f"  Vittorie — steps medio: {np.mean(win_steps):.0f}, "
                          f"migliore: {min(win_steps)}")
                print(f"  {'─'*50}\n")

    except KeyboardInterrupt:
        print(f"\n\n  Interrotto dall'utente dopo {run} run.")

    finally:
        env.close()
        log_file.close()
        print(f"\n  Log salvato in: {log_path}")

        # ═══ Riepilogo finale ═══
        print(f"\n{'═'*60}")
        print(f"  RIEPILOGO FINALE [{agent_label}] — {run} run")
        print(f"{'═'*60}")

        if run > 0:
            wr = wins / run * 100
            avg_kills = np.mean(kills_list)
            avg_steps = np.mean(total_steps_list)

            print(f"  Vittorie:    {wins}/{run} ({wr:.1f}%)")
            print(f"  Kill medi:   {avg_kills:.2f}")
            print(f"  Steps medi:  {avg_steps:.0f}")

            if kills_list:
                k0 = sum(1 for k in kills_list if k == 0)
                k1 = sum(1 for k in kills_list if k == 1)
                k2 = sum(1 for k in kills_list if k == 2)
                k3 = sum(1 for k in kills_list if k >= 3)
                print(f"\n  Distribuzione kill:")
                print(f"    0 kill:  {k0} ({k0/run*100:.1f}%)")
                print(f"    1 kill:  {k1} ({k1/run*100:.1f}%)")
                print(f"    2 kill:  {k2} ({k2/run*100:.1f}%)")
                print(f"    3 kill:  {k3} ({k3/run*100:.1f}%)")

            if win_steps:
                print(f"\n  Vittorie:")
                print(f"    Steps medio: {np.mean(win_steps):.0f}")
                print(f"    Più veloce:  {min(win_steps)} steps")
                print(f"    Più lenta:   {max(win_steps)} steps")

        print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gioca con l'agente PPO o DQN allenato (nessun training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python play.py --agent ppo --model training_output_ppo/champion/phase_4_champion.pth
  python play.py --agent dqn --model training_output_dqn/champion/phase_3_champion.pth
  python play.py --agent ppo --model best.pth --runs 50 --log risultati.csv
  python play.py --agent dqn --model best.pth --runs 0                  # infinite
  python play.py --agent ppo --model best.pth --quiet                   # solo riepilogo
        """,
    )

    parser.add_argument("--agent", type=str, required=True,
                        choices=["ppo", "dqn"],
                        help="Tipo di agente: 'ppo' o 'dqn'")
    parser.add_argument("--model", type=str, required=True,
                        help="Percorso al file .pth del modello allenato")
    parser.add_argument("--port", type=int, default=5555,
                        help="Porta del gioco (default: 5555)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Numero di partite (0 = infinite, Ctrl+C per fermare)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Max steps per partita (default: 5000)")
    parser.add_argument("--quiet", action="store_true",
                        help="Mostra solo riepilogo ogni 10 run")
    parser.add_argument("--log", type=str, default="play_log.csv",
                        help="Percorso del file CSV di output (default: play_log.csv)")

    args = parser.parse_args()

    play(
        agent_type=args.agent,
        model_path=args.model,
        port=args.port,
        num_runs=args.runs,
        max_steps=args.max_steps,
        quiet=args.quiet,
        log_path=args.log,
    )