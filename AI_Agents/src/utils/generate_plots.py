r"""
Script per Generare Grafici Professionali dal Training Log
Autore: Hollow Knight RL Training System
Data: 2026-01-26

Usage:
    python AI_Agents\generate_plots.py --log checkpoints\training_log.txt --type dqn --output plots_dqn\
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# Configurazione stile professionale
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16


def smooth(data, window=20):
    """Applica smoothing con media mobile."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode="nearest")


def load_dqn_log(log_file):
    """Carica il log di training DQN."""
    df = pd.read_csv(log_file)
    return df


def load_ppo_log(log_file):
    """Carica il log di training PPO."""
    df = pd.read_csv(log_file)
    return df


def plot_cumulative_reward(df, output_dir, algorithm="DQN", window=20):
    """Genera grafico del Cumulative Reward con smoothing."""
    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    rewards = df["total_reward"].values
    smoothed = smooth(rewards, window=window)

    # Linea puntuale (trasparente)
    ax.plot(
        episodes, rewards, alpha=0.3, color="#1f77b4", linewidth=1, label="Raw Reward"
    )

    # Linea smoothed (più evidente)
    ax.plot(
        episodes,
        smoothed,
        color="#d62728",
        linewidth=2.5,
        label=f"Smoothed (window={window})",
    )

    # Linea di riferimento a 0
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Cumulative Reward per Episode",
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_cumulative_reward.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_loss(df, output_dir, algorithm="DQN", window=20):
    """Genera grafico dell'Average Loss con smoothing (solo per DQN)."""
    if "avg_loss" not in df.columns:
        print(f"⚠ Colonna 'avg_loss' non trovata nel log {algorithm}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    losses = df["avg_loss"].values
    smoothed = smooth(losses, window=window)

    ax.plot(episodes, losses, alpha=0.3, color="#ff7f0e", linewidth=1, label="Raw Loss")
    ax.plot(
        episodes,
        smoothed,
        color="#2ca02c",
        linewidth=2.5,
        label=f"Smoothed (window={window})",
    )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Average Loss", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Average Loss per Episode", fontweight="bold", pad=20
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_average_loss.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_epsilon_decay(df, output_dir, algorithm="DQN"):
    """Genera grafico dell'Epsilon Decay (solo per DQN)."""
    if "epsilon" not in df.columns:
        print(f"⚠ Colonna 'epsilon' non trovata nel log {algorithm}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    epsilon = df["epsilon"].values

    ax.plot(episodes, epsilon, color="#9467bd", linewidth=2, label="Epsilon")
    ax.fill_between(episodes, 0, epsilon, alpha=0.3, color="#9467bd")

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Epsilon Value", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Epsilon Decay (Exploration Rate)",
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_epsilon_decay.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_episode_length(df, output_dir, algorithm="DQN", window=20):
    """Genera grafico dell'Episode Length con smoothing."""
    if "steps" not in df.columns:
        print(f"⚠ Colonna 'steps' non trovata nel log {algorithm}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    steps = df["steps"].values
    smoothed = smooth(steps, window=window)

    ax.plot(episodes, steps, alpha=0.3, color="#8c564b", linewidth=1, label="Raw Steps")
    ax.plot(
        episodes,
        smoothed,
        color="#e377c2",
        linewidth=2.5,
        label=f"Smoothed (window={window})",
    )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Episode Length (steps)", fontweight="bold")
    ax.set_title(f"{algorithm} Training: Episode Length", fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_episode_length.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_mantis_lords_killed(df, output_dir, algorithm="PPO", window=20):
    """Genera grafico dei Mantis Lords uccisi (solo per PPO)."""
    if "mantis_killed" not in df.columns:
        print(f"⚠ Colonna 'mantis_killed' non trovata nel log {algorithm}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    mantis = df["mantis_killed"].values
    smoothed = smooth(mantis, window=window)

    ax.plot(
        episodes,
        mantis,
        alpha=0.4,
        color="#17becf",
        linewidth=1,
        label="Raw Count",
        marker="o",
        markersize=3,
    )
    ax.plot(
        episodes,
        smoothed,
        color="#d62728",
        linewidth=2.5,
        label=f"Smoothed (window={window})",
    )

    # Linee di riferimento
    ax.axhline(
        y=1, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="1 Mantis"
    )
    ax.axhline(
        y=2, color="green", linestyle="--", linewidth=1, alpha=0.6, label="2 Mantis"
    )
    ax.axhline(
        y=3,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="3 Mantis (Victory)",
    )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Mantis Lords Defeated per Episode",
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 3.5])

    plt.tight_layout()
    output_path = os.path.join(
        output_dir, f"{algorithm.lower()}_mantis_lords_killed.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_distance_vs_reward(df, output_dir, algorithm="DQN"):
    """
    Visualizzazione avanzata: Relazione tra distanza dal boss e reward.
    Mostra come l'agente impara il posizionamento ottimale.
    """
    if "distance_to_boss" not in df.columns or "total_reward" not in df.columns:
        print(
            f"⚠ Colonne 'distance_to_boss' o 'total_reward' non trovate nel log {algorithm}"
        )
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    distance = df["distance_to_boss"].values
    rewards = df["total_reward"].values
    episodes = df["episode"].values

    # Scatter plot con gradient di colore basato sull'episodio
    scatter = ax.scatter(
        distance,
        rewards,
        c=episodes,
        cmap="viridis",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    # Colorbar per indicare progressione episodi
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Episode", fontweight="bold")

    # Zona ottimale (2-6 per DQN, 5-8 per PPO)
    if algorithm.upper() == "DQN":
        ax.axvspan(2, 6, alpha=0.2, color="green", label="Optimal Range [2, 6]")
    else:
        ax.axvspan(5, 8, alpha=0.2, color="green", label="Optimal Range [5, 8]")

    ax.set_xlabel("Average Distance to Boss", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Distance vs Reward Relationship",
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(
        output_dir, f"{algorithm.lower()}_distance_vs_reward.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Grafico salvato: {output_path}")


def plot_multi_metrics_dashboard(df, output_dir, algorithm="DQN", window=20):
    """Genera un dashboard con 4 metriche principali."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"{algorithm} Training Dashboard", fontsize=18, fontweight="bold", y=0.995
    )

    episodes = df["episode"].values

    # 1. Cumulative Reward
    ax = axes[0, 0]
    rewards = df["total_reward"].values
    smoothed_rewards = smooth(rewards, window=window)
    ax.plot(episodes, rewards, alpha=0.3, color="#1f77b4", linewidth=1)
    ax.plot(episodes, smoothed_rewards, color="#d62728", linewidth=2.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title("Cumulative Reward", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 2. Episode Length
    ax = axes[0, 1]
    if "steps" in df.columns:
        steps = df["steps"].values
        smoothed_steps = smooth(steps, window=window)
        ax.plot(episodes, steps, alpha=0.3, color="#8c564b", linewidth=1)
        ax.plot(episodes, smoothed_steps, color="#e377c2", linewidth=2.5)
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Steps", fontweight="bold")
        ax.set_title("Episode Length", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # 3. Loss (DQN) o Mantis Killed (PPO)
    ax = axes[1, 0]
    if algorithm.upper() == "DQN" and "avg_loss" in df.columns:
        losses = df["avg_loss"].values
        smoothed_losses = smooth(losses, window=window)
        ax.plot(episodes, losses, alpha=0.3, color="#ff7f0e", linewidth=1)
        ax.plot(episodes, smoothed_losses, color="#2ca02c", linewidth=2.5)
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Average Loss", fontweight="bold")
        ax.set_title("Training Loss", fontweight="bold")
        ax.grid(True, alpha=0.3)
    elif algorithm.upper() == "PPO" and "mantis_killed" in df.columns:
        mantis = df["mantis_killed"].values
        smoothed_mantis = smooth(mantis, window=window)
        ax.plot(
            episodes,
            mantis,
            alpha=0.4,
            color="#17becf",
            linewidth=1,
            marker="o",
            markersize=2,
        )
        ax.plot(episodes, smoothed_mantis, color="#d62728", linewidth=2.5)
        ax.axhline(y=3, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
        ax.set_title("Mantis Lords Defeated", fontweight="bold")
        ax.set_ylim([-0.2, 3.5])
        ax.grid(True, alpha=0.3)

    # 4. Epsilon (DQN) o Distance (PPO/DQN)
    ax = axes[1, 1]
    if algorithm.upper() == "DQN" and "epsilon" in df.columns:
        epsilon = df["epsilon"].values
        ax.plot(episodes, epsilon, color="#9467bd", linewidth=2)
        ax.fill_between(episodes, 0, epsilon, alpha=0.3, color="#9467bd")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Epsilon", fontweight="bold")
        ax.set_title("Exploration Rate (Epsilon Decay)", fontweight="bold")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
    elif "distance_to_boss" in df.columns:
        distance = df["distance_to_boss"].values
        smoothed_dist = smooth(distance, window=window)
        ax.plot(episodes, distance, alpha=0.3, color="#bcbd22", linewidth=1)
        ax.plot(episodes, smoothed_dist, color="#17becf", linewidth=2.5)
        if algorithm.upper() == "DQN":
            ax.axhspan(2, 6, alpha=0.2, color="green")
        else:
            ax.axhspan(5, 8, alpha=0.2, color="green")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Distance to Boss", fontweight="bold")
        ax.set_title("Average Distance to Boss", fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Dashboard salvato: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici professionali da training log"
    )
    parser.add_argument(
        "--log", type=str, required=True, help="Path al file training_log.txt"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["dqn", "ppo"],
        required=True,
        help="Tipo di algoritmo (dqn o ppo)",
    )
    parser.add_argument(
        "--output", type=str, default="plots", help="Directory di output per i grafici"
    )
    parser.add_argument(
        "--window", type=int, default=20, help="Finestra per smoothing (default: 20)"
    )

    args = parser.parse_args()

    # Crea directory output
    os.makedirs(args.output, exist_ok=True)

    # Carica log
    print(f"\n{'='*60}")
    print(f"Generazione Grafici per {args.type.upper()} Training")
    print(f"{'='*60}")
    print(f"Log file: {args.log}")
    print(f"Output dir: {args.output}")
    print(f"Smoothing window: {args.window}\n")

    if args.type == "dqn":
        df = load_dqn_log(args.log)
        algorithm = "DQN"
    else:
        df = load_ppo_log(args.log)
        algorithm = "PPO"

    print(f"✓ Log caricato: {len(df)} episodi\n")

    # Genera grafici
    print("Generazione grafici individuali...")
    plot_cumulative_reward(df, args.output, algorithm=algorithm, window=args.window)
    plot_episode_length(df, args.output, algorithm=algorithm, window=args.window)

    if algorithm == "DQN":
        plot_loss(df, args.output, algorithm=algorithm, window=args.window)
        plot_epsilon_decay(df, args.output, algorithm=algorithm)
    else:
        plot_mantis_lords_killed(
            df, args.output, algorithm=algorithm, window=args.window
        )

    plot_distance_vs_reward(df, args.output, algorithm=algorithm)

    print("\nGenerazione dashboard...")
    plot_multi_metrics_dashboard(
        df, args.output, algorithm=algorithm, window=args.window
    )

    print(f"\n{'='*60}")
    print(f"✓ Generazione completata! Grafici salvati in: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
