"""
Script per Generare Grafici Professionali e Comparabili dal Training Log

Autore: Hollow Knight RL Training System
Data: 2026-01-26

MODIFICHE: Dashboard unificata con metriche comparabili tra PPO e DQN

Usage:
    python generate_plots.py --log checkpoints/training_log.txt --type dqn --output plots_dqn/
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
    ax.set_ylabel("Steps", fontweight="bold")
    ax.set_title(f"{algorithm} Training: Episode Length", fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_episode_length.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    print(f"✓ Grafico salvato: {output_path}")


def plot_loss_unified(df, output_dir, algorithm="DQN", window=20):
    """
    Genera grafico della Loss unificato.
    - DQN: avg_loss
    - PPO: actor_loss e critic_loss (media o separati)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values

    if algorithm.upper() == "DQN":
        if "avg_loss" not in df.columns:
            print("⚠ Colonna 'avg_loss' non trovata nel log DQN")
            return

        losses = df["avg_loss"].values
        smoothed = smooth(losses, window=window)

        ax.plot(
            episodes, losses, alpha=0.3, color="#ff7f0e", linewidth=1, label="Raw Loss"
        )
        ax.plot(
            episodes,
            smoothed,
            color="#2ca02c",
            linewidth=2.5,
            label=f"Smoothed (window={window})",
        )
        title = "Training Loss"

    elif algorithm.upper() == "PPO":
        # PPO può avere actor_loss e critic_loss
        has_actor = "actor_loss" in df.columns
        has_critic = "critic_loss" in df.columns

        if not has_actor and not has_critic:
            print("⚠ Colonne 'actor_loss' o 'critic_loss' non trovate nel log PPO")
            return

        if has_actor and has_critic:
            # Media delle due loss
            actor_loss = df["actor_loss"].values
            critic_loss = df["critic_loss"].values
            combined_loss = (actor_loss + critic_loss) / 2
            smoothed = smooth(combined_loss, window=window)

            ax.plot(
                episodes,
                combined_loss,
                alpha=0.3,
                color="#ff7f0e",
                linewidth=1,
                label="Raw Combined Loss",
            )
            ax.plot(
                episodes,
                smoothed,
                color="#2ca02c",
                linewidth=2.5,
                label=f"Smoothed (window={window})",
            )
            title = "Training Loss (Actor + Critic Average)"

        elif has_actor:
            actor_loss = df["actor_loss"].values
            smoothed = smooth(actor_loss, window=window)
            ax.plot(
                episodes,
                actor_loss,
                alpha=0.3,
                color="#ff7f0e",
                linewidth=1,
                label="Raw Actor Loss",
            )
            ax.plot(
                episodes,
                smoothed,
                color="#2ca02c",
                linewidth=2.5,
                label=f"Smoothed (window={window})",
            )
            title = "Actor Loss"

        else:
            critic_loss = df["critic_loss"].values
            smoothed = smooth(critic_loss, window=window)
            ax.plot(
                episodes,
                critic_loss,
                alpha=0.3,
                color="#ff7f0e",
                linewidth=1,
                label="Raw Critic Loss",
            )
            ax.plot(
                episodes,
                smoothed,
                color="#2ca02c",
                linewidth=2.5,
                label=f"Smoothed (window={window})",
            )
            title = "Critic Loss"

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title(f"{algorithm} Training: {title}", fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_training_loss.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    print(f"✓ Grafico salvato: {output_path}")


def plot_exploration_rate(df, output_dir, algorithm="DQN", window=20):
    """
    Grafico unificato per l'exploration:
    - DQN: epsilon decay
    - PPO: entropy o learning rate decay (se disponibili)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values

    if algorithm.upper() == "DQN":
        if "epsilon" not in df.columns:
            print("⚠ Colonna 'epsilon' non trovata nel log DQN")
            return

        epsilon = df["epsilon"].values
        ax.plot(episodes, epsilon, color="#9467bd", linewidth=2, label="Epsilon")
        ax.fill_between(episodes, 0, epsilon, alpha=0.3, color="#9467bd")
        ax.set_ylabel("Epsilon Value", fontweight="bold")
        title = "Exploration Rate (Epsilon Decay)"
        ax.set_ylim([-0.05, 1.05])

    elif algorithm.upper() == "PPO":
        # PPO potrebbe avere entropy o learning rate
        if "entropy" in df.columns:
            entropy = df["entropy"].values
            smoothed = smooth(entropy, window=window)
            ax.plot(
                episodes,
                entropy,
                alpha=0.3,
                color="#9467bd",
                linewidth=1,
                label="Raw Entropy",
            )
            ax.plot(
                episodes,
                smoothed,
                color="#d62728",
                linewidth=2.5,
                label=f"Smoothed (window={window})",
            )
            ax.set_ylabel("Entropy", fontweight="bold")
            title = "Policy Entropy (Exploration Measure)"

        elif "learning_rate" in df.columns:
            lr = df["learning_rate"].values
            ax.plot(episodes, lr, color="#9467bd", linewidth=2, label="Learning Rate")
            ax.fill_between(episodes, 0, lr, alpha=0.3, color="#9467bd")
            ax.set_ylabel("Learning Rate", fontweight="bold")
            title = "Learning Rate Decay"

        else:
            print("⚠ Colonne 'entropy' o 'learning_rate' non trovate nel log PPO")
            # Creiamo un grafico placeholder vuoto per mantenere la struttura
            ax.text(
                0.5,
                0.5,
                "No Exploration Data Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            title = "Exploration Metric (Not Available)"

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_title(f"{algorithm} Training: {title}", fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_exploration_rate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    print(f"✓ Grafico salvato: {output_path}")


def plot_mantis_lords_killed_unified(df, output_dir, algorithm="DQN", window=20):
    """
    Grafico unificato per Mantis Lords Killed.
    Funziona sia per PPO che per DQN se hanno la colonna mantis_killed.
    """
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


def plot_unified_dashboard(df, output_dir, algorithm="DQN", window=20):
    """
    Dashboard unificata con 4 metriche comparabili tra DQN e PPO:
    1. Cumulative Reward (top-left)
    2. Episode Length (top-right)
    3. Training Loss (bottom-left)
    4. Exploration Rate / Mantis Killed (bottom-right) - dipende dai dati disponibili
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"{algorithm} Training Dashboard", fontsize=18, fontweight="bold", y=0.995
    )

    episodes = df["episode"].values

    # ===== 1. Cumulative Reward (top-left) =====
    ax = axes[0, 0]
    rewards = df["total_reward"].values
    smoothed_rewards = smooth(rewards, window=window)
    ax.plot(episodes, rewards, alpha=0.3, color="#1f77b4", linewidth=1, label="Raw")
    ax.plot(
        episodes, smoothed_rewards, color="#d62728", linewidth=2.5, label="Smoothed"
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title("Cumulative Reward", fontweight="bold")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # ===== 2. Episode Length (top-right) =====
    ax = axes[0, 1]
    if "steps" in df.columns:
        steps = df["steps"].values
        smoothed_steps = smooth(steps, window=window)
        ax.plot(episodes, steps, alpha=0.3, color="#8c564b", linewidth=1, label="Raw")
        ax.plot(
            episodes, smoothed_steps, color="#e377c2", linewidth=2.5, label="Smoothed"
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Steps", fontweight="bold")
        ax.set_title("Episode Length", fontweight="bold")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No Episode Length Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    # ===== 3. Training Loss (bottom-left) =====
    ax = axes[1, 0]

    if algorithm.upper() == "DQN" and "avg_loss" in df.columns:
        losses = df["avg_loss"].values
        smoothed_losses = smooth(losses, window=window)
        ax.plot(episodes, losses, alpha=0.3, color="#ff7f0e", linewidth=1, label="Raw")
        ax.plot(
            episodes, smoothed_losses, color="#2ca02c", linewidth=2.5, label="Smoothed"
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Average Loss", fontweight="bold")
        ax.set_title("Training Loss", fontweight="bold")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)

    elif algorithm.upper() == "PPO":
        has_actor = "actor_loss" in df.columns
        has_critic = "critic_loss" in df.columns

        if has_actor and has_critic:
            actor_loss = df["actor_loss"].values
            critic_loss = df["critic_loss"].values
            combined = (actor_loss + critic_loss) / 2
            smoothed_loss = smooth(combined, window=window)
            ax.plot(
                episodes, combined, alpha=0.3, color="#ff7f0e", linewidth=1, label="Raw"
            )
            ax.plot(
                episodes,
                smoothed_loss,
                color="#2ca02c",
                linewidth=2.5,
                label="Smoothed",
            )
            ax.set_xlabel("Episode", fontweight="bold")
            ax.set_ylabel("Combined Loss", fontweight="bold")
            ax.set_title("Training Loss (Actor+Critic Avg)", fontweight="bold")
            ax.legend(loc="best", framealpha=0.95)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Loss Data Not Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )

    # ===== 4. Exploration Rate o Mantis Killed (bottom-right) =====
    ax = axes[1, 1]

    # Priorità: Mantis Killed > Epsilon > Entropy > Placeholder
    if "mantis_killed" in df.columns:
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
            label="Raw",
        )
        ax.plot(
            episodes, smoothed_mantis, color="#d62728", linewidth=2.5, label="Smoothed"
        )
        ax.axhline(
            y=3,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="Victory (3)",
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
        ax.set_title("Mantis Lords Defeated", fontweight="bold")
        ax.set_ylim([-0.2, 3.5])
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)

    elif "epsilon" in df.columns:
        epsilon = df["epsilon"].values
        ax.plot(episodes, epsilon, color="#9467bd", linewidth=2, label="Epsilon")
        ax.fill_between(episodes, 0, epsilon, alpha=0.3, color="#9467bd")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Epsilon", fontweight="bold")
        ax.set_title("Exploration Rate (Epsilon Decay)", fontweight="bold")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)

    elif "entropy" in df.columns:
        entropy = df["entropy"].values
        smoothed_ent = smooth(entropy, window=window)
        ax.plot(episodes, entropy, alpha=0.3, color="#9467bd", linewidth=1, label="Raw")
        ax.plot(
            episodes, smoothed_ent, color="#d62728", linewidth=2.5, label="Smoothed"
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Entropy", fontweight="bold")
        ax.set_title("Policy Entropy (Exploration)", fontweight="bold")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3)

    else:
        ax.text(
            0.5,
            0.5,
            "No Additional Metric Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    print(f"✓ Dashboard unificata salvata: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici professionali e comparabili da training log"
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
    print(f"Generazione Grafici UNIFICATI per {args.type.upper()} Training")
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
    print(f"Colonne disponibili: {list(df.columns)}\n")

    # Genera grafici individuali con struttura unificata
    print("Generazione grafici individuali...")
    plot_cumulative_reward(df, args.output, algorithm=algorithm, window=args.window)
    plot_episode_length(df, args.output, algorithm=algorithm, window=args.window)
    plot_loss_unified(df, args.output, algorithm=algorithm, window=args.window)
    plot_exploration_rate(df, args.output, algorithm=algorithm, window=args.window)
    plot_mantis_lords_killed_unified(
        df, args.output, algorithm=algorithm, window=args.window
    )

    print("\nGenerazione dashboard unificata...")
    plot_unified_dashboard(df, args.output, algorithm=algorithm, window=args.window)

    print(f"\n{'='*60}")
    print(f"✓ Generazione completata! Grafici salvati in: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
