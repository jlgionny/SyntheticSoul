r"""
Script Unificato per Generare Grafici di Training
Supporta: DQN solo, PPO solo, o Confronto DQN vs PPO

Usage:
    # Solo DQN
    python AI_Agents\src\utils\generate_plots.py --mode dqn --dqn-log AI_Agents\checkpoints\training_log.txt --output AI_Agents\plots_dqn\episode_100 --window 20

    # Solo PPO
    python AI_Agents\src\utils\generate_plots.py --mode ppo --ppo-log AI_Agents\checkpoints_ppo_mantis\training_log.txt --output AI_Agents\plots_ppo\episode_100 --window 20
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
import numpy as np

# Configurazione stile
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 11


def smooth(data, window=20):
    """Applica smoothing con media mobile."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode="nearest")


def load_log(log_file):
    """Carica il log di training."""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"File non trovato: {log_file}")
    df = pd.read_csv(log_file)
    return df


# ============================================================================
# GRAFICI SINGOLO ALGORITMO
# ============================================================================


def plot_single_reward(df, output_dir, algorithm="DQN", window=20):
    """Grafico Cumulative Reward per singolo algoritmo."""
    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    rewards = df["total_reward"].values
    smoothed = smooth(rewards, window=window)

    color_raw = "#1f77b4" if algorithm == "PPO" else "#ff7f0e"
    color_smooth = "#d62728" if algorithm == "PPO" else "#2ca02c"

    ax.plot(
        episodes, rewards, alpha=0.3, color=color_raw, linewidth=1, label="Raw Reward"
    )
    ax.plot(
        episodes,
        smoothed,
        color=color_smooth,
        linewidth=2.5,
        label=f"Smoothed (w={window})",
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Cumulative Reward", fontweight="bold", fontsize=16
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_reward.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


def plot_single_episode_length(df, output_dir, algorithm="DQN", window=20):
    """Grafico Episode Length per singolo algoritmo."""
    if "steps" not in df.columns:
        print(f"⚠ Colonna 'steps' non trovata per {algorithm}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    steps = df["steps"].values
    smoothed = smooth(steps, window=window)

    color_raw = "#8c564b" if algorithm == "PPO" else "#bcbd22"
    color_smooth = "#e377c2" if algorithm == "PPO" else "#17becf"

    ax.plot(episodes, steps, alpha=0.3, color=color_raw, linewidth=1, label="Raw Steps")
    ax.plot(
        episodes,
        smoothed,
        color=color_smooth,
        linewidth=2.5,
        label=f"Smoothed (w={window})",
    )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Steps", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Episode Length", fontweight="bold", fontsize=16
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_episode_length.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


def plot_single_mantis(df, output_dir, algorithm="DQN", window=20):
    """Grafico Mantis Lords Killed per singolo algoritmo."""
    if "mantis_killed" not in df.columns:
        print(f"⚠ Colonna 'mantis_killed' non trovata per {algorithm}")
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
        marker="o",
        markersize=2,
        label="Raw",
    )
    ax.plot(
        episodes,
        smoothed,
        color="#d62728",
        linewidth=2.5,
        label=f"Smoothed (w={window})",
    )
    ax.axhline(
        y=3, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Victory (3)"
    )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
    ax.set_title(
        f"{algorithm} Training: Mantis Lords Defeated", fontweight="bold", fontsize=16
    )
    ax.set_ylim([-0.2, 3.5])
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_mantis_killed.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


def plot_single_loss(df, output_dir, algorithm="DQN", window=20):
    """Grafico Loss per singolo algoritmo."""
    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values

    if algorithm == "DQN" and "avg_loss" in df.columns:
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
            label=f"Smoothed (w={window})",
        )
        title = "Average Loss"
    elif algorithm == "PPO":
        if "actor_loss" in df.columns and "critic_loss" in df.columns:
            actor = df["actor_loss"].values
            critic = df["critic_loss"].values
            combined = (actor + critic) / 2
            smoothed = smooth(combined, window=window)
            ax.plot(
                episodes,
                combined,
                alpha=0.3,
                color="#ff7f0e",
                linewidth=1,
                label="Raw Combined",
            )
            ax.plot(
                episodes,
                smoothed,
                color="#2ca02c",
                linewidth=2.5,
                label=f"Smoothed (w={window})",
            )
            title = "Combined Loss (Actor + Critic)"
        else:
            print(f"⚠ Colonne loss non trovate per {algorithm}")
            return
    else:
        print(f"⚠ Colonne loss non trovate per {algorithm}")
        return

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title(f"{algorithm} Training: {title}", fontweight="bold", fontsize=16)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_loss.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


def plot_single_exploration(df, output_dir, algorithm="DQN", window=20):
    """Grafico Exploration per singolo algoritmo."""
    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values

    if algorithm == "DQN" and "epsilon" in df.columns:
        epsilon = df["epsilon"].values
        ax.plot(episodes, epsilon, color="#9467bd", linewidth=2, label="Epsilon")
        ax.fill_between(episodes, 0, epsilon, alpha=0.3, color="#9467bd")
        ax.set_ylabel("Epsilon Value", fontweight="bold")
        ax.set_ylim([-0.05, 1.05])
        title = "Epsilon Decay"
    elif algorithm == "PPO" and "entropy" in df.columns:
        entropy = df["entropy"].values
        smoothed = smooth(entropy, window=window)
        ax.plot(episodes, entropy, alpha=0.3, color="#9467bd", linewidth=1, label="Raw")
        ax.plot(
            episodes,
            smoothed,
            color="#d62728",
            linewidth=2.5,
            label=f"Smoothed (w={window})",
        )
        ax.set_ylabel("Entropy", fontweight="bold")
        title = "Policy Entropy"
    else:
        print(f"⚠ Colonne exploration non trovate per {algorithm}")
        return

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_title(f"{algorithm} Training: {title}", fontweight="bold", fontsize=16)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_exploration.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


def plot_single_dashboard(df, output_dir, algorithm="DQN", window=20):
    """Dashboard unificata per singolo algoritmo."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"{algorithm} Training Dashboard", fontsize=18, fontweight="bold", y=0.995
    )

    episodes = df["episode"].values

    # 1. Cumulative Reward
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
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # 2. Episode Length
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
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 3. Mantis Lords
    ax = axes[1, 0]
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
            y=3, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Victory"
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
        ax.set_title("Mantis Lords Progress", fontweight="bold")
        ax.set_ylim([-0.2, 3.5])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 4. Statistics
    ax = axes[1, 1]
    ax.axis("off")

    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    std_reward = np.std(rewards)

    if "mantis_killed" in df.columns:
        max_mantis = np.max(df["mantis_killed"].values)
        avg_mantis = np.mean(df["mantis_killed"].values)
    else:
        max_mantis = avg_mantis = 0

    stats_text = f"""
    {algorithm} TRAINING STATISTICS
    ══════════════════════════════════════

    Episodes: {len(episodes)}

    Rewards:
    ├─ Mean: {mean_reward:.2f}
    ├─ Max: {max_reward:.2f}
    ├─ Min: {min_reward:.2f}
    └─ Std Dev: {std_reward:.2f}

    Mantis Lords:
    ├─ Average Killed: {avg_mantis:.2f}
    └─ Max Killed: {max_mantis:.0f}

    Progress: {"🏆 Boss Defeated!" if max_mantis >= 3 else f"⚔️ Best: {max_mantis:.0f}/3"}
    """

    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


# ============================================================================
# GRAFICI CONFRONTO
# ============================================================================


def plot_comparison_overlay(df_ppo, df_dqn, output_dir, window=20):
    """Overlay PPO vs DQN sugli stessi assi."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "PPO vs DQN: Comparison Dashboard", fontsize=18, fontweight="bold", y=0.995
    )

    episodes_ppo = df_ppo["episode"].values
    episodes_dqn = df_dqn["episode"].values

    # 1. Rewards
    ax = axes[0, 0]
    rewards_ppo = smooth(df_ppo["total_reward"].values, window)
    rewards_dqn = smooth(df_dqn["total_reward"].values, window)
    ax.plot(
        episodes_ppo,
        rewards_ppo,
        color="#d62728",
        linewidth=2.5,
        label="PPO",
        alpha=0.8,
    )
    ax.plot(
        episodes_dqn,
        rewards_dqn,
        color="#2ca02c",
        linewidth=2.5,
        label="DQN",
        alpha=0.8,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title("Cumulative Reward", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # 2. Episode Length
    ax = axes[0, 1]
    if "steps" in df_ppo.columns and "steps" in df_dqn.columns:
        steps_ppo = smooth(df_ppo["steps"].values, window)
        steps_dqn = smooth(df_dqn["steps"].values, window)
        ax.plot(
            episodes_ppo,
            steps_ppo,
            color="#d62728",
            linewidth=2.5,
            label="PPO",
            alpha=0.8,
        )
        ax.plot(
            episodes_dqn,
            steps_dqn,
            color="#2ca02c",
            linewidth=2.5,
            label="DQN",
            alpha=0.8,
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Steps", fontweight="bold")
        ax.set_title("Episode Length", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 3. Mantis Lords
    ax = axes[1, 0]
    if "mantis_killed" in df_ppo.columns and "mantis_killed" in df_dqn.columns:
        mantis_ppo = smooth(df_ppo["mantis_killed"].values, window)
        mantis_dqn = smooth(df_dqn["mantis_killed"].values, window)
        ax.plot(
            episodes_ppo,
            mantis_ppo,
            color="#d62728",
            linewidth=2.5,
            label="PPO",
            alpha=0.8,
        )
        ax.plot(
            episodes_dqn,
            mantis_dqn,
            color="#2ca02c",
            linewidth=2.5,
            label="DQN",
            alpha=0.8,
        )
        ax.axhline(
            y=3, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Victory"
        )
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Mantis Lords Killed", fontweight="bold")
        ax.set_title("Mantis Lords Progress", fontweight="bold")
        ax.set_ylim([-0.2, 3.5])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 4. Stats Comparison
    ax = axes[1, 1]
    ax.axis("off")

    ppo_mean = np.mean(df_ppo["total_reward"].values)
    dqn_mean = np.mean(df_dqn["total_reward"].values)
    ppo_max = np.max(df_ppo["total_reward"].values)
    dqn_max = np.max(df_dqn["total_reward"].values)

    winner = "PPO" if ppo_mean > dqn_mean else "DQN" if dqn_mean > ppo_mean else "TIE"

    stats_text = f"""
    COMPARISON STATISTICS
    ══════════════════════════════════════

    PPO:
    ├─ Episodes: {len(episodes_ppo)}
    ├─ Mean Reward: {ppo_mean:.2f}
    └─ Max Reward: {ppo_max:.2f}

    DQN:
    ├─ Episodes: {len(episodes_dqn)}
    ├─ Mean Reward: {dqn_mean:.2f}
    └─ Max Reward: {dqn_max:.2f}

    Winner (Mean Reward): {winner}
    Difference: {abs(ppo_mean - dqn_mean):.2f}
    """

    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di training per DQN, PPO o confronto"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dqn", "ppo", "compare"],
        required=True,
        help="Modalità: dqn, ppo, o compare",
    )
    parser.add_argument("--dqn-log", type=str, help="Path al log DQN")
    parser.add_argument("--ppo-log", type=str, help="Path al log PPO")
    parser.add_argument(
        "--output", type=str, default="plots", help="Directory di output"
    )
    parser.add_argument("--window", type=int, default=20, help="Finestra smoothing")

    args = parser.parse_args()

    # Validazione argomenti
    if args.mode == "dqn" and not args.dqn_log:
        parser.error("--dqn-log è richiesto per mode=dqn")
    if args.mode == "ppo" and not args.ppo_log:
        parser.error("--ppo-log è richiesto per mode=ppo")
    if args.mode == "compare" and (not args.dqn_log or not args.ppo_log):
        parser.error("--dqn-log e --ppo-log sono richiesti per mode=compare")

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generazione Grafici - Mode: {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Window: {args.window}\n")

    # Modalità DQN
    if args.mode == "dqn":
        df_dqn = load_log(args.dqn_log)
        print(f"✓ DQN: {len(df_dqn)} episodi\n")
        print("Generazione grafici DQN...")
        plot_single_reward(df_dqn, args.output, "DQN", args.window)
        plot_single_episode_length(df_dqn, args.output, "DQN", args.window)
        plot_single_mantis(df_dqn, args.output, "DQN", args.window)
        plot_single_loss(df_dqn, args.output, "DQN", args.window)
        plot_single_exploration(df_dqn, args.output, "DQN", args.window)
        plot_single_dashboard(df_dqn, args.output, "DQN", args.window)

    # Modalità PPO
    elif args.mode == "ppo":
        df_ppo = load_log(args.ppo_log)
        print(f"✓ PPO: {len(df_ppo)} episodi\n")
        print("Generazione grafici PPO...")
        plot_single_reward(df_ppo, args.output, "PPO", args.window)
        plot_single_episode_length(df_ppo, args.output, "PPO", args.window)
        plot_single_mantis(df_ppo, args.output, "PPO", args.window)
        plot_single_loss(df_ppo, args.output, "PPO", args.window)
        plot_single_exploration(df_ppo, args.output, "PPO", args.window)
        plot_single_dashboard(df_ppo, args.output, "PPO", args.window)

    # Modalità Confronto
    elif args.mode == "compare":
        df_ppo = load_log(args.ppo_log)
        df_dqn = load_log(args.dqn_log)
        print(f"✓ PPO: {len(df_ppo)} episodi")
        print(f"✓ DQN: {len(df_dqn)} episodi\n")
        print("Generazione confronto...")
        plot_comparison_overlay(df_ppo, df_dqn, args.output, args.window)

    print(f"\n{'='*60}")
    print(f"✓ Completato! Grafici in: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
