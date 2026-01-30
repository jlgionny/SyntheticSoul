r"""
Professional Training Visualization for Academic Presentations
==============================================================

Genera grafici di alta qualità per presentazioni universitarie di progetti
di Reinforcement Learning. Supporta DQN, PPO e confronti multi-istanza.

Grafici disponibili:
1. Learning Curve con confidence interval
2. Win Rate (Mantis Lords defeated rate)
3. Sample Efficiency (reward per step)
4. Loss Convergence Analysis
5. Exploration vs Exploitation trade-off
6. Multi-Instance Comparison
7. Statistical Summary Table
8. Publication-ready Dashboard

Usage:
    # Singolo algoritmo
    python generate_plots.py --mode dqn --dqn-log path/to/log.txt --output plots/

    # Confronto DQN vs PPO
    python generate_plots.py --mode compare --dqn-log dqn_log.txt --ppo-log ppo_log.txt

    # Multi-istanza (da multi_train.py)
    python generate_plots.py --mode multi --multi-log path/to/training_log.txt

    # Tutti i grafici per presentazione
    python generate_plots.py --mode presentation --dqn-log log.txt --output slides/
"""

import argparse
import os
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# STYLE CONFIGURATION - Academic/Professional Theme
# ============================================================================

# Color palettes
COLORS = {
    "dqn": "#2E86AB",  # Blue
    "ppo": "#A23B72",  # Magenta
    "reward": "#F18F01",  # Orange
    "loss": "#C73E1D",  # Red
    "exploration": "#3A7CA5",  # Steel Blue
    "success": "#2E8B57",  # Sea Green
    "grid": "#E0E0E0",
    "text": "#2C3E50",
    "background": "#FAFAFA",
}


# Professional style setup
def setup_style():
    """Configure matplotlib for academic presentations."""
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            # Figure
            "figure.figsize": (12, 7),
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            # Fonts - Professional and readable
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            # Axes
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#2C3E50",
            "axes.titleweight": "bold",
            "axes.labelweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.alpha": 0.4,
            "grid.linewidth": 0.8,
            # Legend
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            # Savefig
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.facecolor": "white",
        }
    )


setup_style()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode="nearest")


def compute_confidence_interval(
    data: np.ndarray, window: int = 50, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling mean with confidence interval.
    Returns: (mean, lower_bound, upper_bound)
    """
    n = len(data)
    means = []
    lowers = []
    uppers = []

    for i in range(n):
        start = max(0, i - window + 1)
        segment = data[start : i + 1]

        mean = np.mean(segment)
        std = np.std(segment)
        n_samples = len(segment)

        # t-distribution for small samples
        if n_samples > 1:
            t_val = stats.t.ppf((1 + confidence) / 2, n_samples - 1)
            margin = t_val * std / np.sqrt(n_samples)
        else:
            margin = 0

        means.append(mean)
        lowers.append(mean - margin)
        uppers.append(mean + margin)

    return np.array(means), np.array(lowers), np.array(uppers)


def compute_win_rate(mantis_killed: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute rolling win rate (3 mantis killed = win)."""
    wins = (mantis_killed >= 3).astype(float)
    return smooth(wins * 100, window)


def load_log(log_file: str) -> pd.DataFrame:
    """Load training log with error handling."""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    df = pd.read_csv(log_file)

    # Standardize column names
    column_mapping = {
        "total_reward": "reward",
        "avg_loss": "loss",
        "epsilon_or_entropy": "exploration",
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    return df


def get_algorithm_from_log(df: pd.DataFrame) -> str:
    """Detect algorithm type from log columns."""
    if "epsilon" in df.columns:
        return "DQN"
    elif "entropy" in df.columns or "actor_loss" in df.columns:
        return "PPO"
    return "Unknown"


# ============================================================================
# CORE PLOTTING FUNCTIONS - Academic Quality
# ============================================================================


def plot_learning_curve(
    df: pd.DataFrame,
    output_dir: str,
    algorithm: str = "DQN",
    window: int = 50,
    show_ci: bool = True,
):
    """
    Learning Curve with Confidence Interval.
    This is the most important graph for academic presentations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    episodes = df["episode"].values
    rewards = (
        df["total_reward"].values
        if "total_reward" in df.columns
        else df["reward"].values
    )

    color = COLORS["dqn"] if algorithm == "DQN" else COLORS["ppo"]

    # Raw data (very transparent)
    ax.plot(
        episodes, rewards, alpha=0.15, color=color, linewidth=0.8, label="_nolegend_"
    )

    if show_ci:
        # Compute confidence interval
        mean, lower, upper = compute_confidence_interval(rewards, window)

        # Confidence band
        ax.fill_between(
            episodes,
            lower,
            upper,
            alpha=0.25,
            color=color,
            label="95% Confidence Interval",
        )

        # Mean line
        ax.plot(
            episodes,
            mean,
            color=color,
            linewidth=2.5,
            label=f"{algorithm} Mean Reward (window={window})",
        )
    else:
        smoothed = smooth(rewards, window)
        ax.plot(
            episodes,
            smoothed,
            color=color,
            linewidth=2.5,
            label=f"{algorithm} Smoothed Reward",
        )

    # Reference lines
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.6)

    # Best reward annotation
    best_idx = np.argmax(rewards)
    best_reward = rewards[best_idx]
    ax.annotate(
        f"Best: {best_reward:.1f}",
        xy=(episodes[best_idx], best_reward),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
    )

    ax.set_xlabel("Training Episode", fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontweight="bold")
    ax.set_title(f"{algorithm} Learning Curve", fontweight="bold", fontsize=18, pad=15)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add training progress annotation
    final_mean = (
        np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
    )
    initial_mean = np.mean(rewards[:window]) if len(rewards) >= window else rewards[0]
    improvement = (
        ((final_mean - initial_mean) / abs(initial_mean) * 100)
        if initial_mean != 0
        else 0
    )

    textstr = f"Episodes: {len(episodes)}\nImprovement: {improvement:+.1f}%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#CCCCCC")
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_learning_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_win_rate(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN", window: int = 50
):
    """
    Win Rate over Training.
    Shows the probability of defeating all 3 Mantis Lords.
    """
    if "mantis_killed" not in df.columns:
        print("  [SKIP] Win rate plot - 'mantis_killed' column not found")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    mantis = df["mantis_killed"].values

    color = COLORS["success"]

    # Compute win rate
    win_rate = compute_win_rate(mantis, window)

    # Raw wins as scatter
    wins = mantis >= 3
    ax.scatter(
        episodes[wins],
        np.ones(np.sum(wins)) * 100,
        alpha=0.3,
        color=color,
        s=20,
        label="Victories",
    )

    # Win rate line
    ax.plot(
        episodes,
        win_rate,
        color=color,
        linewidth=2.5,
        label=f"Win Rate (rolling {window} episodes)",
    )

    # Fill under curve
    ax.fill_between(episodes, 0, win_rate, alpha=0.2, color=color)

    # Reference lines
    ax.axhline(
        y=50,
        color="#888888",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="50% Baseline",
    )

    ax.set_xlabel("Training Episode", fontweight="bold")
    ax.set_ylabel("Win Rate (%)", fontweight="bold")
    ax.set_title(
        f"{algorithm} Victory Rate Against Mantis Lords",
        fontweight="bold",
        fontsize=18,
        pad=15,
    )
    ax.set_ylim([-5, 105])
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Statistics annotation
    total_wins = np.sum(wins)
    total_games = len(episodes)
    overall_rate = total_wins / total_games * 100
    final_rate = win_rate[-1] if len(win_rate) > 0 else 0

    textstr = f"Total Wins: {total_wins}/{total_games}\nOverall: {overall_rate:.1f}%\nFinal Rate: {final_rate:.1f}%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#CCCCCC")
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_win_rate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_sample_efficiency(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN", window: int = 50
):
    """
    Sample Efficiency Analysis.
    Shows reward gained per environment step (important metric for RL).
    """
    if "steps" not in df.columns:
        print("  [SKIP] Sample efficiency plot - 'steps' column not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    episodes = df["episode"].values
    rewards = (
        df["total_reward"].values
        if "total_reward" in df.columns
        else df["reward"].values
    )
    steps = df["steps"].values

    color = COLORS["dqn"] if algorithm == "DQN" else COLORS["ppo"]

    # Left: Cumulative steps vs cumulative reward
    cumulative_steps = np.cumsum(steps)
    cumulative_reward = np.cumsum(rewards)

    ax1.plot(cumulative_steps, cumulative_reward, color=color, linewidth=2)
    ax1.fill_between(cumulative_steps, 0, cumulative_reward, alpha=0.2, color=color)
    ax1.set_xlabel("Total Environment Steps", fontweight="bold")
    ax1.set_ylabel("Cumulative Reward", fontweight="bold")
    ax1.set_title("Sample Efficiency", fontweight="bold", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Efficiency metric
    if cumulative_steps[-1] > 0:
        efficiency = cumulative_reward[-1] / cumulative_steps[-1] * 1000
        ax1.text(
            0.98,
            0.02,
            f"Efficiency: {efficiency:.2f} reward/1000 steps",
            transform=ax1.transAxes,
            fontsize=10,
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Right: Reward per episode vs steps per episode
    reward_per_step = rewards / np.maximum(steps, 1)
    smoothed_rps = smooth(reward_per_step, window)

    ax2.plot(episodes, reward_per_step, alpha=0.2, color=color, linewidth=0.8)
    ax2.plot(
        episodes, smoothed_rps, color=color, linewidth=2.5, label="Reward per Step"
    )
    ax2.set_xlabel("Training Episode", fontweight="bold")
    ax2.set_ylabel("Reward / Step", fontweight="bold")
    ax2.set_title("Per-Episode Efficiency", fontweight="bold", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"{algorithm} Sample Efficiency Analysis",
        fontweight="bold",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_sample_efficiency.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_loss_analysis(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN", window: int = 30
):
    """
    Loss Convergence Analysis with trend detection.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values

    if algorithm == "DQN" and "avg_loss" in df.columns:
        losses = df["avg_loss"].values
        label = "TD Loss"
    elif algorithm == "PPO" and "actor_loss" in df.columns:
        actor = df["actor_loss"].values
        critic = (
            df["critic_loss"].values
            if "critic_loss" in df.columns
            else np.zeros_like(actor)
        )
        losses = actor + critic
        label = "Total Loss (Actor + Critic)"
    elif "loss" in df.columns:
        losses = df["loss"].values
        label = "Loss"
    else:
        print("  [SKIP] Loss plot - no loss column found")
        return

    # Remove NaN and infinite values
    valid_mask = np.isfinite(losses)
    episodes_valid = episodes[valid_mask]
    losses_valid = losses[valid_mask]

    if len(losses_valid) == 0:
        print("  [SKIP] Loss plot - no valid loss values")
        return

    color = COLORS["loss"]

    # Raw loss
    ax.plot(episodes_valid, losses_valid, alpha=0.2, color=color, linewidth=0.8)

    # Smoothed loss
    smoothed = smooth(losses_valid, window)
    ax.plot(
        episodes_valid,
        smoothed,
        color=color,
        linewidth=2.5,
        label=f"{label} (smoothed)",
    )

    # Trend line (linear regression)
    if len(episodes_valid) > 10:
        z = np.polyfit(episodes_valid, losses_valid, 1)
        p = np.poly1d(z)
        ax.plot(
            episodes_valid,
            p(episodes_valid),
            "--",
            color="#333333",
            linewidth=1.5,
            alpha=0.7,
            label=f"Trend (slope={z[0]:.2e})",
        )

    ax.set_xlabel("Training Episode", fontweight="bold")
    ax.set_ylabel("Loss Value", fontweight="bold")
    ax.set_title(
        f"{algorithm} Loss Convergence", fontweight="bold", fontsize=18, pad=15
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Use log scale if loss varies by orders of magnitude
    loss_range = np.max(losses_valid) / (np.min(losses_valid[losses_valid > 0]) + 1e-10)
    if loss_range > 100:
        ax.set_yscale("log")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_loss_convergence.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_exploration_exploitation(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN"
):
    """
    Exploration vs Exploitation trade-off visualization.
    Shows epsilon decay (DQN) or entropy (PPO) alongside performance.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    episodes = df["episode"].values
    rewards = (
        df["total_reward"].values
        if "total_reward" in df.columns
        else df["reward"].values
    )

    # Primary axis: Reward
    color1 = COLORS["reward"]
    smoothed_rewards = smooth(rewards, 30)
    ax1.plot(episodes, smoothed_rewards, color=color1, linewidth=2.5, label="Reward")
    ax1.fill_between(episodes, 0, smoothed_rewards, alpha=0.1, color=color1)
    ax1.set_xlabel("Training Episode", fontweight="bold")
    ax1.set_ylabel("Cumulative Reward", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Secondary axis: Exploration metric
    ax2 = ax1.twinx()
    color2 = COLORS["exploration"]

    if algorithm == "DQN" and "epsilon" in df.columns:
        exploration = df["epsilon"].values
        label = "Epsilon (Exploration Rate)"
        ax2.set_ylim([0, 1.1])
    elif algorithm == "PPO" and "entropy" in df.columns:
        exploration = df["entropy"].values
        label = "Policy Entropy"
    elif "exploration" in df.columns:
        exploration = df["exploration"].values
        label = "Exploration Metric"
    else:
        print("  [SKIP] Exploration plot - no exploration column found")
        plt.close()
        return

    ax2.plot(
        episodes, exploration, color=color2, linewidth=2.5, linestyle="--", label=label
    )
    ax2.set_ylabel(label, color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(
        f"{algorithm} Exploration-Exploitation Trade-off",
        fontweight="bold",
        fontsize=18,
        pad=15,
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(
        output_dir, f"{algorithm.lower()}_exploration_exploitation.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_mantis_progress(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN", window: int = 30
):
    """
    Detailed Mantis Lords progress visualization.
    """
    if "mantis_killed" not in df.columns:
        print("  [SKIP] Mantis progress plot - 'mantis_killed' column not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    episodes = df["episode"].values
    mantis = df["mantis_killed"].values

    color = COLORS["dqn"] if algorithm == "DQN" else COLORS["ppo"]

    # Left: Progress over time
    smoothed = smooth(mantis, window)
    ax1.scatter(episodes, mantis, alpha=0.3, color=color, s=15, label="Per Episode")
    ax1.plot(
        episodes, smoothed, color=color, linewidth=2.5, label=f"Moving Avg (w={window})"
    )
    ax1.axhline(
        y=3,
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Victory Threshold",
    )
    ax1.axhline(y=1, color="#888888", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhline(y=2, color="#888888", linestyle=":", linewidth=1, alpha=0.5)

    ax1.set_xlabel("Training Episode", fontweight="bold")
    ax1.set_ylabel("Mantis Lords Defeated", fontweight="bold")
    ax1.set_title("Progress Over Training", fontweight="bold", fontsize=14)
    ax1.set_ylim([-0.2, 3.5])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Right: Distribution histogram
    counts = [np.sum(mantis == i) for i in range(4)]
    bars = ax2.bar(
        [0, 1, 2, 3],
        counts,
        color=[color, color, color, COLORS["success"]],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add percentages on bars
    total = len(mantis)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.annotate(
            f"{count}\n({count/total*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax2.set_xlabel("Mantis Lords Defeated", fontweight="bold")
    ax2.set_ylabel("Number of Episodes", fontweight="bold")
    ax2.set_title("Outcome Distribution", fontweight="bold", fontsize=14)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["0 (Loss)", "1", "2", "3 (Victory)"])
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"{algorithm} Mantis Lords Combat Performance",
        fontweight="bold",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{algorithm.lower()}_mantis_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


# ============================================================================
# COMPARISON PLOTS - DQN vs PPO
# ============================================================================


def plot_comparison_learning_curves(
    df_dqn: pd.DataFrame, df_ppo: pd.DataFrame, output_dir: str, window: int = 50
):
    """
    Side-by-side and overlay comparison of learning curves.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Data
    ep_dqn = df_dqn["episode"].values
    ep_ppo = df_ppo["episode"].values
    rew_dqn = (
        df_dqn["total_reward"].values
        if "total_reward" in df_dqn.columns
        else df_dqn["reward"].values
    )
    rew_ppo = (
        df_ppo["total_reward"].values
        if "total_reward" in df_ppo.columns
        else df_ppo["reward"].values
    )

    # Top-left: DQN with CI
    ax1 = fig.add_subplot(gs[0, 0])
    mean_dqn, lower_dqn, upper_dqn = compute_confidence_interval(rew_dqn, window)
    ax1.fill_between(ep_dqn, lower_dqn, upper_dqn, alpha=0.25, color=COLORS["dqn"])
    ax1.plot(ep_dqn, mean_dqn, color=COLORS["dqn"], linewidth=2.5, label="DQN")
    ax1.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("DQN Learning Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: PPO with CI
    ax2 = fig.add_subplot(gs[0, 1])
    mean_ppo, lower_ppo, upper_ppo = compute_confidence_interval(rew_ppo, window)
    ax2.fill_between(ep_ppo, lower_ppo, upper_ppo, alpha=0.25, color=COLORS["ppo"])
    ax2.plot(ep_ppo, mean_ppo, color=COLORS["ppo"], linewidth=2.5, label="PPO")
    ax2.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("PPO Learning Curve", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Overlay comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        ep_dqn,
        smooth(rew_dqn, window),
        color=COLORS["dqn"],
        linewidth=2.5,
        label=f"DQN (n={len(ep_dqn)})",
    )
    ax3.plot(
        ep_ppo,
        smooth(rew_ppo, window),
        color=COLORS["ppo"],
        linewidth=2.5,
        label=f"PPO (n={len(ep_ppo)})",
    )
    ax3.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward (Smoothed)")
    ax3.set_title("Direct Comparison", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Compute statistics
    stats_data = {
        "Metric": [
            "Episodes",
            "Mean Reward",
            "Max Reward",
            "Min Reward",
            "Std Dev",
            "Final Reward (last 50)",
        ],
        "DQN": [
            len(ep_dqn),
            f"{np.mean(rew_dqn):.2f}",
            f"{np.max(rew_dqn):.2f}",
            f"{np.min(rew_dqn):.2f}",
            f"{np.std(rew_dqn):.2f}",
            (
                f"{np.mean(rew_dqn[-50:]):.2f}"
                if len(rew_dqn) >= 50
                else f"{np.mean(rew_dqn):.2f}"
            ),
        ],
        "PPO": [
            len(ep_ppo),
            f"{np.mean(rew_ppo):.2f}",
            f"{np.max(rew_ppo):.2f}",
            f"{np.min(rew_ppo):.2f}",
            f"{np.std(rew_ppo):.2f}",
            (
                f"{np.mean(rew_ppo[-50:]):.2f}"
                if len(rew_ppo) >= 50
                else f"{np.mean(rew_ppo):.2f}"
            ),
        ],
    }

    # Add win rate if available
    if "mantis_killed" in df_dqn.columns and "mantis_killed" in df_ppo.columns:
        dqn_wins = np.sum(df_dqn["mantis_killed"].values >= 3) / len(df_dqn) * 100
        ppo_wins = np.sum(df_ppo["mantis_killed"].values >= 3) / len(df_ppo) * 100
        stats_data["Metric"].append("Win Rate")
        stats_data["DQN"].append(f"{dqn_wins:.1f}%")
        stats_data["PPO"].append(f"{ppo_wins:.1f}%")

    # Create table
    table = ax4.table(
        cellText=list(zip(stats_data["Metric"], stats_data["DQN"], stats_data["PPO"])),
        colLabels=["Metric", "DQN", "PPO"],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor("#4A4A4A")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(stats_data["Metric"]) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")

    ax4.set_title("Statistical Comparison", fontweight="bold", pad=20)

    plt.suptitle(
        "DQN vs PPO: Comprehensive Comparison", fontweight="bold", fontsize=20, y=0.98
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_learning_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


def plot_comparison_win_rates(
    df_dqn: pd.DataFrame, df_ppo: pd.DataFrame, output_dir: str, window: int = 50
):
    """
    Compare win rates between DQN and PPO.
    """
    if "mantis_killed" not in df_dqn.columns or "mantis_killed" not in df_ppo.columns:
        print("  [SKIP] Win rate comparison - 'mantis_killed' column not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Win rate over time
    ep_dqn = df_dqn["episode"].values
    ep_ppo = df_ppo["episode"].values
    wr_dqn = compute_win_rate(df_dqn["mantis_killed"].values, window)
    wr_ppo = compute_win_rate(df_ppo["mantis_killed"].values, window)

    ax1.plot(ep_dqn, wr_dqn, color=COLORS["dqn"], linewidth=2.5, label="DQN")
    ax1.plot(ep_ppo, wr_ppo, color=COLORS["ppo"], linewidth=2.5, label="PPO")
    ax1.axhline(y=50, color="#888888", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Training Episode", fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontweight="bold")
    ax1.set_title("Win Rate Over Training", fontweight="bold", fontsize=14)
    ax1.set_ylim([-5, 105])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Final distribution comparison (bar chart)
    mantis_dqn = df_dqn["mantis_killed"].values
    mantis_ppo = df_ppo["mantis_killed"].values

    x = np.arange(4)
    width = 0.35

    counts_dqn = [np.sum(mantis_dqn == i) / len(mantis_dqn) * 100 for i in range(4)]
    counts_ppo = [np.sum(mantis_ppo == i) / len(mantis_ppo) * 100 for i in range(4)]

    bars1 = ax2.bar(
        x - width / 2, counts_dqn, width, label="DQN", color=COLORS["dqn"], alpha=0.8
    )
    bars2 = ax2.bar(
        x + width / 2, counts_ppo, width, label="PPO", color=COLORS["ppo"], alpha=0.8
    )

    ax2.set_xlabel("Mantis Lords Defeated", fontweight="bold")
    ax2.set_ylabel("Percentage of Episodes (%)", fontweight="bold")
    ax2.set_title("Outcome Distribution", fontweight="bold", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["0", "1", "2", "3 (Win)"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 2:  # Only label if > 2%
                ax2.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.suptitle(
        "DQN vs PPO: Victory Performance", fontweight="bold", fontsize=18, y=1.02
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_win_rates.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


# ============================================================================
# MULTI-INSTANCE PLOTS
# ============================================================================


def plot_multi_instance(df: pd.DataFrame, output_dir: str, window: int = 30):
    """
    Visualize multi-instance training results.
    """
    if "instance" not in df.columns:
        print("  [SKIP] Multi-instance plot - 'instance' column not found")
        return

    instances = df["instance"].unique()
    n_instances = len(instances)

    if n_instances < 2:
        print("  [SKIP] Multi-instance plot - only 1 instance found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_instances))

    # Top-left: Learning curves per instance
    ax = axes[0, 0]
    for i, inst in enumerate(instances):
        inst_df = df[df["instance"] == inst]
        rewards = (
            inst_df["total_reward"].values
            if "total_reward" in inst_df.columns
            else inst_df["reward"].values
        )
        smoothed = smooth(rewards, window)
        ax.plot(
            range(len(smoothed)),
            smoothed,
            color=colors[i],
            linewidth=2,
            label=f"Instance {inst}",
            alpha=0.8,
        )
    ax.set_xlabel("Episode (per instance)", fontweight="bold")
    ax.set_ylabel("Reward", fontweight="bold")
    ax.set_title("Learning Curves by Instance", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Top-right: Global learning curve (all instances combined)
    ax = axes[0, 1]
    global_rewards = df.groupby("episode")[
        "total_reward" if "total_reward" in df.columns else "reward"
    ].mean()
    mean, lower, upper = compute_confidence_interval(global_rewards.values, window)
    episodes = global_rewards.index.values
    ax.fill_between(episodes, lower, upper, alpha=0.3, color=COLORS["dqn"])
    ax.plot(episodes, mean, color=COLORS["dqn"], linewidth=2.5)
    ax.set_xlabel("Global Episode", fontweight="bold")
    ax.set_ylabel("Mean Reward", fontweight="bold")
    ax.set_title("Combined Learning Curve (All Instances)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Instance comparison boxplot
    ax = axes[1, 0]
    rewards_by_instance = [
        df[df["instance"] == inst][
            "total_reward" if "total_reward" in df.columns else "reward"
        ].values
        for inst in instances
    ]
    bp = ax.boxplot(rewards_by_instance, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"Inst {i}" for i in instances])
    ax.set_xlabel("Instance", fontweight="bold")
    ax.set_ylabel("Reward Distribution", fontweight="bold")
    ax.set_title("Performance Distribution by Instance", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Statistics
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "MULTI-INSTANCE STATISTICS\n" + "=" * 40 + "\n\n"
    for inst in instances:
        inst_df = df[df["instance"] == inst]
        rewards = (
            inst_df["total_reward"].values
            if "total_reward" in inst_df.columns
            else inst_df["reward"].values
        )
        stats_text += f"Instance {inst}:\n"
        stats_text += f"  Episodes: {len(rewards)}\n"
        stats_text += f"  Mean: {np.mean(rewards):.2f}\n"
        stats_text += f"  Max: {np.max(rewards):.2f}\n"
        if "mantis_killed" in inst_df.columns:
            wins = np.sum(inst_df["mantis_killed"].values >= 3)
            stats_text += f"  Wins: {wins} ({wins/len(rewards)*100:.1f}%)\n"
        stats_text += "\n"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(
        "Multi-Instance Training Analysis", fontweight="bold", fontsize=18, y=0.98
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, "multi_instance_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


# ============================================================================
# PRESENTATION DASHBOARD
# ============================================================================


def plot_presentation_dashboard(
    df: pd.DataFrame, output_dir: str, algorithm: str = "DQN", window: int = 50
):
    """
    Single comprehensive dashboard for presentations.
    Designed to fit on one slide with all key information.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    episodes = df["episode"].values
    rewards = (
        df["total_reward"].values
        if "total_reward" in df.columns
        else df["reward"].values
    )

    color = COLORS["dqn"] if algorithm == "DQN" else COLORS["ppo"]

    # 1. Main Learning Curve (large, top-left spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    mean, lower, upper = compute_confidence_interval(rewards, window)
    ax1.fill_between(episodes, lower, upper, alpha=0.25, color=color)
    ax1.plot(episodes, mean, color=color, linewidth=3, label="Mean Reward")
    ax1.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Training Episode", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Cumulative Reward", fontweight="bold", fontsize=12)
    ax1.set_title(
        "Learning Curve with 95% Confidence Interval", fontweight="bold", fontsize=14
    )
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Key Stats Box (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    final_mean = (
        np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
    )
    best_reward = np.max(rewards)
    improvement = (
        ((final_mean - rewards[0]) / abs(rewards[0]) * 100) if rewards[0] != 0 else 0
    )

    stats_lines = [
        "TRAINING SUMMARY",
        f"{'='*25}",
        "",
        f"Algorithm: {algorithm}",
        f"Episodes: {len(episodes)}",
        "",
        f"Best Reward: {best_reward:.1f}",
        f"Final Mean: {final_mean:.1f}",
        f"Improvement: {improvement:+.1f}%",
    ]

    if "mantis_killed" in df.columns:
        wins = np.sum(df["mantis_killed"].values >= 3)
        win_rate = wins / len(episodes) * 100
        stats_lines.extend(
            ["", f"Victories: {wins}/{len(episodes)}", f"Win Rate: {win_rate:.1f}%"]
        )

    stats_text = "\n".join(stats_lines)
    ax2.text(
        0.1,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round", facecolor="#F8F8F8", edgecolor="#CCCCCC", alpha=0.95
        ),
    )

    # 3. Win Rate / Mantis Progress (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if "mantis_killed" in df.columns:
        mantis = df["mantis_killed"].values
        win_rate = compute_win_rate(mantis, window)
        ax3.plot(episodes, win_rate, color=COLORS["success"], linewidth=2.5)
        ax3.fill_between(episodes, 0, win_rate, alpha=0.2, color=COLORS["success"])
        ax3.axhline(y=50, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
        ax3.set_ylim([-5, 105])
        ax3.set_ylabel("Win Rate (%)", fontweight="bold")
    else:
        smoothed = smooth(rewards, window)
        ax3.plot(episodes, smoothed, color=color, linewidth=2.5)
        ax3.set_ylabel("Reward", fontweight="bold")
    ax3.set_xlabel("Episode", fontweight="bold")
    ax3.set_title("Victory Rate", fontweight="bold", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Sample Efficiency (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    if "steps" in df.columns:
        steps = df["steps"].values
        cumulative_steps = np.cumsum(steps)
        cumulative_reward = np.cumsum(rewards)
        ax4.plot(cumulative_steps / 1000, cumulative_reward, color=color, linewidth=2.5)
        ax4.set_xlabel("Steps (thousands)", fontweight="bold")
        ax4.set_ylabel("Cumulative Reward", fontweight="bold")
        ax4.set_title("Sample Efficiency", fontweight="bold", fontsize=12)
    else:
        ax4.plot(episodes, np.cumsum(rewards), color=color, linewidth=2.5)
        ax4.set_xlabel("Episode", fontweight="bold")
        ax4.set_ylabel("Cumulative Reward", fontweight="bold")
        ax4.set_title("Total Reward Accumulated", fontweight="bold", fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. Loss Convergence (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    loss_col = (
        "avg_loss"
        if "avg_loss" in df.columns
        else "loss" if "loss" in df.columns else None
    )
    if loss_col:
        losses = df[loss_col].values
        valid = np.isfinite(losses)
        if np.sum(valid) > 0:
            ax5.plot(
                episodes[valid],
                smooth(losses[valid], 20),
                color=COLORS["loss"],
                linewidth=2.5,
            )
            ax5.set_xlabel("Episode", fontweight="bold")
            ax5.set_ylabel("Loss", fontweight="bold")
            ax5.set_title("Loss Convergence", fontweight="bold", fontsize=12)
            ax5.grid(True, alpha=0.3)

    # 6. Outcome Distribution (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    if "mantis_killed" in df.columns:
        mantis = df["mantis_killed"].values
        ax6.set_xticks([0, 1, 2, 3])
        ax6.set_xticklabels(["0", "1", "2", "3 (Win)"])
        ax6.set_xlabel("Mantis Lords Defeated", fontweight="bold")
        ax6.set_ylabel("Episodes", fontweight="bold")
        ax6.set_title("Outcome Distribution", fontweight="bold", fontsize=12)
        ax6.grid(True, alpha=0.3, axis="y")

    # 7. Exploration Metric (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    if "epsilon" in df.columns:
        ax7.plot(
            episodes, df["epsilon"].values, color=COLORS["exploration"], linewidth=2.5
        )
        ax7.set_ylabel("Epsilon", fontweight="bold")
        ax7.set_title(
            "Exploration Rate (Epsilon Decay)", fontweight="bold", fontsize=12
        )
        ax7.set_ylim([0, 1.1])
    elif "entropy" in df.columns:
        ax7.plot(
            episodes,
            smooth(df["entropy"].values, 20),
            color=COLORS["exploration"],
            linewidth=2.5,
        )
        ax7.set_ylabel("Entropy", fontweight="bold")
        ax7.set_title("Policy Entropy", fontweight="bold", fontsize=12)
    ax7.set_xlabel("Episode", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # 8. Reward Distribution (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.hist(rewards, bins=30, color=color, alpha=0.7, edgecolor="black")
    ax8.axvline(
        x=np.mean(rewards),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(rewards):.1f}",
    )
    ax8.set_xlabel("Reward", fontweight="bold")
    ax8.set_ylabel("Frequency", fontweight="bold")
    ax8.set_title("Reward Distribution", fontweight="bold", fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"{algorithm} Training Results - Hollow Knight Mantis Lords",
        fontweight="bold",
        fontsize=22,
        y=0.98,
    )

    plt.tight_layout()
    output_path = os.path.join(
        output_dir, f"{algorithm.lower()}_presentation_dashboard.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {output_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def generate_all_plots(
    df: pd.DataFrame, output_dir: str, algorithm: str, window: int = 50
):
    """Generate all plots for a single algorithm."""
    print(f"\nGenerating {algorithm} plots...")

    plot_learning_curve(df, output_dir, algorithm, window, show_ci=True)
    plot_win_rate(df, output_dir, algorithm, window)
    plot_sample_efficiency(df, output_dir, algorithm, window)
    plot_loss_analysis(df, output_dir, algorithm, window // 2)
    plot_exploration_exploitation(df, output_dir, algorithm)
    plot_mantis_progress(df, output_dir, algorithm, window)
    plot_presentation_dashboard(df, output_dir, algorithm, window)


def main():
    parser = argparse.ArgumentParser(
        description="Generate professional training visualizations for academic presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all DQN plots
  python generate_plots.py --mode dqn --dqn-log training_log.txt --output plots/

  # Compare DQN vs PPO
  python generate_plots.py --mode compare --dqn-log dqn_log.txt --ppo-log ppo_log.txt

  # Multi-instance analysis
  python generate_plots.py --mode multi --multi-log multi_training_log.txt

  # Quick presentation dashboard only
  python generate_plots.py --mode presentation --dqn-log log.txt --output slides/
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["dqn", "ppo", "compare", "multi", "presentation"],
        help="Plot generation mode",
    )
    parser.add_argument("--dqn-log", type=str, help="Path to DQN training log")
    parser.add_argument("--ppo-log", type=str, help="Path to PPO training log")
    parser.add_argument("--multi-log", type=str, help="Path to multi-instance log")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "dqn" and not args.dqn_log:
        parser.error("--dqn-log required for mode=dqn")
    if args.mode == "ppo" and not args.ppo_log:
        parser.error("--ppo-log required for mode=ppo")
    if args.mode == "compare" and (not args.dqn_log or not args.ppo_log):
        parser.error("Both --dqn-log and --ppo-log required for mode=compare")
    if args.mode == "multi" and not args.multi_log:
        parser.error("--multi-log required for mode=multi")
    if args.mode == "presentation" and not (args.dqn_log or args.ppo_log):
        parser.error("At least one log file required for mode=presentation")

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 60)
    print("PROFESSIONAL TRAINING VISUALIZATION")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Output: {args.output}")
    print(f"Smoothing Window: {args.window}")
    print("=" * 60)

    if args.mode == "dqn":
        df = load_log(args.dqn_log)
        print(f"Loaded: {len(df)} episodes")
        generate_all_plots(df, args.output, "DQN", args.window)

    elif args.mode == "ppo":
        df = load_log(args.ppo_log)
        print(f"Loaded: {len(df)} episodes")
        generate_all_plots(df, args.output, "PPO", args.window)

    elif args.mode == "compare":
        df_dqn = load_log(args.dqn_log)
        df_ppo = load_log(args.ppo_log)
        print(f"DQN: {len(df_dqn)} episodes")
        print(f"PPO: {len(df_ppo)} episodes")

        print("\nGenerating comparison plots...")
        plot_comparison_learning_curves(df_dqn, df_ppo, args.output, args.window)
        plot_comparison_win_rates(df_dqn, df_ppo, args.output, args.window)

        # Also generate individual dashboards
        plot_presentation_dashboard(df_dqn, args.output, "DQN", args.window)
        plot_presentation_dashboard(df_ppo, args.output, "PPO", args.window)

    elif args.mode == "multi":
        df = load_log(args.multi_log)
        algorithm = get_algorithm_from_log(df)
        print(f"Loaded: {len(df)} total entries")
        print(f"Detected algorithm: {algorithm}")

        plot_multi_instance(df, args.output, args.window)
        plot_presentation_dashboard(df, args.output, algorithm, args.window)

    elif args.mode == "presentation":
        if args.dqn_log:
            df = load_log(args.dqn_log)
            print(f"DQN: {len(df)} episodes")
            plot_presentation_dashboard(df, args.output, "DQN", args.window)
        if args.ppo_log:
            df = load_log(args.ppo_log)
            print(f"PPO: {len(df)} episodes")
            plot_presentation_dashboard(df, args.output, "PPO", args.window)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
