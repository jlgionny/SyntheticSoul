r"""
═══════════════════════════════════════════════════════════════════════
  Training & Inference Visualization — Mantis Lords RL
  ═══════════════════════════════════════════════════════════════════

  Genera grafici per training (DQN/PPO) e inference (play.py).

  TRAINING:
    python generate_plots.py --mode ppo --ppo-log training_output_ppo/phase_4/training_log_ppo.csv
    python generate_plots.py --mode dqn --dqn-log training_output_dqn/phase_3/training_log_dqn.csv
    python generate_plots.py --mode compare --dqn-log dqn.csv --ppo-log ppo.csv
    python generate_plots.py --mode multi --multi-log training_log_ppo.csv

  INFERENCE (play.py):
    python generate_plots.py --mode play --play-log play_log.csv
    python generate_plots.py --mode play --play-log play_log.csv --output play_plots/

  DASHBOARD:
    python generate_plots.py --mode presentation --ppo-log ppo.csv --output slides/
═══════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

RESTORE_REWARD_SCALE = True
REWARD_SCALE_FACTOR = 5.0

# Modern color palette
C = {
    "dqn":       "#0ea5e9",   # Sky blue
    "ppo":       "#c026d3",   # Fuchsia
    "reward":    "#f59e0b",   # Amber
    "loss":      "#ef4444",   # Red
    "explore":   "#6366f1",   # Indigo
    "success":   "#10b981",   # Emerald
    "kill0":     "#ef4444",   # Red
    "kill1":     "#f59e0b",   # Amber
    "kill2":     "#6366f1",   # Indigo
    "kill3":     "#10b981",   # Emerald (victory)
    "v1":        "#94a3b8",   # Slate
    "v2":        "#0ea5e9",   # Sky
    "v3":        "#10b981",   # Emerald
    "inst0":     "#0ea5e9",   # Sky
    "inst1":     "#c026d3",   # Fuchsia
    "inst2":     "#f59e0b",   # Amber
    "grid":      "#e2e8f0",
    "text":      "#1e293b",
    "muted":     "#94a3b8",
    "bg":        "#ffffff",
}


def setup_style():
    """Modern, clean matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 8),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": C["text"],
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "grid.color": C["grid"],
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#e2e8f0",
        "legend.fancybox": True,
        "lines.linewidth": 2.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white",
    })


setup_style()


# ============================================================================
# UTILITY
# ============================================================================

def smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    if len(data) < window:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode="nearest")


def ci(data: np.ndarray, window: int = 50, conf: float = 0.95):
    """Rolling mean + confidence interval."""
    n = len(data)
    means, lowers, uppers = [], [], []
    for i in range(n):
        seg = data[max(0, i - window + 1): i + 1]
        m = np.mean(seg)
        s = np.std(seg)
        ns = len(seg)
        if ns > 1:
            t = stats.t.ppf((1 + conf) / 2, ns - 1)
            margin = t * s / np.sqrt(ns)
        else:
            margin = 0
        means.append(m)
        lowers.append(m - margin)
        uppers.append(m + margin)
    return np.array(means), np.array(lowers), np.array(uppers)


def win_rate_rolling(mantis: np.ndarray, window: int = 50) -> np.ndarray:
    wins = (mantis >= 3).astype(float)
    return smooth(wins * 100, window)


def detect_iterations(df: pd.DataFrame):
    """Auto-detect training iterations from episode numbers."""
    eps = df["episode"].values
    max_ep = eps.max()
    if max_ep <= 2000:
        return {"v1": df}
    elif max_ep <= 4000:
        return {
            "v1": df[df["episode"] <= 2000],
            "v2": df[(df["episode"] > 2000) & (df["episode"] <= 4000)],
        }
    else:
        result = {
            "v1": df[df["episode"] <= 2000],
            "v2": df[(df["episode"] > 2000) & (df["episode"] <= 4000)],
            "v3": df[df["episode"] > 4000],
        }
        return {k: v for k, v in result.items() if len(v) > 0}


def add_subtitle(ax, text: str):
    """Add muted subtitle under the title."""
    ax.text(0.0, 1.04, text, transform=ax.transAxes, fontsize=10,
            color=C["muted"], va="bottom")


def stat_box(ax, lines: list, x=0.02, y=0.97):
    """Overlay a stat box."""
    text = "\n".join(lines)
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, va="top",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", edgecolor="#e2e8f0", alpha=0.92))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_log(log_file: str) -> pd.DataFrame:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"File non trovato: {log_file}")
    df = pd.read_csv(log_file)
    remap = {"total_reward": "reward", "avg_loss": "loss", "instance_id": "instance"}
    for old, new in remap.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "epsilon" in df.columns and "exploration" not in df.columns:
        df["exploration"] = df["epsilon"]
    elif "entropy" in df.columns and "exploration" not in df.columns:
        df["exploration"] = df["entropy"]
    numerics = ["reward", "steps", "mantis_killed", "boss_hp", "boss_defeated",
                "epsilon", "entropy", "exploration", "learning_rate", "loss",
                "num_updates", "episode", "instance", "phase"]
    for col in numerics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if RESTORE_REWARD_SCALE and "reward" in df.columns:
        df["reward"] = df["reward"] * REWARD_SCALE_FACTOR
    return df


def load_play_log(log_file: str) -> pd.DataFrame:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"File non trovato: {log_file}")
    df = pd.read_csv(log_file)
    numerics = ["run", "mantis_killed", "boss_hp", "player_hp", "steps", "duration_sec"]
    for col in numerics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_algo(df: pd.DataFrame) -> str:
    if "epsilon" in df.columns and "entropy" not in df.columns:
        return "DQN"
    elif "entropy" in df.columns:
        return "PPO"
    return "RL"


# ============================================================================
# TRAINING PLOTS
# ============================================================================

def plot_learning_curve(df, out, algo="PPO", window=50):
    fig, ax = plt.subplots(figsize=(14, 8))
    episodes = df["episode"].values
    rewards = df["reward"].values
    color = C["dqn"] if algo == "DQN" else C["ppo"]

    ax.plot(episodes, rewards, alpha=0.08, color=color, linewidth=0.5)
    mean, lower, upper = ci(rewards, window)
    ax.fill_between(episodes, lower, upper, alpha=0.15, color=color)
    ax.plot(episodes, mean, color=color, linewidth=2.5, label=f"Media mobile (w={window})")
    ax.axhline(y=0, color=C["muted"], linestyle="--", linewidth=0.8, alpha=0.5)

    best_idx = np.argmax(rewards)
    ax.annotate(f"Best: {rewards[best_idx]:.0f}", xy=(episodes[best_idx], rewards[best_idx]),
                xytext=(15, 10), textcoords="offset points", fontsize=10, color=color,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward cumulativo")
    ax.set_title(f"{algo} — Learning Curve")
    add_subtitle(ax, f"{len(episodes)} episodi · 95% CI")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)

    final = np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
    initial = np.mean(rewards[:window]) if len(rewards) >= window else rewards[0]
    improv = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
    stat_box(ax, [f"Episodi: {len(episodes)}", f"Best: {rewards[best_idx]:.0f}",
                  f"Finale: {final:.0f}", f"Δ: {improv:+.0f}%"])

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_learning_curve.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_learning_curve.png")


def plot_kill_and_win_rate(df, out, algo="PPO", window=50):
    if "mantis_killed" not in df.columns:
        return
    fig, ax1 = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values
    mantis = df["mantis_killed"].values

    kill_smooth = smooth(mantis.astype(float), window)
    ax1.plot(episodes, kill_smooth, color=C["dqn"] if algo == "DQN" else C["ppo"],
             linewidth=2.5, label="Kill rate medio")
    ax1.axhline(y=1.8, color=C["muted"], linestyle="--", linewidth=1, alpha=0.6, label="Target 1.8")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Kill rate medio", color=C["text"])
    ax1.set_ylim([-0.1, max(3.2, kill_smooth.max() + 0.2)])
    ax1.set_yticks([0, 1, 2, 3])

    ax2 = ax1.twinx()
    wr = win_rate_rolling(mantis, window)
    ax2.fill_between(episodes, 0, wr, alpha=0.12, color=C["success"])
    ax2.plot(episodes, wr, color=C["success"], linewidth=2, linestyle="--", label="Win rate %")
    ax2.set_ylabel("Win rate (%)", color=C["success"])
    ax2.set_ylim([-2, 105])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    total_wins = np.sum(mantis >= 3)
    ax1.set_title(f"{algo} — Kill Rate & Win Rate")
    add_subtitle(ax1, f"{total_wins} vittorie su {len(episodes)} episodi")
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_kill_win_rate.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_kill_win_rate.png")


def plot_cross_iteration(df, out, algo="PPO"):
    """v1 vs v2 vs v3 a pari posizione."""
    iters = detect_iterations(df)
    if len(iters) < 2:
        print("  [SKIP] Cross-iteration — solo 1 iterazione trovata")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"v1": C["v1"], "v2": C["v2"], "v3": C["v3"]}
    bar_width = 0.25

    windows = list(range(0, 2000, 250))
    x = np.arange(len(windows))

    for i, (label, it_df) in enumerate(iters.items()):
        offset = 2000 * (i) if label != "v1" else 0
        if label == "v2": offset = 2000
        elif label == "v3": offset = 4000
        else: offset = 0

        rates = []
        for start in windows:
            end = start + 250
            mask = ((it_df["episode"] - offset) > start) & ((it_df["episode"] - offset) <= end)
            chunk = it_df[mask]
            if len(chunk) > 0 and "mantis_killed" in chunk.columns:
                k2 = (chunk["mantis_killed"] >= 2).sum()
                rates.append(k2 / len(chunk) * 100)
            else:
                rates.append(0)

        pos = x + i * bar_width - bar_width * (len(iters) - 1) / 2
        ax.bar(pos, rates, bar_width * 0.85, color=colors.get(label, C["muted"]),
               alpha=0.75, label=label.upper(), edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-{s+250}" for s in windows], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Posizione relativa nella run (episodi)")
    ax.set_ylabel("2+ kill rate (%)")
    ax.set_title(f"{algo} — Confronto Cross-Iterazione")
    add_subtitle(ax, f"{len(iters)} iterazioni · 2+ kill rate a pari posizione")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_cross_iteration.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_cross_iteration.png")


def plot_per_instance(df, out, algo="PPO", window=30):
    if "instance" not in df.columns:
        return
    instances = sorted(df["instance"].dropna().unique())
    if len(instances) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    inst_colors = [C["inst0"], C["inst1"], C["inst2"]]

    for i, inst in enumerate(instances[:3]):
        idf = df[df["instance"] == inst]
        kills = smooth(idf["mantis_killed"].values.astype(float), window)
        color = inst_colors[i % 3]
        ax1.plot(range(len(kills)), kills, color=color, linewidth=2, label=f"Inst {int(inst)}", alpha=0.85)

    ax1.set_xlabel("Episodio (per istanza)")
    ax1.set_ylabel("Kill rate medio")
    ax1.set_title("Kill Rate per Istanza")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # Boxplot
    data = [df[df["instance"] == inst]["reward"].values for inst in instances[:3]]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.6)
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(inst_colors[j % 3])
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(C["text"])
            line.set_linewidth(1.2)
    ax2.set_xticklabels([f"Inst {int(inst)}" for inst in instances[:3]])
    ax2.set_ylabel("Distribuzione Reward")
    ax2.set_title("Performance per Istanza")
    ax2.grid(True, alpha=0.25, axis="y")

    plt.suptitle(f"{algo} — Analisi Multi-Istanza", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_per_instance.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_per_instance.png")


def plot_exploration(df, out, algo="PPO"):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values
    rewards = smooth(df["reward"].values, 30)

    ax1.fill_between(episodes, 0, rewards, alpha=0.08, color=C["reward"])
    ax1.plot(episodes, rewards, color=C["reward"], linewidth=2.2, label="Reward")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward (smoothed)", color=C["reward"])

    ax2 = ax1.twinx()
    if "epsilon" in df.columns:
        ax2.plot(episodes, df["epsilon"].values, color=C["explore"], linewidth=2.2,
                 linestyle="--", label="Epsilon")
        ax2.set_ylabel("Epsilon", color=C["explore"])
    elif "entropy" in df.columns:
        ax2.plot(episodes, smooth(df["entropy"].values, 20), color=C["explore"],
                 linewidth=2.2, linestyle="--", label="Entropy")
        ax2.set_ylabel("Entropy", color=C["explore"])
    else:
        plt.close()
        return

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"{algo} — Exploration vs Exploitation")
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_exploration.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_exploration.png")


def plot_outcome_evolution(df, out, algo="PPO"):
    """Stacked bar di kill distribution per chunk temporale."""
    if "mantis_killed" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values
    mantis = df["mantis_killed"].values

    chunk_size = max(250, len(df) // 12)
    labels, k0s, k1s, k2s, k3s = [], [], [], [], []

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = mantis[start:end]
        ep_start = int(episodes[start])
        ep_end = int(episodes[end - 1])
        labels.append(f"{ep_start}-{ep_end}")
        total = len(chunk)
        k0s.append(np.sum(chunk == 0) / total * 100)
        k1s.append(np.sum(chunk == 1) / total * 100)
        k2s.append(np.sum(chunk == 2) / total * 100)
        k3s.append(np.sum(chunk >= 3) / total * 100)

    x = np.arange(len(labels))
    w = 0.65
    ax.bar(x, k0s, w, label="0 kill", color=C["kill0"], alpha=0.75)
    ax.bar(x, k1s, w, bottom=k0s, label="1 kill", color=C["kill1"], alpha=0.75)
    b2 = np.array(k0s) + np.array(k1s)
    ax.bar(x, k2s, w, bottom=b2, label="2 kill", color=C["kill2"], alpha=0.75)
    b3 = b2 + np.array(k2s)
    ax.bar(x, k3s, w, bottom=b3, label="3 kill (vittoria)", color=C["kill3"], alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Episodi")
    ax.set_ylabel("Distribuzione (%)")
    ax.set_title(f"{algo} — Evoluzione Outcome")
    add_subtitle(ax, "Come cambia la distribuzione dei kill nel tempo")
    ax.legend(loc="upper left", ncol=4)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_outcome_evolution.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_outcome_evolution.png")


def plot_loss(df, out, algo="PPO", window=30):
    loss_col = None
    for col in ["loss", "avg_loss"]:
        if col in df.columns:
            loss_col = col
            break
    if not loss_col:
        print("  [SKIP] Loss — nessuna colonna loss trovata")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    episodes = df["episode"].values
    losses = df[loss_col].values
    valid = np.isfinite(losses)
    if np.sum(valid) == 0:
        plt.close()
        return

    ax.plot(episodes[valid], losses[valid], alpha=0.1, color=C["loss"], linewidth=0.5)
    ax.plot(episodes[valid], smooth(losses[valid], window), color=C["loss"], linewidth=2.5, label="Loss (smoothed)")

    if np.sum(valid) > 10:
        z = np.polyfit(episodes[valid], losses[valid], 1)
        p = np.poly1d(z)
        ax.plot(episodes[valid], p(episodes[valid]), "--", color=C["text"], linewidth=1.2,
                alpha=0.5, label=f"Trend (slope={z[0]:.2e})")

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Loss")
    ax.set_title(f"{algo} — Loss Convergence")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    if np.max(losses[valid]) > 100:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_loss.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_loss.png")


def plot_sample_efficiency(df, out, algo="PPO", window=50):
    if "steps" not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    color = C["dqn"] if algo == "DQN" else C["ppo"]
    episodes = df["episode"].values
    rewards = df["reward"].values
    steps = df["steps"].values

    cum_steps = np.cumsum(steps)
    cum_reward = np.cumsum(rewards)
    ax1.plot(cum_steps / 1000, cum_reward, color=color, linewidth=2)
    ax1.fill_between(cum_steps / 1000, 0, cum_reward, alpha=0.1, color=color)
    ax1.set_xlabel("Steps totali (migliaia)")
    ax1.set_ylabel("Reward cumulativo")
    ax1.set_title("Sample Efficiency", fontweight="bold", fontsize=13)
    ax1.grid(True, alpha=0.25)

    rps = rewards / np.maximum(steps, 1)
    ax2.plot(episodes, rps, alpha=0.1, color=color, linewidth=0.5)
    ax2.plot(episodes, smooth(rps, window), color=color, linewidth=2.5, label="Reward/step")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Reward / Step")
    ax2.set_title("Efficienza per Episodio", fontweight="bold", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.suptitle(f"{algo} — Sample Efficiency", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{algo.lower()}_sample_efficiency.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_sample_efficiency.png")


def plot_dashboard(df, out, algo="PPO", window=50):
    """Dashboard 3x3 per presentazione."""
    fig = plt.figure(figsize=(22, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32)
    color = C["dqn"] if algo == "DQN" else C["ppo"]
    episodes = df["episode"].values
    rewards = df["reward"].values

    # 1. Learning Curve
    ax = fig.add_subplot(gs[0, :2])
    mean, lower, upper = ci(rewards, window)
    ax.fill_between(episodes, lower, upper, alpha=0.15, color=color)
    ax.plot(episodes, mean, color=color, linewidth=2.5)
    ax.set_title("Learning Curve (95% CI)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.2)

    # 2. Stats
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    final = np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
    best = np.max(rewards)
    wins = int(np.sum(df["mantis_killed"].values >= 3)) if "mantis_killed" in df.columns else 0
    wr = wins / len(episodes) * 100
    lines = [
        f"  {algo} TRAINING SUMMARY", f"  {'─'*28}",
        f"  Episodi:     {len(episodes)}",
        f"  Best Reward: {best:.0f}",
        f"  Media finale:{final:.0f}",
        f"  Vittorie:    {wins} ({wr:.1f}%)",
    ]
    ax.text(0.05, 0.92, "\n".join(lines), transform=ax.transAxes, fontsize=12,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    # 3. Kill/Win Rate
    ax = fig.add_subplot(gs[1, 0])
    if "mantis_killed" in df.columns:
        mantis = df["mantis_killed"].values
        ax.plot(episodes, smooth(mantis.astype(float), window), color=color, linewidth=2)
        ax.axhline(y=3, color=C["success"], linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel("Kill rate")
    ax.set_title("Kill Rate", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)

    # 4. Sample Efficiency
    ax = fig.add_subplot(gs[1, 1])
    if "steps" in df.columns:
        cum_s = np.cumsum(df["steps"].values)
        cum_r = np.cumsum(rewards)
        ax.plot(cum_s / 1000, cum_r, color=color, linewidth=2)
        ax.set_xlabel("Steps (k)")
    ax.set_title("Sample Efficiency", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)

    # 5. Loss
    ax = fig.add_subplot(gs[1, 2])
    for col in ["loss", "avg_loss"]:
        if col in df.columns:
            losses = df[col].values
            valid = np.isfinite(losses)
            if np.sum(valid) > 0:
                ax.plot(episodes[valid], smooth(losses[valid], 20), color=C["loss"], linewidth=2)
            break
    ax.set_title("Loss", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)

    # 6. Outcome Distribution
    ax = fig.add_subplot(gs[2, 0])
    if "mantis_killed" in df.columns:
        mantis = df["mantis_killed"].values
        counts = [np.sum(mantis == i) for i in range(4)]
        cols = [C["kill0"], C["kill1"], C["kill2"], C["kill3"]]
        bars = ax.bar([0, 1, 2, 3], counts, color=cols, alpha=0.75, edgecolor="white")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", "1", "2", "3 (W)"])
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Distribuzione Outcome", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")

    # 7. Exploration
    ax = fig.add_subplot(gs[2, 1])
    if "epsilon" in df.columns:
        ax.plot(episodes, df["epsilon"].values, color=C["explore"], linewidth=2)
        ax.set_ylabel("Epsilon")
    elif "entropy" in df.columns:
        ax.plot(episodes, smooth(df["entropy"].values, 20), color=C["explore"], linewidth=2)
        ax.set_ylabel("Entropy")
    ax.set_title("Esplorazione", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)

    # 8. Reward Distribution
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(rewards, bins=40, color=color, alpha=0.65, edgecolor="white")
    ax.axvline(x=np.mean(rewards), color=C["loss"], linestyle="--", linewidth=2,
               label=f"Media: {np.mean(rewards):.0f}")
    ax.set_title("Distribuzione Reward", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{algo} — Training Dashboard", fontweight="bold", fontsize=20, y=0.995)
    plt.savefig(os.path.join(out, f"{algo.lower()}_dashboard.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] {algo.lower()}_dashboard.png")


def plot_comparison(df_dqn, df_ppo, out, window=50):
    """DQN vs PPO side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Learning curves
    ax = axes[0, 0]
    for df, algo, color in [(df_dqn, "DQN", C["dqn"]), (df_ppo, "PPO", C["ppo"])]:
        ep = df["episode"].values
        rew = df["reward"].values
        mean, lower, upper = ci(rew, window)
        ax.fill_between(ep, lower, upper, alpha=0.12, color=color)
        ax.plot(ep, mean, color=color, linewidth=2.5, label=algo)
    ax.set_title("Learning Curve", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Kill rate
    ax = axes[0, 1]
    for df, algo, color in [(df_dqn, "DQN", C["dqn"]), (df_ppo, "PPO", C["ppo"])]:
        if "mantis_killed" in df.columns:
            ep = df["episode"].values
            ax.plot(ep, smooth(df["mantis_killed"].values.astype(float), window),
                    color=color, linewidth=2.5, label=algo)
    ax.set_title("Kill Rate", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Sample efficiency
    ax = axes[1, 0]
    for df, algo, color in [(df_dqn, "DQN", C["dqn"]), (df_ppo, "PPO", C["ppo"])]:
        if "steps" in df.columns:
            ax.plot(np.cumsum(df["steps"].values) / 1000, np.cumsum(df["reward"].values),
                    color=color, linewidth=2.5, label=algo)
    ax.set_title("Sample Efficiency", fontweight="bold", fontsize=13)
    ax.set_xlabel("Steps (k)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Outcome distribution
    ax = axes[1, 1]
    x = np.arange(4)
    w = 0.35
    for i, (df, algo, color) in enumerate([(df_dqn, "DQN", C["dqn"]), (df_ppo, "PPO", C["ppo"])]):
        if "mantis_killed" in df.columns:
            mantis = df["mantis_killed"].values
            counts = [np.sum(mantis == k) / len(mantis) * 100 for k in range(4)]
            ax.bar(x + i * w - w / 2, counts, w * 0.9, color=color, alpha=0.7, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(["0 kill", "1 kill", "2 kill", "3 kill"])
    ax.set_ylabel("%")
    ax.set_title("Distribuzione Outcome", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("DQN vs PPO — Confronto", fontweight="bold", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dqn_vs_ppo_comparison.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] dqn_vs_ppo_comparison.png")


# ============================================================================
# PLAY (INFERENCE) PLOTS
# ============================================================================

def plot_play_results(df, out):
    """Genera tutti i grafici dall'output di play.py."""
    print("\nGenerating PLAY plots...")

    runs = df["run"].values
    kills = df["mantis_killed"].values
    boss_hp = df["boss_hp"].values
    steps = df["steps"].values
    is_win = (df["result"] == "WIN").values
    agents = df["agent"].unique() if "agent" in df.columns else ["?"]
    agent_label = agents[0] if len(agents) == 1 else " + ".join(agents)
    models = df["model"].unique() if "model" in df.columns else ["?"]

    # ─── 1. Win Rate Cumulativo ───
    fig, ax = plt.subplots(figsize=(14, 7))
    cum_wins = np.cumsum(is_win)
    cum_wr = cum_wins / np.arange(1, len(runs) + 1) * 100
    ax.plot(runs, cum_wr, color=C["success"], linewidth=2.5)
    ax.fill_between(runs, 0, cum_wr, alpha=0.12, color=C["success"])
    ax.axhline(y=50, color=C["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Run")
    ax.set_ylabel("Win Rate Cumulativo (%)")
    ax.set_ylim([-2, 105])
    ax.set_title(f"Inference — Win Rate Cumulativo")
    add_subtitle(ax, f"{agent_label} · {int(cum_wins[-1])}/{len(runs)} vittorie ({cum_wr[-1]:.1f}%)")
    ax.grid(True, alpha=0.25)
    stat_box(ax, [f"Runs: {len(runs)}", f"Vittorie: {int(cum_wins[-1])}",
                  f"Win Rate: {cum_wr[-1]:.1f}%",
                  f"Kill medi: {np.mean(kills):.2f}"])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "play_win_rate.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] play_win_rate.png")

    # ─── 2. Kill Distribution ───
    fig, ax = plt.subplots(figsize=(12, 7))
    counts = [np.sum(kills == i) for i in range(4)]
    cols = [C["kill0"], C["kill1"], C["kill2"], C["kill3"]]
    bars = ax.bar([0, 1, 2, 3], counts, color=cols, alpha=0.8, edgecolor="white", width=0.65)
    for bar, count in zip(bars, counts):
        pct = count / len(kills) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.015,
                f"{count}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 kill\n(sconfitta)", "1 kill", "2 kill", "3 kill\n(vittoria)"])
    ax.set_ylabel("Numero di run")
    ax.set_title("Inference — Distribuzione Kill")
    add_subtitle(ax, f"{len(runs)} run totali")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "play_kill_distribution.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] play_kill_distribution.png")

    # ─── 3. Boss HP Rimanente ───
    fig, ax = plt.subplots(figsize=(14, 7))
    loss_hp = boss_hp[~is_win]
    if len(loss_hp) > 0:
        ax.hist(loss_hp, bins=30, color=C["loss"], alpha=0.6, edgecolor="white", label="Sconfitte")
    ax.axvline(x=0, color=C["success"], linewidth=2, linestyle="--", label="Vittoria (HP=0)")
    ax.set_xlabel("Boss HP rimanente")
    ax.set_ylabel("Frequenza")
    ax.set_title("Inference — Boss HP alla Fine")
    add_subtitle(ax, f"Quanto manca alla vittoria nelle sconfitte")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    if len(loss_hp) > 0:
        stat_box(ax, [f"HP medio (sconfitte): {np.mean(loss_hp):.0f}",
                      f"HP mediano: {np.median(loss_hp):.0f}",
                      f"Quasi vittorie (HP<100): {np.sum(loss_hp < 100)}"])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "play_boss_hp.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] play_boss_hp.png")

    # ─── 4. Steps Distribution (Win vs Loss) ───
    fig, ax = plt.subplots(figsize=(14, 7))
    win_steps = steps[is_win]
    loss_steps = steps[~is_win]
    if len(loss_steps) > 0:
        ax.hist(loss_steps, bins=25, color=C["loss"], alpha=0.5, edgecolor="white", label="Sconfitte")
    if len(win_steps) > 0:
        ax.hist(win_steps, bins=25, color=C["success"], alpha=0.6, edgecolor="white", label="Vittorie")
    ax.set_xlabel("Steps per run")
    ax.set_ylabel("Frequenza")
    ax.set_title("Inference — Durata Run")
    add_subtitle(ax, "Distribuzione steps per vittorie e sconfitte")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    lines = [f"Steps medio totale: {np.mean(steps):.0f}"]
    if len(win_steps) > 0:
        lines.append(f"Steps medio (vitt): {np.mean(win_steps):.0f}")
        lines.append(f"Più veloce: {np.min(win_steps)}")
    stat_box(ax, lines)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "play_steps_distribution.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] play_steps_distribution.png")

    # ─── 5. Play Dashboard ───
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Win rate cumulativo
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(runs, cum_wr, color=C["success"], linewidth=2)
    ax.fill_between(runs, 0, cum_wr, alpha=0.1, color=C["success"])
    ax.axhline(y=50, color=C["muted"], linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate Cumulativo", fontweight="bold", fontsize=12)
    ax.set_ylim([-2, 105])
    ax.grid(True, alpha=0.2)

    # Kill distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.bar([0, 1, 2, 3], counts, color=cols, alpha=0.8, edgecolor="white")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "3(W)"])
    ax.set_title("Kill Distribution", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")

    # Stats
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    total_w = int(cum_wins[-1])
    slines = [
        f"  INFERENCE SUMMARY", f"  {'─'*26}",
        f"  Agente:   {agent_label}",
        f"  Modello:  {models[0] if len(models)==1 else 'multi'}",
        f"  Run:      {len(runs)}",
        f"  Vittorie: {total_w} ({cum_wr[-1]:.1f}%)",
        f"  Kill medi:{np.mean(kills):.2f}",
    ]
    if len(win_steps) > 0:
        slines.append(f"  Best run: {np.min(win_steps)} steps")
    ax.text(0.05, 0.92, "\n".join(slines), transform=ax.transAxes, fontsize=12,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    # Boss HP
    ax = fig.add_subplot(gs[1, 0])
    if len(loss_hp) > 0:
        ax.hist(loss_hp, bins=20, color=C["loss"], alpha=0.6, edgecolor="white")
    ax.set_title("Boss HP (sconfitte)", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")

    # Steps win vs loss
    ax = fig.add_subplot(gs[1, 1])
    if len(loss_steps) > 0:
        ax.hist(loss_steps, bins=20, color=C["loss"], alpha=0.5, edgecolor="white", label="Loss")
    if len(win_steps) > 0:
        ax.hist(win_steps, bins=20, color=C["success"], alpha=0.6, edgecolor="white", label="Win")
    ax.set_title("Steps (Win vs Loss)", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Kill per run (scatter)
    ax = fig.add_subplot(gs[1, 2])
    scatter_colors = [cols[min(k, 3)] for k in kills]
    ax.scatter(runs, kills, c=scatter_colors, alpha=0.6, s=25, edgecolors="white", linewidth=0.3)
    ax.plot(runs, smooth(kills.astype(float), max(5, len(runs)//10)), color=C["text"],
            linewidth=2, alpha=0.7)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Kill per Run", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Inference Dashboard — {agent_label}", fontweight="bold", fontsize=18, y=0.995)
    plt.savefig(os.path.join(out, "play_dashboard.png"), dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] play_dashboard.png")

    # ─── 6. Multi-agent comparison (se ci sono sia PPO che DQN) ───
    if "agent" in df.columns and len(agents) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        for agent_name in agents:
            adf = df[df["agent"] == agent_name]
            a_wins = (adf["result"] == "WIN").values
            a_cum_wr = np.cumsum(a_wins) / np.arange(1, len(a_wins) + 1) * 100
            color = C["ppo"] if "PPO" in agent_name.upper() else C["dqn"]
            ax1.plot(range(1, len(a_wins) + 1), a_cum_wr, color=color, linewidth=2.5, label=agent_name)

        ax1.set_title("Win Rate per Agente", fontweight="bold", fontsize=13)
        ax1.set_xlabel("Run")
        ax1.set_ylabel("Win Rate (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        x = np.arange(4)
        w = 0.35
        for i, agent_name in enumerate(agents):
            adf = df[df["agent"] == agent_name]
            ak = adf["mantis_killed"].values
            acounts = [np.sum(ak == k) / len(ak) * 100 for k in range(4)]
            color = C["ppo"] if "PPO" in agent_name.upper() else C["dqn"]
            ax2.bar(x + i * w - w / 2, acounts, w * 0.9, color=color, alpha=0.7, label=agent_name)

        ax2.set_xticks(x)
        ax2.set_xticklabels(["0 kill", "1 kill", "2 kill", "3 kill"])
        ax2.set_ylabel("%")
        ax2.set_title("Kill Distribution per Agente", fontweight="bold", fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.25, axis="y")

        fig.suptitle("PPO vs DQN — Inference Comparison", fontweight="bold", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "play_agent_comparison.png"), dpi=300, facecolor="white")
        plt.close()
        print(f"  [OK] play_agent_comparison.png")


# ============================================================================
# GENERATION ENTRY POINTS
# ============================================================================

def generate_all_training(df, out, algo, window=50):
    print(f"\nGenerating {algo} training plots...")
    plot_learning_curve(df, out, algo, window)
    plot_kill_and_win_rate(df, out, algo, window)
    plot_cross_iteration(df, out, algo)
    plot_per_instance(df, out, algo, window)
    plot_exploration(df, out, algo)
    plot_outcome_evolution(df, out, algo)
    plot_loss(df, out, algo, window // 2)
    plot_sample_efficiency(df, out, algo, window)
    plot_dashboard(df, out, algo, window)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici per training e inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python generate_plots.py --mode ppo --ppo-log training_log_ppo.csv
  python generate_plots.py --mode dqn --dqn-log training_log_dqn.csv
  python generate_plots.py --mode compare --dqn-log dqn.csv --ppo-log ppo.csv
  python generate_plots.py --mode multi --multi-log training_log_ppo.csv
  python generate_plots.py --mode play --play-log play_log.csv
  python generate_plots.py --mode presentation --ppo-log ppo.csv
        """)

    parser.add_argument("--mode", type=str, required=True,
                        choices=["dqn", "ppo", "compare", "multi", "play", "presentation"])
    parser.add_argument("--dqn-log", type=str)
    parser.add_argument("--ppo-log", type=str)
    parser.add_argument("--multi-log", type=str)
    parser.add_argument("--play-log", type=str)
    parser.add_argument("--output", type=str, default="plots")
    parser.add_argument("--window", type=int, default=50)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  TRAINING & INFERENCE VISUALIZATION")
    print(f"  Output: {args.output}/")
    print(f"{'═'*60}")

    if args.mode == "dqn":
        df = load_log(args.dqn_log)
        print(f"  Loaded: {len(df)} episodes (DQN)")
        generate_all_training(df, args.output, "DQN", args.window)

    elif args.mode == "ppo":
        df = load_log(args.ppo_log)
        print(f"  Loaded: {len(df)} episodes (PPO)")
        generate_all_training(df, args.output, "PPO", args.window)

    elif args.mode == "compare":
        if args.dqn_log and args.ppo_log:
            df_dqn = load_log(args.dqn_log)
            df_ppo = load_log(args.ppo_log)
            print(f"  DQN: {len(df_dqn)} eps | PPO: {len(df_ppo)} eps")
            plot_comparison(df_dqn, df_ppo, args.output, args.window)
            plot_dashboard(df_dqn, args.output, "DQN", args.window)
            plot_dashboard(df_ppo, args.output, "PPO", args.window)

    elif args.mode == "multi":
        df = load_log(args.multi_log)
        algo = get_algo(df)
        print(f"  Loaded: {len(df)} entries ({algo})")
        plot_per_instance(df, args.output, algo, args.window)
        plot_dashboard(df, args.output, algo, args.window)

    elif args.mode == "play":
        if not args.play_log:
            print("  [ERRORE] Serve --play-log per la modalità play")
            return
        df = load_play_log(args.play_log)
        print(f"  Loaded: {len(df)} run")
        plot_play_results(df, args.output)

    elif args.mode == "presentation":
        if args.dqn_log:
            df = load_log(args.dqn_log)
            print(f"  DQN: {len(df)} episodes")
            plot_dashboard(df, args.output, "DQN", args.window)
        if args.ppo_log:
            df = load_log(args.ppo_log)
            print(f"  PPO: {len(df)} episodes")
            plot_dashboard(df, args.output, "PPO", args.window)

    print(f"\n{'═'*60}")
    print(f"  Tutti i grafici salvati in: {args.output}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()