#!/usr/bin/env python3
"""
5G-Advanced Joint 2D DRL Agent Training Results Plotter
File: plot_training_results.py

Visualizes PPO DRL training results:
1. Episode Reward Convergence Curve
2. P99 Latency Reduction Trend (with 20ms SLA Target Line)
3. Power Consumption Trade-off (gNB Power & UE Power)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


CSV_COLUMNS = {
    "Episode",
    "EpisodeReward",
    "AvgP99LatencyMs",
    "AvgGnbPowerW",
    "AvgUePowerMw",
    "Loss",
}


def load_history(csv_filename):
    csv_path = Path(csv_filename).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training history CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = CSV_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training history CSV is missing columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Training history CSV has no episode rows: {csv_path}")

    try:
        return {
            "episodes": np.array([int(row["Episode"]) for row in rows]),
            "episode_rewards": np.array([float(row["EpisodeReward"]) for row in rows]),
            "p99_latencies": np.array([float(row["AvgP99LatencyMs"]) for row in rows]),
            "gnb_powers": np.array([float(row["AvgGnbPowerW"]) for row in rows]),
            "ue_powers": np.array([float(row["AvgUePowerMw"]) for row in rows]),
            "losses": np.array([float(row["Loss"]) for row in rows]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Training history CSV contains invalid numeric data: {csv_path}") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot previously saved PPO training history without running training."
    )
    parser.add_argument("--history-csv", default="ppo_training_history.csv")
    parser.add_argument("--output", default="plot_training_results.png")
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    if args.smooth_window <= 0:
        parser.error("--smooth-window must be greater than zero")

    return args


def main():
    args = parse_args()
    history = load_history(args.history_csv)

    print(f"📊 Reading training history from: {Path(args.history_csv).resolve()}")

    episodes = history["episodes"]
    rewards = np.array(history["episode_rewards"])
    p99_latencies = np.array(history["p99_latencies"])
    gnb_powers = np.array(history["gnb_powers"])
    ue_powers = np.array(history["ue_powers"])
    
    # Configure publication-quality Seaborn theme
    sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    fig.suptitle('5G-Advanced Joint 2D DRL (PPO) Convergence & Performance Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Subplot 1: Episode Reward Convergence Curve
    ax1 = axes[0]
    ax1.plot(episodes, rewards, color='#2b5c8f', linewidth=2.2, marker='o', markersize=4, label='Episode Reward')
    
    window = args.smooth_window
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], rewards_smooth, color='#e74c3c', linestyle='--', linewidth=2.0, label=f'Moving Average ({window} Ep)')
        
    ax1.set_title('Episode Reward Convergence (Multi-Objective Optimization)', fontsize=11, fontweight='bold', loc='left')
    ax1.set_ylabel('Reward Score', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    
    ax1.annotate(f'Initial: {rewards[0]:.1f}', xy=(episodes[0], rewards[0]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(facecolor='#2b5c8f', shrink=0.08, width=1, headwidth=5),
                 fontsize=9, fontweight='bold')
    if len(rewards) > 1:
        ax1.annotate(f'Final: {rewards[-1]:.1f}', xy=(episodes[-1], rewards[-1]),
                     xytext=(-75, -25), textcoords='offset points',
                     arrowprops=dict(facecolor='#2b5c8f', shrink=0.08, width=1, headwidth=5),
                     fontsize=9, fontweight='bold')
                 
    # Subplot 2: P99 Latency Reduction Trend
    ax2 = axes[1]
    ax2.plot(episodes, p99_latencies, color='#d35400', linewidth=2.2, marker='s', markersize=4, label='Avg P99 Latency')
    
    ax2.axhline(y=20.0, color='#27ae60', linestyle='--', linewidth=2.0, label='20 ms QoS SLA Target Bound')
    ax2.fill_between(episodes, 0, 20.0, color='#27ae60', alpha=0.1, label='SLA Compliant Zone')
    
    ax2.set_title('P99 Tail Latency Reduction Trend (Target SLA = 20.0 ms)', fontsize=11, fontweight='bold', loc='left')
    ax2.set_ylabel('P99 Latency (ms)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(max(p99_latencies) * 1.15, 30.0))
    ax2.legend(loc='upper right', frameon=True)
    
    # Subplot 3: Power Consumption Trade-off
    ax3 = axes[2]
    color_gnb = '#2980b9'
    color_ue = '#8e44ad'
    
    line1 = ax3.plot(episodes, gnb_powers, color=color_gnb, linewidth=2.2, marker='^', markersize=4, label='gNB Power (W)')
    ax3.set_ylabel('gNB Avg Power (W)', color=color_gnb, fontsize=10, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color_gnb)
    
    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(episodes, ue_powers, color=color_ue, linewidth=2.2, marker='v', markersize=4, label='UE Power (mW)')
    ax3_twin.set_ylabel('UE Avg Power (mW)', color=color_ue, fontsize=10, fontweight='bold')
    ax3_twin.tick_params(axis='y', labelcolor=color_ue)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right', frameon=True)
    
    ax3.set_title('gNB vs UE Power Consumption Trade-off Balancing', fontsize=11, fontweight='bold', loc='left')
    ax3.set_xlabel('Training Episodes', fontsize=10, fontweight='bold')
    
    sns.despine(fig=fig, right=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_png = Path(args.output).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training results plot saved successfully to: {output_png}")


if __name__ == "__main__":
    main()
