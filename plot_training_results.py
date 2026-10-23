#!/usr/bin/env python3
"""Plot PPO training convergence, optimization diagnostics, QoS, and energy."""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


REQUIRED_COLUMNS = {
    "Episode",
    "EpisodeReward",
    "AvgP99LatencyMs",
    "AvgGnbPowerW",
    "AvgUePowerMw",
    "Loss",
}

OPTIONAL_COLUMNS = {
    "reward_gnb": "RewardGnbPower",
    "reward_ue": "RewardUePower",
    "reward_latency": "RewardLatency",
    "reward_backlog": "RewardBacklog",
    "policy_loss": "PolicyLoss",
    "value_loss": "ValueLoss",
    "entropy": "Entropy",
    "approx_kl": "ApproxKl",
    "clip_fraction": "ClipFraction",
    "qos_satisfaction_pct": "QosSatisfactionRatePct",
    "sla_violation_rate": "SlaViolationRate",
    "qos_target_ms": "QosTargetMs",
    "gnb_efficiency": "GnbEnergyEfficiencyMbitPerJ",
    "network_efficiency": "NetworkEnergyEfficiencyMbitPerJ",
    "gnb_kwh_per_tb": "GnbEnergyKwhPerTb",
    "offered_rate": "AvgOfferedRateMbps",
    "delivered_rate": "AvgDeliveredRateMbps",
    "backlog": "AvgBacklogBytes",
    "max_ue_backlog": "AvgMaxUeBacklogBytes",
}


def _float_column(rows, column, default=np.nan):
    values = []
    for row in rows:
        raw_value = row.get(column, "")
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, dtype=float)


def load_history(csv_filename):
    csv_path = Path(csv_filename).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training history CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training history CSV is missing columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Training history CSV has no episode rows: {csv_path}")

    try:
        history = {
            "episodes": np.array([int(row["Episode"]) for row in rows]),
            "episode_rewards": _float_column(rows, "EpisodeReward"),
            "p99_latencies": _float_column(rows, "AvgP99LatencyMs"),
            "gnb_powers": _float_column(rows, "AvgGnbPowerW"),
            "ue_powers": _float_column(rows, "AvgUePowerMw"),
            "losses": _float_column(rows, "Loss"),
        }
        for key, column in OPTIONAL_COLUMNS.items():
            history[key] = _float_column(rows, column)
        return history
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Training history CSV contains invalid numeric data: {csv_path}") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot previously saved PPO training history without running training."
    )
    parser.add_argument("--history-csv", default="ppo_training_history.csv")
    parser.add_argument("--output", default="plot_training_results.png")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--qos-target-ms",
        type=float,
        default=None,
        help="Override the CSV QoS target; legacy CSV files default to 20 ms",
    )
    args = parser.parse_args()

    if args.smooth_window <= 0:
        parser.error("--smooth-window must be greater than zero")
    if args.qos_target_ms is not None and args.qos_target_ms <= 0:
        parser.error("--qos-target-ms must be greater than zero")

    return args


def _has_data(values):
    return np.any(np.isfinite(values))


def _plot_series(ax, episodes, values, label, color, marker=None, alpha=0.35):
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    ax.plot(
        episodes[finite],
        values[finite],
        color=color,
        linewidth=1.1,
        marker=marker,
        markersize=3,
        alpha=alpha,
        label=f"{label} (raw)",
    )


def _plot_smoothed(ax, episodes, values, label, color, window, linestyle="-"):
    finite = np.isfinite(values)
    x = episodes[finite]
    y = values[finite]
    if len(y) == 0:
        return
    effective_window = min(window, len(y))
    if effective_window == 1:
        smooth = y
        smooth_x = x
    else:
        kernel = np.ones(effective_window) / effective_window
        smooth = np.convolve(y, kernel, mode="valid")
        smooth_x = x[effective_window - 1:]
    ax.plot(
        smooth_x,
        smooth,
        color=color,
        linewidth=2.1,
        linestyle=linestyle,
        label=f"{label} ({effective_window}-ep mean)",
    )


def _combined_legend(ax, twin=None, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        handles += twin_handles
        labels += twin_labels
    if handles:
        ax.legend(handles, labels, loc=loc, fontsize=8, frameon=True)


def main():
    args = parse_args()
    history = load_history(args.history_csv)
    print(f"📊 Reading training history from: {Path(args.history_csv).resolve()}")

    episodes = history["episodes"]
    window = args.smooth_window
    csv_targets = history["qos_target_ms"]
    if args.qos_target_ms is not None:
        qos_targets = np.full(len(episodes), args.qos_target_ms, dtype=float)
    elif _has_data(csv_targets):
        qos_targets = csv_targets
    else:
        qos_targets = np.full(len(episodes), 20.0, dtype=float)

    qos_satisfaction = history["qos_satisfaction_pct"].copy()
    if not _has_data(qos_satisfaction) and _has_data(history["sla_violation_rate"]):
        qos_satisfaction = 100.0 * (1.0 - history["sla_violation_rate"])

    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")
    fig, axes = plt.subplots(4, 2, figsize=(15, 19), sharex=True)
    fig.suptitle(
        "5G-Advanced Joint 2D PPO Training Diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    _plot_series(ax, episodes, rewards, "Episode reward", "#2b5c8f", "o")
    _plot_smoothed(ax, episodes, rewards, "Episode reward", "#2b5c8f", window)
    ax.set_title("Reward Convergence", fontweight="bold", loc="left")
    ax.set_ylabel("Episode reward")
    _combined_legend(ax)

    ax = axes[0, 1]
    reward_components = [
        ("gNB power", history["reward_gnb"], "#2980b9"),
        ("UE power", history["reward_ue"], "#8e44ad"),
        ("P99 latency", history["reward_latency"], "#d35400"),
        ("Backlog", history["reward_backlog"], "#16a085"),
    ]
    for label, values, color in reward_components:
        _plot_smoothed(ax, episodes, values, label, color, window)
    if not any(_has_data(values) for _, values, _ in reward_components):
        ax.text(
            0.5,
            0.5,
            "Reward components unavailable in legacy CSV",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Reward Decomposition", fontweight="bold", loc="left")
    ax.set_ylabel("Episode reward contribution")
    _combined_legend(ax)

    ax = axes[1, 0]
    _plot_smoothed(
        ax, episodes, history["losses"], "Total loss", "#2c3e50", window
    )
    _plot_smoothed(
        ax, episodes, history["policy_loss"], "Policy loss", "#c0392b", window
    )
    _plot_smoothed(
        ax, episodes, history["value_loss"], "Value loss", "#27ae60", window
    )
    entropy_ax = ax.twinx()
    _plot_smoothed(
        entropy_ax,
        episodes,
        history["entropy"],
        "Policy entropy",
        "#f39c12",
        window,
        linestyle="--",
    )
    ax.set_title("PPO Loss Decomposition", fontweight="bold", loc="left")
    ax.set_ylabel("Loss")
    entropy_ax.set_ylabel("Entropy", color="#f39c12")
    entropy_ax.tick_params(axis="y", labelcolor="#f39c12")
    _combined_legend(ax, entropy_ax)

    ax = axes[1, 1]
    p99 = history["p99_latencies"]
    _plot_series(ax, episodes, p99, "Average P99 latency", "#d35400", "s")
    _plot_smoothed(
        ax, episodes, p99, "Average P99 latency", "#d35400", window
    )
    ax.plot(
        episodes,
        qos_targets,
        color="#1e8449",
        linestyle="--",
        linewidth=2.0,
        label="QoS target",
    )
    qos_ax = ax.twinx()
    if _has_data(qos_satisfaction):
        qos_ax.plot(
            episodes,
            qos_satisfaction,
            color="#2471a3",
            linewidth=1.8,
            label="QoS satisfaction",
        )
    ax.set_title("Tail Latency and QoS Compliance", fontweight="bold", loc="left")
    ax.set_ylabel("P99 latency (ms)")
    qos_ax.set_ylabel("QoS satisfaction (%)", color="#2471a3")
    qos_ax.set_ylim(0.0, 105.0)
    qos_ax.tick_params(axis="y", labelcolor="#2471a3")
    _combined_legend(ax, qos_ax)

    ax = axes[2, 0]
    _plot_smoothed(
        ax,
        episodes,
        history["gnb_efficiency"],
        "gNB efficiency",
        "#2980b9",
        window,
    )
    _plot_smoothed(
        ax,
        episodes,
        history["network_efficiency"],
        "Network efficiency",
        "#16a085",
        window,
    )
    kwh_ax = ax.twinx()
    _plot_smoothed(
        kwh_ax,
        episodes,
        history["gnb_kwh_per_tb"],
        "gNB energy intensity",
        "#7f8c8d",
        window,
        linestyle="--",
    )
    ax.set_title("Energy Efficiency", fontweight="bold", loc="left")
    ax.set_ylabel("Delivered traffic (Mbit/J)")
    kwh_ax.set_ylabel("gNB energy intensity (kWh/TB)", color="#7f8c8d")
    kwh_ax.tick_params(axis="y", labelcolor="#7f8c8d")
    _combined_legend(ax, kwh_ax)

    ax = axes[2, 1]
    _plot_smoothed(
        ax, episodes, history["gnb_powers"], "gNB power", "#2980b9", window
    )
    ue_ax = ax.twinx()
    _plot_smoothed(
        ue_ax, episodes, history["ue_powers"], "UE power", "#8e44ad", window
    )
    ax.set_title("gNB and UE Power", fontweight="bold", loc="left")
    ax.set_ylabel("Average gNB power (W)", color="#2980b9")
    ue_ax.set_ylabel("Average UE power (mW/UE)", color="#8e44ad")
    ax.tick_params(axis="y", labelcolor="#2980b9")
    ue_ax.tick_params(axis="y", labelcolor="#8e44ad")
    _combined_legend(ax, ue_ax)

    ax = axes[3, 0]
    _plot_series(
        ax, episodes, history["offered_rate"], "Offered load",
        "#c0392b", "o", alpha=0.25,
    )
    _plot_smoothed(
        ax, episodes, history["offered_rate"], "Offered load",
        "#c0392b", window,
    )
    _plot_series(
        ax, episodes, history["delivered_rate"], "Delivered throughput",
        "#2471a3", "s", alpha=0.25,
    )
    _plot_smoothed(
        ax, episodes, history["delivered_rate"], "Delivered throughput",
        "#2471a3", window,
    )
    ax.set_title("Offered Load and Delivered Throughput", fontweight="bold", loc="left")
    ax.set_ylabel("Aggregate traffic (Mbps)")
    _combined_legend(ax)

    ax = axes[3, 1]
    delivery_ratio = np.divide(
        100.0 * history["delivered_rate"],
        history["offered_rate"],
        out=np.full_like(history["delivered_rate"], np.nan),
        where=history["offered_rate"] > 0.0,
    )
    _plot_smoothed(
        ax, episodes, delivery_ratio, "Delivery ratio", "#1e8449", window,
    )
    backlog_ax = ax.twinx()
    _plot_smoothed(
        backlog_ax, episodes, history["backlog"], "RLC backlog",
        "#7d3c98", window, linestyle="--",
    )
    ax.set_title("Traffic Delivery and Queue", fontweight="bold", loc="left")
    ax.set_ylabel("Delivered / offered (%)", color="#1e8449")
    backlog_ax.set_ylabel("Average RLC backlog (bytes)", color="#7d3c98")
    ax.tick_params(axis="y", labelcolor="#1e8449")
    backlog_ax.tick_params(axis="y", labelcolor="#7d3c98")
    _combined_legend(ax, backlog_ax)

    for ax in axes[3, :]:
        ax.set_xlabel("Training episode")
    sns.despine(fig=fig, right=False)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_png = Path(args.output).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Training diagnostics plot saved to: {output_png}")


if __name__ == "__main__":
    main()
