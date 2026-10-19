#!/usr/bin/env python3
"""
5G-Advanced Joint 2D DRL Agent - Deterministic Evaluation Script
File: eval_drl_agent.py
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

from train_drl_agent import PPOAgent
from ns3_environment import OBSERVATION_DIM, Ns3MultiUserEnv

def evaluate_deterministic_policy(
    model_path="ppo_5g_joint2d_model.pt",
    ns3_path="./ns3",
    num_seeds=10,
    num_ues=10,
    udp_rate=2.0,
    enable_burst_traffic=False,
    synchronize_burst_traffic=False,
    udp_burst_rate=3.0,
    burst_mean_on_ms=80.0,
    burst_mean_off_ms=120.0,
    gnb_tx_power_dbm=30.0,
    gnb_p0_active_w=150.0,
    gnb_p_micro_sleep_w=15.0,
    gnb_p_tx_max_w=40.0,
    gnb_delta_p=4.7,
    gnb_narrow_capacity_mbps=100.0,
    gnb_wide_capacity_mbps=100.0,
    gnb_wide_active_power_overhead_w=0.0,
    gnb_wide_sleep_power_overhead_w=0.0,
    steps_per_ep=250,
    qos_target_ms=30.0,
    warmup_ms=200.0,
    max_active_ms=30.0,
    initial_active_ms=30.0,
    enable_mock=False,
    csv_out="deterministic_evaluation_results.csv",
    ns3_csv_out="multi_user_rl_step.csv",
):
    print("=" * 110)
    print("🎯 開始執行 5G-Advanced Joint 2D DRL 模型固定策略 (Deterministic Evaluation) 跨種子評估...")
    print(f"ℹ️  模擬模式: {'Python Mock' if enable_mock else '真實 C++ ns-3 引擎 (' + ns3_path + ')'}")
    print("=" * 110)

    episode_sim_time = steps_per_ep * 0.040 + warmup_ms / 1000.0 + 0.1
    csv_out = os.path.abspath(csv_out)
    ns3_csv_out = os.path.abspath(ns3_csv_out)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(ns3_csv_out), exist_ok=True)

    env = Ns3MultiUserEnv(
        ns3_path=ns3_path,
        num_ues=num_ues,
        udp_rate=udp_rate,
        enable_burst_traffic=enable_burst_traffic,
        synchronize_burst_traffic=synchronize_burst_traffic,
        udp_burst_rate=udp_burst_rate,
        burst_mean_on_ms=burst_mean_on_ms,
        burst_mean_off_ms=burst_mean_off_ms,
        gnb_tx_power_dbm=gnb_tx_power_dbm,
        gnb_p0_active_w=gnb_p0_active_w,
        gnb_p_micro_sleep_w=gnb_p_micro_sleep_w,
        gnb_p_tx_max_w=gnb_p_tx_max_w,
        gnb_delta_p=gnb_delta_p,
        gnb_narrow_capacity_mbps=gnb_narrow_capacity_mbps,
        gnb_wide_capacity_mbps=gnb_wide_capacity_mbps,
        gnb_wide_active_power_overhead_w=gnb_wide_active_power_overhead_w,
        gnb_wide_sleep_power_overhead_w=gnb_wide_sleep_power_overhead_w,
        sim_time=episode_sim_time,
        enable_mock=enable_mock,
        csv_filename=ns3_csv_out,
        qos_target_ms=qos_target_ms,
        controller_warmup_ms=warmup_ms,
        controller_max_active_ms=max_active_ms,
        controller_initial_active_ms=initial_active_ms,
    )
    env.max_steps = steps_per_ep

    agent = PPOAgent(state_dim=OBSERVATION_DIM, action_dim=10, hidden_dim=64)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model weights not found: {model_path}")
    try:
        agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
    except RuntimeError as exc:
        raise RuntimeError(
            f"model {model_path!r} is incompatible with the current "
            f"{OBSERVATION_DIM}D observation space; retrain the policy"
        ) from exc
    agent.policy.eval()
    print(f"✅ 成功載入模型權重: {model_path}")
    print(
        f"🎯 SLA={qos_target_ms:.1f}ms | Active Window=10–{max_active_ms:.0f}ms "
        f"(initial {initial_active_ms:.0f}ms) | "
        f"gNB Tx={gnb_tx_power_dbm:.1f}dBm | warm-up={warmup_ms:.0f}ms\n"
    )

    eval_results = []
    seeds = [101 + i for i in range(num_seeds)]
    load_threshold_mbps = num_ues * (udp_rate + 0.5 * udp_burst_rate)

    print(f"{'Seed':<6} | {'Episode Reward':<16} | {'Avg P99 Latency (ms)':<22} | {'Avg gNB Power (W)':<18} | {'Avg UE Power (mW)':<18} | {'SLA Violation Rate':<20}")
    print("-" * 110)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs, _ = env.reset(seed=seed)
        
        ep_reward = 0.0
        p99_latencies = []
        gnb_powers = []
        ue_powers = []
        backlogs = []
        max_ue_backlogs = []
        offered_rates = []
        delivered_rates = []
        avg_cqis = []
        p10_cqis = []
        min_cqis = []
        avg_dl_mcs_values = []
        active_windows = []
        bwp_states = []
        wide_bwp_steps = 0
        sla_violations = 0

        for step in range(steps_per_ep):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                logits, _ = agent.policy(state_tensor)
                action = torch.argmax(logits, dim=-1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            p99_latencies.append(info.get("p99_latency_ms", 0.0))
            gnb_powers.append(info.get("gnb_power_w", 0.0))
            ue_powers.append(info.get("ue_power_mw", 0.0))
            backlogs.append(info.get("backlog_bytes", 0.0))
            max_ue_backlogs.append(info.get("max_ue_backlog_bytes", 0.0))
            offered_rates.append(info.get("offered_rate_mbps", 0.0))
            delivered_rates.append(info.get("delivered_rate_mbps", 0.0))
            avg_cqis.append(info.get("avg_cqi", 0.0))
            p10_cqis.append(info.get("p10_cqi", 0.0))
            min_cqis.append(info.get("min_cqi", 0.0))
            avg_dl_mcs_values.append(info.get("avg_dl_mcs", 0.0))
            active_windows.append(info.get("w_active_ms", 0.0))
            bwp_states.append(int(info.get("bwp_state", 0)))
            wide_bwp_steps += int(info.get("bwp_state", 0) == 1)

            if info.get("p99_latency_ms", 0.0) > qos_target_ms:
                sla_violations += 1

            obs = next_obs
            if terminated or truncated:
                break

        avg_p99 = np.mean(p99_latencies)
        avg_gnb = np.mean(gnb_powers)
        avg_ue = np.mean(ue_powers)
        avg_backlog = np.mean(backlogs)
        final_backlog = backlogs[-1]
        avg_active_window = np.mean(active_windows)
        wide_bwp_rate = (wide_bwp_steps / max(1, len(p99_latencies))) * 100.0
        sla_viol_rate = (sla_violations / max(1, len(p99_latencies))) * 100.0
        control_interval_sec = 0.040
        total_delivered_mbits = float(np.sum(delivered_rates) * control_interval_sec)
        gnb_control_energy_j = float(np.sum(gnb_powers) * control_interval_sec)
        gnb_energy_efficiency = (
            total_delivered_mbits / gnb_control_energy_j
            if gnb_control_energy_j > 0.0 else np.nan
        )
        gnb_energy_kwh_per_tb = (
            (8.0e6 / 3.6e6) / gnb_energy_efficiency
            if gnb_energy_efficiency > 0.0 else np.nan
        )
        ue_control_energy_j_all_ues = float(
            np.sum(ue_powers) / 1000.0 * control_interval_sec * num_ues
        )
        offered_array = np.asarray(offered_rates)
        active_array = np.asarray(active_windows)
        bwp_array = np.asarray(bwp_states)
        low_load = offered_array < load_threshold_mbps
        high_load = ~low_load

        def conditional_mean(values, mask):
            return float(np.mean(values[mask])) if np.any(mask) else np.nan

        eval_results.append({
            "Seed": seed,
            "EpisodeReward": ep_reward,
            "AvgP99LatencyMs": avg_p99,
            "AvgGnbPowerW": avg_gnb,
            "AvgUePowerMw": avg_ue,
            "AvgBacklogBytes": avg_backlog,
            "AvgMaxUeBacklogBytes": np.mean(max_ue_backlogs),
            "AvgOfferedRateMbps": np.mean(offered_rates),
            "AvgDeliveredRateMbps": np.mean(delivered_rates),
            "TotalDeliveredMbits": total_delivered_mbits,
            "GnbControlEnergyJ": gnb_control_energy_j,
            "GnbEnergyEfficiencyMbitPerJ": gnb_energy_efficiency,
            "GnbEnergyKwhPerTb": gnb_energy_kwh_per_tb,
            "UeControlEnergyJAllUes": ue_control_energy_j_all_ues,
            "MinOfferedRateMbps": np.min(offered_array),
            "MaxOfferedRateMbps": np.max(offered_array),
            "StdOfferedRateMbps": np.std(offered_array),
            "AvgCqi": np.mean(avg_cqis),
            "AvgP10Cqi": np.mean(p10_cqis),
            "AvgMinCqi": np.mean(min_cqis),
            "AvgDlMcs": np.mean(avg_dl_mcs_values),
            "FinalBacklogBytes": final_backlog,
            "AvgActiveWindowMs": avg_active_window,
            "WideBwpRatePct": wide_bwp_rate,
            "LowLoadThresholdMbps": load_threshold_mbps,
            "LowLoadWideBwpRatePct": 100.0 * conditional_mean(bwp_array, low_load),
            "HighLoadWideBwpRatePct": 100.0 * conditional_mean(bwp_array, high_load),
            "LowLoadAvgActiveWindowMs": conditional_mean(active_array, low_load),
            "HighLoadAvgActiveWindowMs": conditional_mean(active_array, high_load),
            "SlaTargetMs": qos_target_ms,
            "SLA_Violation_Rate_Pct": sla_viol_rate,
            "SLA_Compliance_Rate_Pct": 100.0 - sla_viol_rate
        })

        print(f"{seed:<6} | {ep_reward:<16.2f} | {avg_p99:<22.2f} | {avg_gnb:<18.2f} | {avg_ue:<18.2f} | {sla_viol_rate:<20.1f} %")

    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(csv_out, index=False)

    print("-" * 110)
    print("\n📊 固定策略評估統計總結 (Deterministic Evaluation Summary):")
    print(f"  • 平均 P99 尾端延遲 (ms)     : {df_eval['AvgP99LatencyMs'].mean():.2f} ms ± {df_eval['AvgP99LatencyMs'].std():.2f} ms")
    print(f"  • 平均 gNB 基地台功耗 (W)    : {df_eval['AvgGnbPowerW'].mean():.2f} W ± {df_eval['AvgGnbPowerW'].std():.2f} W")
    print(f"  • gNB 控制期間總能量 (J)    : {df_eval['GnbControlEnergyJ'].mean():.2f} J")
    print(f"  • gNB 能源效率 (Mbit/J)      : {df_eval['GnbEnergyEfficiencyMbitPerJ'].mean():.4f}")
    print(f"  • gNB 正規化能耗 (kWh/TB)    : {df_eval['GnbEnergyKwhPerTb'].mean():.2f}")
    print(f"  • 平均 UE 手機端功耗 (mW)   : {df_eval['AvgUePowerMw'].mean():.2f} mW ± {df_eval['AvgUePowerMw'].std():.2f} mW")
    print(f"  • 平均 RLC backlog (bytes)   : {df_eval['AvgBacklogBytes'].mean():.1f}")
    print(f"  • 平均 Active Window (ms)    : {df_eval['AvgActiveWindowMs'].mean():.2f}")
    print(f"  • 平均 Wide-BWP 使用率 (%)   : {df_eval['WideBwpRatePct'].mean():.1f}%")
    print(f"  • 平均 {qos_target_ms:g}ms SLA 違規率 (%) : {df_eval['SLA_Violation_Rate_Pct'].mean():.1f}%")
    print(f"\n💾 評估結果數據已匯出至: {csv_out}\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy with deterministic actions."
    )
    parser.add_argument("--model-path", default="ppo_5g_joint2d_model.pt")
    parser.add_argument("--ns3-path", default="./ns3")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-ues", type=int, default=10)
    parser.add_argument("--udp-rate", type=float, default=2.0)
    parser.add_argument("--enable-burst-traffic", action="store_true")
    parser.add_argument("--synchronize-burst-traffic", action="store_true")
    parser.add_argument("--udp-burst-rate", type=float, default=3.0)
    parser.add_argument("--burst-mean-on-ms", type=float, default=80.0)
    parser.add_argument("--burst-mean-off-ms", type=float, default=120.0)
    parser.add_argument("--gnb-tx-power-dbm", type=float, default=30.0)
    parser.add_argument("--gnb-p0-active-w", type=float, default=150.0)
    parser.add_argument("--gnb-p-micro-sleep-w", type=float, default=15.0)
    parser.add_argument("--gnb-p-tx-max-w", type=float, default=40.0)
    parser.add_argument("--gnb-delta-p", type=float, default=4.7)
    parser.add_argument("--gnb-narrow-capacity-mbps", type=float, default=100.0)
    parser.add_argument("--gnb-wide-capacity-mbps", type=float, default=100.0)
    parser.add_argument("--gnb-wide-active-power-overhead-w", type=float, default=0.0)
    parser.add_argument("--gnb-wide-sleep-power-overhead-w", type=float, default=0.0)
    parser.add_argument("--steps-per-ep", type=int, default=250)
    parser.add_argument("--qos-target-ms", type=float, default=30.0)
    parser.add_argument("--warmup-ms", type=float, default=200.0)
    parser.add_argument("--max-active-ms", type=float, default=30.0)
    parser.add_argument("--initial-active-ms", type=float, default=30.0)
    parser.add_argument("--csv-out", default="deterministic_evaluation_results.csv")
    parser.add_argument("--ns3-csv-out", default="multi_user_rl_step.csv")
    parser.add_argument("--enable-mock", action="store_true")
    args = parser.parse_args()

    if args.num_seeds <= 0:
        parser.error("--num-seeds must be greater than zero")
    if args.num_ues <= 0:
        parser.error("--num-ues must be greater than zero")
    if args.udp_rate <= 0:
        parser.error("--udp-rate must be greater than zero")
    if args.enable_burst_traffic and args.udp_burst_rate <= 0:
        parser.error("--udp-burst-rate must be greater than zero")
    if args.enable_burst_traffic and (
        args.burst_mean_on_ms <= 0 or args.burst_mean_off_ms <= 0
    ):
        parser.error("burst On/Off means must be greater than zero")
    gnb_nonnegative = [
        args.gnb_p0_active_w,
        args.gnb_p_micro_sleep_w,
        args.gnb_p_tx_max_w,
        args.gnb_delta_p,
        args.gnb_wide_active_power_overhead_w,
        args.gnb_wide_sleep_power_overhead_w,
    ]
    if any(value < 0.0 for value in gnb_nonnegative):
        parser.error("gNB energy-model power parameters cannot be negative")
    if args.gnb_narrow_capacity_mbps <= 0.0 or args.gnb_wide_capacity_mbps <= 0.0:
        parser.error("gNB energy-model capacities must be greater than zero")
    if args.steps_per_ep <= 0:
        parser.error("--steps-per-ep must be greater than zero")
    if args.qos_target_ms <= 0:
        parser.error("--qos-target-ms must be greater than zero")
    if args.warmup_ms < 0:
        parser.error("--warmup-ms cannot be negative")
    if not 10.0 < args.max_active_ms <= 40.0:
        parser.error("--max-active-ms must be greater than 10 and at most 40")
    if not 10.0 <= args.initial_active_ms <= args.max_active_ms:
        parser.error("--initial-active-ms must be between 10 and --max-active-ms")

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_deterministic_policy(
        model_path=cli_args.model_path,
        ns3_path=cli_args.ns3_path,
        num_seeds=cli_args.num_seeds,
        num_ues=cli_args.num_ues,
        udp_rate=cli_args.udp_rate,
        enable_burst_traffic=cli_args.enable_burst_traffic,
        synchronize_burst_traffic=cli_args.synchronize_burst_traffic,
        udp_burst_rate=cli_args.udp_burst_rate,
        burst_mean_on_ms=cli_args.burst_mean_on_ms,
        burst_mean_off_ms=cli_args.burst_mean_off_ms,
        gnb_tx_power_dbm=cli_args.gnb_tx_power_dbm,
        gnb_p0_active_w=cli_args.gnb_p0_active_w,
        gnb_p_micro_sleep_w=cli_args.gnb_p_micro_sleep_w,
        gnb_p_tx_max_w=cli_args.gnb_p_tx_max_w,
        gnb_delta_p=cli_args.gnb_delta_p,
        gnb_narrow_capacity_mbps=cli_args.gnb_narrow_capacity_mbps,
        gnb_wide_capacity_mbps=cli_args.gnb_wide_capacity_mbps,
        gnb_wide_active_power_overhead_w=cli_args.gnb_wide_active_power_overhead_w,
        gnb_wide_sleep_power_overhead_w=cli_args.gnb_wide_sleep_power_overhead_w,
        steps_per_ep=cli_args.steps_per_ep,
        qos_target_ms=cli_args.qos_target_ms,
        warmup_ms=cli_args.warmup_ms,
        max_active_ms=cli_args.max_active_ms,
        initial_active_ms=cli_args.initial_active_ms,
        enable_mock=cli_args.enable_mock,
        csv_out=cli_args.csv_out,
        ns3_csv_out=cli_args.ns3_csv_out,
    )
