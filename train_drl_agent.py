#!/usr/bin/env python3
"""
5G-Advanced Joint 2D DRL Agent - PyTorch PPO Training Main Script
File: train_drl_agent.py

Features:
  - Real-time Socket IPC Co-Simulation with ns-3
  - Continuous Stateful Physical Layer Dynamics
  - Actor-Critic Neural Network Architecture (Categorical Policy for Discrete 10D Action Space)
  - Generalized Advantage Estimation (GAE-Lambda)
  - Clipped Surrogate Objective with Entropy Regularization
  - Multi-Objective Optimization Tracking: gNB Power, UE Power, P99 Delay SLA, Jain's Fairness
"""

import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ns3_environment import OBSERVATION_DIM, Ns3MultiUserEnv

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=OBSERVATION_DIM, action_dim=10, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(self, state, action=None):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


class PPOAgent:
    def __init__(self, 
                 state_dim=OBSERVATION_DIM,
                 action_dim=10, 
                 hidden_dim=64,
                 lr=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_eps=0.2, 
                 c_val=0.5, 
                 c_ent=0.01, 
                 ppo_epochs=10, 
                 batch_size=32):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def update(self):
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        gae = 0.0
        
        values_extended = values + [0.0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1.0 - float(dones[t])) - values_extended[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = permutation[start_idx : start_idx + self.batch_size]

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(b_states, b_actions)

                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values, b_returns)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.c_val * value_loss + self.c_ent * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.buffer.clear()
        return total_loss.item()


def train_ppo_agent(
    num_episodes=10,
    max_steps_per_episode=50,
    num_ues=10,
    udp_rate=2.0,
    enable_burst_traffic=False,
    synchronize_burst_traffic=False,
    udp_burst_rate=3.0,
    burst_mean_on_ms=80.0,
    burst_mean_off_ms=120.0,
    gnb_tx_power_dbm=30.0,
    gnb_energy_bs_category="BsCat1",
    gnb_energy_ref_config="Set1",
    gnb_energy_eta_mode="Dual",
    gnb_energy_antenna_ratio=0.4,
    gnb_energy_power_unit_w=1.0,
    ue_energy_power_unit_mw=1.0,
    history_csv="ppo_training_history.csv",
    ns3_csv="multi_user_rl_step.csv",
    model_out="ppo_5g_joint2d_model.pt",
    torchscript_out="ppo_5g_joint2d_torchscript.pt",
    seed=42,
    qos_target_ms=30.0,
    warmup_ms=200.0,
    max_active_ms=30.0,
    initial_active_ms=30.0,
    weight_gnb_power=0.15,
    weight_ue_power=0.15,
    weight_latency=0.45,
    weight_backlog=0.25,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    episode_sim_time = max_steps_per_episode * 0.040 + warmup_ms / 1000.0 + 0.1
    ns3_csv = os.path.abspath(ns3_csv)
    os.makedirs(os.path.dirname(ns3_csv), exist_ok=True)
    print("=" * 110)
    print("🚀 開始執行 5G-Advanced Joint 2D DRL (PyTorch PPO) 即時 Socket IPC 連續狀態訓練程序...")
    print(f"📊 設定: {num_episodes} 回合 | 每回合 {max_steps_per_episode} 步 (40ms/步, {episode_sim_time:.1f}s/ns-3 process) | enable_mock = False")
    traffic_label = (
        f"base {udp_rate:g} + burst {udp_burst_rate:g} Mbps/UE "
        f"({('synchronized' if synchronize_burst_traffic else 'independent')}, "
        f"{burst_mean_on_ms:g}/{burst_mean_off_ms:g} ms On/Off)"
        if enable_burst_traffic else f"constant {udp_rate:g} Mbps/UE"
    )
    print(f"📡 流量: {traffic_label}")
    print(f"🎯 SLA={qos_target_ms:.1f}ms | Active Window=10–{max_active_ms:.0f}ms (initial {initial_active_ms:.0f}ms) | gNB Tx={gnb_tx_power_dbm:.1f}dBm | warm-up={warmup_ms:.0f}ms | seed={seed}")
    print(
        "🔋 Energy: 5G-LENA TR 38.864/38.840 | "
        f"{gnb_energy_bs_category}/{gnb_energy_ref_config}/{gnb_energy_eta_mode} | "
        f"gNB unit={gnb_energy_power_unit_w:g}W | UE unit={ue_energy_power_unit_mw:g}mW"
    )
    print(f"⚡ 運算硬體裝置: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 110)

    env = Ns3MultiUserEnv(
        num_ues=num_ues,
        udp_rate=udp_rate,
        enable_burst_traffic=enable_burst_traffic,
        synchronize_burst_traffic=synchronize_burst_traffic,
        udp_burst_rate=udp_burst_rate,
        burst_mean_on_ms=burst_mean_on_ms,
        burst_mean_off_ms=burst_mean_off_ms,
        gnb_tx_power_dbm=gnb_tx_power_dbm,
        gnb_energy_bs_category=gnb_energy_bs_category,
        gnb_energy_ref_config=gnb_energy_ref_config,
        gnb_energy_eta_mode=gnb_energy_eta_mode,
        gnb_energy_antenna_ratio=gnb_energy_antenna_ratio,
        gnb_energy_power_unit_w=gnb_energy_power_unit_w,
        ue_energy_power_unit_mw=ue_energy_power_unit_mw,
        enable_mock=False,
        sim_time=episode_sim_time,
        csv_filename=ns3_csv,
        qos_target_ms=qos_target_ms,
        controller_warmup_ms=warmup_ms,
        controller_max_active_ms=max_active_ms,
        controller_initial_active_ms=initial_active_ms,
        rng_seed=seed,
        weight_gnb_power=weight_gnb_power,
        weight_ue_power=weight_ue_power,
        weight_latency=weight_latency,
        weight_backlog=weight_backlog,
    )
    env.max_steps = max_steps_per_episode
    agent = PPOAgent(state_dim=OBSERVATION_DIM, action_dim=10, hidden_dim=64, lr=3e-4)

    history = {
        "episode_rewards": [],
        "p99_latencies": [],
        "gnb_powers": [],
        "ue_powers": [],
        "backlogs": [],
        "max_ue_backlogs": [],
        "offered_rates": [],
        "delivered_rates": [],
        "avg_cqis": [],
        "p10_cqis": [],
        "min_cqis": [],
        "avg_dl_mcs": [],
        "avg_active_windows": [],
        "wide_bwp_rates": [],
        "sla_violation_rates": [],
        "losses": []
    }

    history_csv = os.path.abspath(history_csv)
    os.makedirs(os.path.dirname(history_csv), exist_ok=True)
    history_file = open(history_csv, "w", newline="", encoding="utf-8")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "Episode",
            "EpisodeReward",
            "AvgP99LatencyMs",
            "AvgGnbPowerW",
            "AvgUePowerMw",
            "AvgBacklogBytes",
            "AvgMaxUeBacklogBytes",
            "AvgOfferedRateMbps",
            "AvgDeliveredRateMbps",
            "TotalDeliveredMbits",
            "GnbControlEnergyJ",
            "GnbEnergyEfficiencyMbitPerJ",
            "GnbEnergyKwhPerTb",
            "UeControlEnergyJAllUes",
            "MinOfferedRateMbps",
            "MaxOfferedRateMbps",
            "StdOfferedRateMbps",
            "AvgCqi",
            "AvgP10Cqi",
            "AvgMinCqi",
            "AvgDlMcs",
            "AvgActiveWindowMs",
            "WideBwpRatePct",
            "SlaViolationRate",
            "Loss",
            "EnergyModel",
            "FrequencyControlMode",
            "PhysicalBandwidthMHz",
            "GnbEnergyBsCategory",
            "GnbEnergyRefConfig",
            "GnbEnergyEtaMode",
            "GnbEnergyAntennaRatio",
            "GnbEnergyPowerUnitW",
            "UeEnergyPowerUnitMw",
        ],
    )
    history_writer.writeheader()
    history_file.flush()
    print(f"📁 訓練紀錄 CSV: {history_csv}")

    start_time = time.time()

    print(f"\n{'Episode':<10} | {'Episode Reward':<16} | {'Avg P99延遲(ms)':<16} | {'Avg gNB功耗(W)':<16} | {'Avg UE功耗(mW)':<16} | {'Loss':<10}")
    print("-" * 110)

    try:
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            ep_p99_delays = []
            ep_gnb_powers = []
            ep_ue_powers = []
            ep_backlogs = []
            ep_max_ue_backlogs = []
            ep_offered_rates = []
            ep_delivered_rates = []
            ep_avg_cqis = []
            ep_p10_cqis = []
            ep_min_cqis = []
            ep_avg_dl_mcs = []
            ep_active_windows = []
            ep_wide_bwp = []
            ep_sla_violations = []

            for step in range(max_steps_per_episode):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

                agent.buffer.add(obs, action, log_prob, reward, value, done)

                obs = next_obs
                episode_reward += reward

                ep_p99_delays.append(step_info["p99_latency_ms"])
                ep_gnb_powers.append(step_info["gnb_power_w"])
                ep_ue_powers.append(step_info["ue_power_mw"])
                ep_backlogs.append(step_info["backlog_bytes"])
                ep_max_ue_backlogs.append(step_info["max_ue_backlog_bytes"])
                ep_offered_rates.append(step_info["offered_rate_mbps"])
                ep_delivered_rates.append(step_info["delivered_rate_mbps"])
                ep_avg_cqis.append(step_info["avg_cqi"])
                ep_p10_cqis.append(step_info["p10_cqi"])
                ep_min_cqis.append(step_info["min_cqi"])
                ep_avg_dl_mcs.append(step_info["avg_dl_mcs"])
                ep_active_windows.append(step_info["w_active_ms"])
                ep_wide_bwp.append(float(step_info["bwp_state"] == 1))
                ep_sla_violations.append(
                    float(step_info["p99_latency_ms"] > qos_target_ms)
                )

                if done:
                    break

            loss_val = agent.update()

            avg_p99 = np.mean(ep_p99_delays)
            avg_gnb = np.mean(ep_gnb_powers)
            avg_ue = np.mean(ep_ue_powers)
            avg_backlog = np.mean(ep_backlogs)
            avg_max_ue_backlog = np.mean(ep_max_ue_backlogs)
            avg_offered_rate = np.mean(ep_offered_rates)
            avg_delivered_rate = np.mean(ep_delivered_rates)
            min_offered_rate = np.min(ep_offered_rates)
            max_offered_rate = np.max(ep_offered_rates)
            std_offered_rate = np.std(ep_offered_rates)
            avg_cqi = np.mean(ep_avg_cqis)
            avg_p10_cqi = np.mean(ep_p10_cqis)
            avg_min_cqi = np.mean(ep_min_cqis)
            avg_dl_mcs = np.mean(ep_avg_dl_mcs)
            avg_active_window = np.mean(ep_active_windows)
            wide_bwp_rate_pct = 100.0 * np.mean(ep_wide_bwp)
            sla_violation_rate = np.mean(ep_sla_violations)
            control_interval_sec = 0.040
            total_delivered_mbits = float(
                np.sum(ep_delivered_rates) * control_interval_sec
            )
            gnb_control_energy_j = float(
                np.sum(ep_gnb_powers) * control_interval_sec
            )
            gnb_energy_efficiency = (
                total_delivered_mbits / gnb_control_energy_j
                if gnb_control_energy_j > 0.0 else float("nan")
            )
            gnb_energy_kwh_per_tb = (
                (8.0e6 / 3.6e6) / gnb_energy_efficiency
                if gnb_energy_efficiency > 0.0 else float("nan")
            )
            ue_control_energy_j_all_ues = float(
                np.sum(ep_ue_powers) / 1000.0
                * control_interval_sec * num_ues
            )

            history["episode_rewards"].append(episode_reward)
            history["p99_latencies"].append(avg_p99)
            history["gnb_powers"].append(avg_gnb)
            history["ue_powers"].append(avg_ue)
            history["backlogs"].append(avg_backlog)
            history["max_ue_backlogs"].append(avg_max_ue_backlog)
            history["offered_rates"].append(avg_offered_rate)
            history["delivered_rates"].append(avg_delivered_rate)
            history["avg_cqis"].append(avg_cqi)
            history["p10_cqis"].append(avg_p10_cqi)
            history["min_cqis"].append(avg_min_cqi)
            history["avg_dl_mcs"].append(avg_dl_mcs)
            history["avg_active_windows"].append(avg_active_window)
            history["wide_bwp_rates"].append(wide_bwp_rate_pct)
            history["sla_violation_rates"].append(sla_violation_rate)
            history["losses"].append(loss_val)

            history_writer.writerow({
                "Episode": episode,
                "EpisodeReward": episode_reward,
                "AvgP99LatencyMs": avg_p99,
                "AvgGnbPowerW": avg_gnb,
                "AvgUePowerMw": avg_ue,
                "AvgBacklogBytes": avg_backlog,
                "AvgMaxUeBacklogBytes": avg_max_ue_backlog,
                "AvgOfferedRateMbps": avg_offered_rate,
                "AvgDeliveredRateMbps": avg_delivered_rate,
                "TotalDeliveredMbits": total_delivered_mbits,
                "GnbControlEnergyJ": gnb_control_energy_j,
                "GnbEnergyEfficiencyMbitPerJ": gnb_energy_efficiency,
                "GnbEnergyKwhPerTb": gnb_energy_kwh_per_tb,
                "UeControlEnergyJAllUes": ue_control_energy_j_all_ues,
                "MinOfferedRateMbps": min_offered_rate,
                "MaxOfferedRateMbps": max_offered_rate,
                "StdOfferedRateMbps": std_offered_rate,
                "AvgCqi": avg_cqi,
                "AvgP10Cqi": avg_p10_cqi,
                "AvgMinCqi": avg_min_cqi,
                "AvgDlMcs": avg_dl_mcs,
                "AvgActiveWindowMs": avg_active_window,
                "WideBwpRatePct": wide_bwp_rate_pct,
                "SlaViolationRate": sla_violation_rate,
                "Loss": loss_val,
                "EnergyModel": "5G-LENA-TR38.864-TR38.840",
                "FrequencyControlMode": "PRB-profile-on-single-physical-BWP",
                "PhysicalBandwidthMHz": 100,
                "GnbEnergyBsCategory": gnb_energy_bs_category,
                "GnbEnergyRefConfig": gnb_energy_ref_config,
                "GnbEnergyEtaMode": gnb_energy_eta_mode,
                "GnbEnergyAntennaRatio": gnb_energy_antenna_ratio,
                "GnbEnergyPowerUnitW": gnb_energy_power_unit_w,
                "UeEnergyPowerUnitMw": ue_energy_power_unit_mw,
            })
            history_file.flush()

            print(f"{episode:<10} | {episode_reward:<16.4f} | {avg_p99:<16.2f} | {avg_gnb:<16.2f} | {avg_ue:<16.2f} | {loss_val:<10.4f}")
    finally:
        env.close()
        history_file.close()

    elapsed_sec = time.time() - start_time
    print("-" * 110)
    print(f"✅ 連續狀態 PPO 代理人訓練完成！總耗時: {elapsed_sec:.2f} 秒")

    model_save_path = os.path.abspath(model_out)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(agent.policy.state_dict(), model_save_path)
    print(f"💾 模型權重已成功儲存至: {model_save_path}")
    print(f"📊 訓練紀錄已儲存至: {history_csv}")

    try:
        scripted_policy = torch.jit.script(agent.policy)
        scripted_save_path = os.path.abspath(torchscript_out)
        os.makedirs(os.path.dirname(scripted_save_path), exist_ok=True)
        scripted_policy.save(scripted_save_path)
        print(f"⚡ C++ LibTorch 整合檔 (TorchScript) 已匯出至: {scripted_save_path}")
    except Exception as e:
        print(f"⚠️ TorchScript 匯出提示: {e}")

    print("=" * 110 + "\n")
    return history, agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 5G-Advanced PPO agent with ns-3 IPC."
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-ues", type=int, default=10)
    parser.add_argument("--udp-rate", type=float, default=2.0)
    parser.add_argument("--enable-burst-traffic", action="store_true")
    parser.add_argument("--synchronize-burst-traffic", action="store_true")
    parser.add_argument("--udp-burst-rate", type=float, default=3.0)
    parser.add_argument("--burst-mean-on-ms", type=float, default=80.0)
    parser.add_argument("--burst-mean-off-ms", type=float, default=120.0)
    parser.add_argument("--gnb-tx-power-dbm", type=float, default=30.0)
    parser.add_argument("--gnb-energy-bs-category", choices=("BsCat1", "BsCat2"), default="BsCat1")
    parser.add_argument("--gnb-energy-ref-config", choices=("Set1", "Set2", "Set3"), default="Set1")
    parser.add_argument("--gnb-energy-eta-mode", choices=("Single", "Dual"), default="Dual")
    parser.add_argument("--gnb-energy-antenna-ratio", type=float, default=0.4)
    parser.add_argument("--gnb-energy-power-unit-w", type=float, default=1.0)
    parser.add_argument("--ue-energy-power-unit-mw", type=float, default=1.0)
    parser.add_argument("--history-csv", default="ppo_training_history.csv")
    parser.add_argument("--ns3-csv", default="multi_user_rl_step.csv")
    parser.add_argument("--model-out", default="ppo_5g_joint2d_model.pt")
    parser.add_argument("--torchscript-out", default="ppo_5g_joint2d_torchscript.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qos-target-ms", type=float, default=30.0)
    parser.add_argument("--warmup-ms", type=float, default=200.0)
    parser.add_argument("--max-active-ms", type=float, default=30.0)
    parser.add_argument("--initial-active-ms", type=float, default=30.0)
    parser.add_argument("--weight-gnb-power", type=float, default=0.15)
    parser.add_argument("--weight-ue-power", type=float, default=0.15)
    parser.add_argument("--weight-latency", type=float, default=0.45)
    parser.add_argument("--weight-backlog", type=float, default=0.25)
    parser.add_argument("--rerun-train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num-episodes must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
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
    if not -50.0 <= args.gnb_tx_power_dbm <= 60.0:
        parser.error("--gnb-tx-power-dbm must be between -50 and 60")
    if args.gnb_energy_power_unit_w <= 0.0 or args.ue_energy_power_unit_mw <= 0.0:
        parser.error("official Energy model PowerUnit values must be greater than zero")
    if not 0.0 <= args.gnb_energy_antenna_ratio <= 1.0:
        parser.error("--gnb-energy-antenna-ratio must be inside [0, 1]")
    if args.qos_target_ms <= 0:
        parser.error("--qos-target-ms must be greater than zero")
    if args.warmup_ms < 0:
        parser.error("--warmup-ms cannot be negative")
    if not 10.0 < args.max_active_ms <= 40.0:
        parser.error("--max-active-ms must be greater than 10 and at most 40")
    if not 10.0 <= args.initial_active_ms <= args.max_active_ms:
        parser.error("--initial-active-ms must be between 10 and --max-active-ms")
    reward_weights = [
        args.weight_gnb_power,
        args.weight_ue_power,
        args.weight_latency,
        args.weight_backlog,
    ]
    if any(weight < 0 for weight in reward_weights):
        parser.error("reward weights cannot be negative")
    if sum(reward_weights) <= 0:
        parser.error("at least one reward weight must be positive")

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train_ppo_agent(
        num_episodes=cli_args.num_episodes,
        max_steps_per_episode=cli_args.max_steps,
        num_ues=cli_args.num_ues,
        udp_rate=cli_args.udp_rate,
        enable_burst_traffic=cli_args.enable_burst_traffic,
        synchronize_burst_traffic=cli_args.synchronize_burst_traffic,
        udp_burst_rate=cli_args.udp_burst_rate,
        burst_mean_on_ms=cli_args.burst_mean_on_ms,
        burst_mean_off_ms=cli_args.burst_mean_off_ms,
        gnb_tx_power_dbm=cli_args.gnb_tx_power_dbm,
        gnb_energy_bs_category=cli_args.gnb_energy_bs_category,
        gnb_energy_ref_config=cli_args.gnb_energy_ref_config,
        gnb_energy_eta_mode=cli_args.gnb_energy_eta_mode,
        gnb_energy_antenna_ratio=cli_args.gnb_energy_antenna_ratio,
        gnb_energy_power_unit_w=cli_args.gnb_energy_power_unit_w,
        ue_energy_power_unit_mw=cli_args.ue_energy_power_unit_mw,
        history_csv=cli_args.history_csv,
        ns3_csv=cli_args.ns3_csv,
        model_out=cli_args.model_out,
        torchscript_out=cli_args.torchscript_out,
        seed=cli_args.seed,
        qos_target_ms=cli_args.qos_target_ms,
        warmup_ms=cli_args.warmup_ms,
        max_active_ms=cli_args.max_active_ms,
        initial_active_ms=cli_args.initial_active_ms,
        weight_gnb_power=cli_args.weight_gnb_power,
        weight_ue_power=cli_args.weight_ue_power,
        weight_latency=cli_args.weight_latency,
        weight_backlog=cli_args.weight_backlog,
    )
