#!/usr/bin/env python3
"""
5G-Advanced Joint 2D DRL Agent - PyTorch PPO Training Main Script
File: train_drl_agent.py

Features:
  - Real-time Socket IPC Co-Simulation with ns-3
  - Continuous Stateful Physical Layer Dynamics
  - Actor-Critic network with categorical absolute BWP/active-window actions
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

from ns3_environment import (
    ACTION_ENCODING,
    S5_ACTION_ENCODING,
    OBSERVATION_DIM,
    REWARD_MODE,
    Ns3MultiUserEnv,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=OBSERVATION_DIM, action_dim=22, hidden_dim=64,
                 context_conditioned_actor=False):
        super(ActorCritic, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.context_conditioned_actor = bool(context_conditioned_actor)
        actor_input_dim = hidden_dim + (1 if self.context_conditioned_actor else 0)
        if isinstance(action_dim, (list, tuple, np.ndarray)):
            self.action_dims = tuple(int(value) for value in action_dim)
            self.actor_heads = nn.ModuleList(
                nn.Linear(actor_input_dim, branch_dim)
                for branch_dim in self.action_dims
            )
            self.actor_head = None
        else:
            self.action_dims = None
            self.actor_head = nn.Linear(actor_input_dim, int(action_dim))
            self.actor_heads = None
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_net(state)
        actor_features = features
        if self.context_conditioned_actor:
            actor_features = torch.cat((features, state[:, 4:5]), dim=-1)
        if self.action_dims is None:
            logits = self.actor_head(actor_features)
        else:
            logits = [head(actor_features) for head in self.actor_heads]
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(self, state, action=None):
        logits, value = self.forward(state)
        if self.action_dims is None:
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            distributions = [Categorical(logits=branch) for branch in logits]
            if action is None:
                action = torch.stack(
                    [dist.sample() for dist in distributions], dim=-1
                )
            log_prob = torch.stack(
                [dist.log_prob(action[..., index])
                 for index, dist in enumerate(distributions)], dim=-1
            ).sum(dim=-1)
            entropy = torch.stack(
                [dist.entropy() for dist in distributions], dim=-1
            ).sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.contexts = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.contexts.clear()

    def add(self, state, action, log_prob, reward, value, done, context=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.contexts.append(context)


class PPOAgent:
    def __init__(self, 
                 state_dim=OBSERVATION_DIM,
                 action_dim=22,
                 hidden_dim=64,
                 lr=1e-4,
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_eps=0.2, 
                 c_val=0.5, 
                 c_ent=0.01, 
                 ppo_epochs=4,
                 batch_size=64,
                 target_kl=0.01,
                 context_conditioned_actor=False,
                 safe_prior_actions=None,
                 safe_prior_mass=0.0):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        if isinstance(action_dim, (list, tuple, np.ndarray)):
            self.action_dims = tuple(int(value) for value in action_dim)
            self.action_dim = None
        else:
            self.action_dims = None
            self.action_dim = int(action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(
            state_dim, action_dim, hidden_dim,
            context_conditioned_actor=context_conditioned_actor,
        ).to(self.device)
        self.safe_prior_actions = tuple(sorted(set(safe_prior_actions or ())))
        self.safe_prior_mass = float(safe_prior_mass)
        if self.safe_prior_actions and self.action_dims is not None:
            raise ValueError("safe-prior actions are only supported for scalar actions")
        if self.safe_prior_actions:
            if any(action < 0 or action >= action_dim for action in self.safe_prior_actions):
                raise ValueError("safe-prior action index is outside the action space")
            if len(self.safe_prior_actions) >= action_dim:
                raise ValueError("safe-prior actions must not cover the entire action space")
            if not 0.0 < self.safe_prior_mass < 1.0:
                raise ValueError("safe_prior_mass must be inside (0, 1)")
            preferred_count = len(self.safe_prior_actions)
            other_count = action_dim - preferred_count
            logit_boost = np.log(
                (self.safe_prior_mass / (1.0 - self.safe_prior_mass))
                * (other_count / preferred_count)
            )
            with torch.no_grad():
                self.policy.actor_head.bias.zero_()
                self.policy.actor_head.bias[list(self.safe_prior_actions)] = logit_boost
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)
        if self.action_dims is None:
            return action.item(), log_prob.item(), value.item()
        return action.squeeze(0).cpu().numpy().astype(np.int64), log_prob.item(), value.item()

    def select_deterministic_action(self, state):
        """Return the greedy action used for deployment/checkpoint evaluation."""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(state_tensor)
        if self.action_dims is None:
            return int(torch.argmax(logits, dim=-1).item())
        return np.asarray(
            [int(torch.argmax(branch, dim=-1).item()) for branch in logits],
            dtype=np.int64,
        )

    def deterministic_action_distribution(self, state):
        """Return greedy action plus per-head probabilities and entropies."""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(state_tensor)
            if self.action_dims is None:
                probabilities = [torch.softmax(logits, dim=-1)[0].cpu().numpy()]
            else:
                probabilities = [
                    torch.softmax(branch, dim=-1)[0].cpu().numpy()
                    for branch in logits
                ]
        action = np.asarray(
            [int(np.argmax(probability)) for probability in probabilities],
            dtype=np.int64,
        ) if self.action_dims is not None else int(np.argmax(probabilities[0]))
        entropies = [
            float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
            for probability in probabilities
        ]
        return action, probabilities, entropies

    def update(self):
        if not self.buffer.rewards:
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "entropy": float("nan"),
                "approx_kl": float("nan"),
                "clip_fraction": float("nan"),
                "explained_variance": float("nan"),
                "value_target_mean": float("nan"),
                "value_target_std": float("nan"),
                "value_prediction_mean": float("nan"),
                "value_prediction_std": float("nan"),
                "value_return_correlation": float("nan"),
                "normalized_value_error": float("nan"),
                "actor_grad_norm": float("nan"),
                "critic_grad_norm": float("nan"),
                "feature_grad_norm": float("nan"),
                "epoch_kl_trace": "",
                "epoch_clip_fraction_trace": "",
                "action_counts": "",
                "action_frequencies": "",
                "epochs_completed": 0,
                "samples": 0,
            }

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
        value_predictions = torch.FloatTensor(values).to(self.device)
        returns = advantages + value_predictions

        return_mean = returns.mean().item()
        return_std = returns.std(unbiased=False).item()
        value_mean = value_predictions.mean().item()
        value_std = value_predictions.std(unbiased=False).item()
        return_variance_before_update = torch.var(returns, unbiased=False)
        if return_variance_before_update.item() > 1e-12:
            normalized_value_error = (
                torch.mean((returns - value_predictions) ** 2)
                / return_variance_before_update
            ).item()
        else:
            normalized_value_error = float("nan")
        if return_variance_before_update.item() > 1e-12 and value_std > 1e-12:
            centered_returns = returns - returns.mean()
            centered_values = value_predictions - value_predictions.mean()
            value_return_correlation = (
                (centered_returns * centered_values).mean()
                / (
                    centered_returns.std(unbiased=False)
                    * centered_values.std(unbiased=False)
                )
            ).item()
        else:
            value_return_correlation = float("nan")

        if self.action_dims is None:
            action_counts = torch.bincount(
                actions, minlength=self.action_dim
            ).cpu().numpy()
            action_frequencies = action_counts / max(1, len(actions))
        else:
            action_counts = [
                torch.bincount(actions[:, index], minlength=branch_dim)
                .cpu().numpy()
                for index, branch_dim in enumerate(self.action_dims)
            ]
            action_frequencies = [
                counts / max(1, len(actions)) for counts in action_counts
            ]
        
        # Normalize advantages independently per traffic context.  This keeps
        # a higher-load context's larger absolute return scale from dominating
        # updates for lower-load contexts while preserving one shared actor.
        if self.buffer.contexts and all(context is not None for context in self.buffer.contexts):
            context_array = np.asarray(self.buffer.contexts, dtype=object)
            normalized = torch.empty_like(advantages)
            for context in np.unique(context_array):
                indices = np.flatnonzero(context_array == context)
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                values_for_context = advantages[index_tensor]
                normalized[index_tensor] = (
                    values_for_context - values_for_context.mean()
                ) / (values_for_context.std(unbiased=False) + 1e-8)
            advantages = normalized
        else:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        dataset_size = len(states)
        update_metrics = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
            "feature_grad_norm": [],
        }
        epoch_kl_trace = []
        epoch_clip_fraction_trace = []
        epochs_completed = 0
        for _ in range(self.ppo_epochs):
            epoch_kls = []
            epoch_clip_fractions = []
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
                actor_parameters = (
                    self.policy.actor_head.parameters()
                    if self.action_dims is None else
                    (parameter for head in self.policy.actor_heads
                     for parameter in head.parameters())
                )
                for metric_name, parameters in (
                    ("actor_grad_norm", actor_parameters),
                    ("critic_grad_norm", self.policy.critic_head.parameters()),
                    ("feature_grad_norm", self.policy.feature_net.parameters()),
                ):
                    squared_norm = sum(
                        parameter.grad.detach().pow(2).sum().item()
                        for parameter in parameters
                        if parameter.grad is not None
                    )
                    update_metrics[metric_name].append(float(np.sqrt(squared_norm)))
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - b_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratios - 1.0) > self.clip_eps).float().mean()
                    )
                update_metrics["loss"].append(total_loss.item())
                update_metrics["policy_loss"].append(policy_loss.item())
                update_metrics["value_loss"].append(value_loss.item())
                update_metrics["entropy"].append(entropy.mean().item())
                update_metrics["approx_kl"].append(approx_kl.item())
                update_metrics["clip_fraction"].append(clip_fraction.item())
                epoch_kls.append(approx_kl.item())
                epoch_clip_fractions.append(clip_fraction.item())

            epochs_completed += 1
            epoch_kl_trace.append(float(np.mean(epoch_kls)))
            epoch_clip_fraction_trace.append(float(np.mean(epoch_clip_fractions)))
            if self.target_kl is not None and epoch_kl_trace[-1] > self.target_kl:
                break

        with torch.no_grad():
            _, final_values = self.policy(states)
            final_values = final_values.squeeze(-1)
            return_variance = torch.var(returns, unbiased=False)
            if return_variance.item() > 1e-12:
                explained_variance = 1.0 - (
                    torch.var(returns - final_values, unbiased=False) / return_variance
                )
                explained_variance = explained_variance.item()
            else:
                explained_variance = float("nan")

        self.buffer.clear()
        return {
            key: float(np.mean(values)) for key, values in update_metrics.items()
        } | {
            "explained_variance": explained_variance,
            "value_target_mean": return_mean,
            "value_target_std": return_std,
            "value_prediction_mean": value_mean,
            "value_prediction_std": value_std,
            "value_return_correlation": value_return_correlation,
            "normalized_value_error": normalized_value_error,
            "epoch_kl_trace": ";".join(f"{value:.8g}" for value in epoch_kl_trace),
            "epoch_clip_fraction_trace": ";".join(
                f"{value:.8g}" for value in epoch_clip_fraction_trace
            ),
            "action_counts": (
                ";".join(str(int(value)) for value in action_counts)
                if self.action_dims is None else
                "|".join(",".join(str(int(value)) for value in counts)
                           for counts in action_counts)
            ),
            "action_frequencies": (
                ";".join(f"{value:.8g}" for value in action_frequencies)
                if self.action_dims is None else
                "|".join(",".join(f"{value:.8g}" for value in frequencies)
                           for frequencies in action_frequencies)
            ),
            "epochs_completed": epochs_completed,
            "samples": dataset_size,
        }


def train_ppo_agent(
    num_episodes=10,
    max_steps_per_episode=50,
    num_ues=10,
    udp_rate=2.0,
    udp_rate_choices=None,
    udp_rate_block_size=1,
    enable_burst_traffic=False,
    synchronize_burst_traffic=False,
    udp_burst_rate=3.0,
    burst_mean_on_ms=80.0,
    burst_mean_off_ms=120.0,
    gnb_tx_power_dbm=30.0,
    gnb_energy_bs_category="BsCat1",
    gnb_energy_ref_config="Custom",
    gnb_energy_reference_tx_power_dbm=30.0,
    gnb_energy_eta_mode="Dual",
    gnb_energy_antenna_ratio=0.4,
    gnb_energy_power_unit_w=1.0,
    ue_energy_power_unit_mw=1.0,
    enable_true_bwp=True,
    history_csv="ppo_training_history.csv",
    step_csv="ppo_training_steps.csv",
    ns3_csv="multi_user_rl_step.csv",
    model_out="ppo_5g_joint2d_model.pt",
    torchscript_out="ppo_5g_joint2d_torchscript.pt",
    best_model_out="ppo_5g_joint2d_best_model.pt",
    best_torchscript_out="ppo_5g_joint2d_best_torchscript.pt",
    checkpoint_dir="ppo_checkpoints",
    checkpoint_interval=10,
    episodes_per_update=4,
    rollout_samples=0,
    learning_rate=1e-4,
    value_loss_coefficient=0.5,
    entropy_coefficient=0.01,
    context_conditioned_actor=False,
    ppo_epochs=4,
    batch_size=64,
    target_kl=0.01,
    best_qos_threshold_pct=95.0,
    best_window=4,
    deterministic_eval_interval=4,
    deterministic_eval_seeds=(101,),
    deterministic_eval_csv=None,
    seed=42,
    qos_target_ms=30.0,
    warmup_ms=200.0,
    max_active_ms=30.0,
    initial_active_ms=30.0,
    active_window_options=None,
    fixed_bwp_state=None,
    enable_dci_dynamic=True,
    weight_gnb_power=0.15,
    weight_ue_power=0.15,
    weight_system_power=0.30,
    weight_latency=0.45,
    weight_backlog=0.25,
    reward_mode=REWARD_MODE,
    backlog_safe_bytes=50000.0,
    terminal_qos_bonus=0.0,
    bwp_switch_penalty=0.0,
    safe_prior_active_ms=None,
    safe_prior_mass=0.7,
    enable_per_ue_bwp_action=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if udp_rate_choices is None:
        traffic_rates = (float(udp_rate),)
    else:
        traffic_rates = tuple(float(rate) for rate in udp_rate_choices)
    if not traffic_rates or any(rate <= 0.0 for rate in traffic_rates):
        raise ValueError("udp_rate_choices must contain positive rates")
    if int(udp_rate_block_size) <= 0:
        raise ValueError("udp_rate_block_size must be positive")
    traffic_schedule = []
    if int(udp_rate_block_size) == 1:
        traffic_rng = np.random.default_rng(seed + 20261020)
        while len(traffic_schedule) < num_episodes:
            traffic_schedule.extend(traffic_rng.permutation(traffic_rates).tolist())
    else:
        # Deterministic context blocks: hold each load for several complete
        # episodes before moving to the next one.
        while len(traffic_schedule) < num_episodes:
            for rate in traffic_rates:
                traffic_schedule.extend([rate] * int(udp_rate_block_size))
                if len(traffic_schedule) >= num_episodes:
                    break
    traffic_schedule = traffic_schedule[:num_episodes]
    offered_rate_max_mbps = num_ues * (
        max(traffic_rates) + (udp_burst_rate if enable_burst_traffic else 0.0)
    )
    rollout_samples_target = int(rollout_samples)
    if rollout_samples_target <= 0:
        rollout_samples_target = int(episodes_per_update * max_steps_per_episode)
    deterministic_eval_seeds = tuple(int(value) for value in deterministic_eval_seeds)
    if not deterministic_eval_seeds:
        raise ValueError("deterministic_eval_seeds must not be empty")

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
    if len(traffic_rates) > 1:
        traffic_label = (
            "episode-randomized base rates "
            + "/".join(f"{rate:g}" for rate in traffic_rates)
            + " Mbps/UE (balanced shuffled blocks)"
        )
    print(f"📡 流量: {traffic_label}")
    print(
        f"👁️ Offered-load observation normalization: "
        f"0–{offered_rate_max_mbps:g} Mbps total"
    )
    print(f"🎯 SLA={qos_target_ms:.1f}ms | Active Window=10–{max_active_ms:.0f}ms (initial {initial_active_ms:.0f}ms) | gNB Tx={gnb_tx_power_dbm:.1f}dBm | warm-up={warmup_ms:.0f}ms | seed={seed}")
    print(
        "🔋 Energy: 5G-LENA TR 38.864/38.840 | "
        f"{gnb_energy_bs_category}/{gnb_energy_ref_config}/{gnb_energy_eta_mode} | "
        f"reference Tx={gnb_energy_reference_tx_power_dbm:g}dBm | "
        f"gNB unit={gnb_energy_power_unit_w:g}W | UE unit={ue_energy_power_unit_mw:g}mW"
    )
    print(
        "📶 Frequency control: "
        + ("physical 20/100 MHz BWP" if enable_true_bwp else "legacy PRB profile")
    )
    print(
        f"🎁 Reward: {reward_mode} | system-power={weight_system_power:g} | "
        f"latency={weight_latency:g} | backlog={weight_backlog:g} "
        f"above {backlog_safe_bytes:g} bytes | bwp-switch={bwp_switch_penalty:g}"
    )
    print(
        f"🧠 PPO: lr={learning_rate:g} | rollout={episodes_per_update} episodes "
        f"(~{rollout_samples_target} target samples) | "
        f"epochs={ppo_epochs} | batch={batch_size} | target-KL={target_kl:g} | "
        f"value-loss-coef={value_loss_coefficient:g} | entropy-coef={entropy_coefficient:g} | "
        f"context-actor={context_conditioned_actor}"
    )
    print(
        f"🎯 Best-model selection: deterministic evaluation every "
        f"{deterministic_eval_interval} episodes after PPO update | "
        f"seeds={list(deterministic_eval_seeds)} | "
        f"QoS threshold={best_qos_threshold_pct:g}%"
    )
    print(f"⚡ 運算硬體裝置: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 110)

    env = Ns3MultiUserEnv(
        num_ues=num_ues,
        udp_rate=udp_rate,
        offered_rate_max_mbps=offered_rate_max_mbps,
        enable_burst_traffic=enable_burst_traffic,
        synchronize_burst_traffic=synchronize_burst_traffic,
        udp_burst_rate=udp_burst_rate,
        burst_mean_on_ms=burst_mean_on_ms,
        burst_mean_off_ms=burst_mean_off_ms,
        gnb_tx_power_dbm=gnb_tx_power_dbm,
        gnb_energy_bs_category=gnb_energy_bs_category,
        gnb_energy_ref_config=gnb_energy_ref_config,
        gnb_energy_reference_tx_power_dbm=gnb_energy_reference_tx_power_dbm,
        gnb_energy_eta_mode=gnb_energy_eta_mode,
        gnb_energy_antenna_ratio=gnb_energy_antenna_ratio,
        gnb_energy_power_unit_w=gnb_energy_power_unit_w,
        ue_energy_power_unit_mw=ue_energy_power_unit_mw,
        enable_true_bwp=enable_true_bwp,
        enable_mock=False,
        sim_time=episode_sim_time,
        csv_filename=ns3_csv,
        qos_target_ms=qos_target_ms,
        controller_warmup_ms=warmup_ms,
        controller_max_active_ms=max_active_ms,
        controller_initial_active_ms=initial_active_ms,
        active_window_options=active_window_options,
        fixed_bwp_state=fixed_bwp_state,
        enable_dci_dynamic=enable_dci_dynamic,
        rng_seed=seed,
        weight_gnb_power=weight_gnb_power,
        weight_ue_power=weight_ue_power,
        weight_system_power=weight_system_power,
        weight_latency=weight_latency,
        weight_backlog=weight_backlog,
        reward_mode=reward_mode,
        backlog_safe_bytes=backlog_safe_bytes,
        bwp_switch_penalty=bwp_switch_penalty,
        enable_per_ue_bwp_action=enable_per_ue_bwp_action,
    )
    env.max_steps = max_steps_per_episode
    if enable_per_ue_bwp_action:
        print(
            f"🎛️ Action: S5 factorized MultiDiscrete {env.action_space.nvec.tolist()} | "
            f"observation_dim={env.observation_dim} | "
            f"Active={env.ACTIVE_WINDOW_OPTIONS} ms × per-UE BWP"
        )
    else:
        print(
            f"🎛️ Action: {ACTION_ENCODING} | {env.action_space.n} actions | "
            f"Active={env.ACTIVE_WINDOW_OPTIONS} ms × BWP=(20, 100) MHz"
        )
    safe_prior_actions = []
    if safe_prior_active_ms:
        for active_ms in safe_prior_active_ms:
            if active_ms not in env.ACTIVE_WINDOW_OPTIONS:
                raise ValueError(
                    f"safe-prior active window {active_ms:g} ms is not in "
                    f"{env.ACTIVE_WINDOW_OPTIONS}"
                )
            active_index = env.ACTIVE_WINDOW_OPTIONS.index(active_ms)
            safe_prior_actions.append(active_index * 2 + 1)
        print(
            f"🛟 Safe prior: actions={safe_prior_actions} | "
            f"active={list(safe_prior_active_ms)} ms | 100 MHz | "
            f"target mass={safe_prior_mass:.1%}"
        )
    else:
        print("🛟 Safe prior: disabled (default actor initialization)")

    action_dim = (
        env.action_space.nvec.tolist()
        if enable_per_ue_bwp_action else env.action_space.n
    )
    agent = PPOAgent(
        state_dim=env.observation_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=learning_rate,
        c_val=value_loss_coefficient,
        c_ent=entropy_coefficient,
        context_conditioned_actor=context_conditioned_actor,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        target_kl=target_kl,
        safe_prior_actions=safe_prior_actions,
        safe_prior_mass=safe_prior_mass if safe_prior_actions else 0.0,
    )

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
    step_csv = os.path.abspath(step_csv)
    if deterministic_eval_csv is None:
        history_root, history_ext = os.path.splitext(history_csv)
        deterministic_eval_csv = (
            history_root + "_deterministic_eval" + (history_ext or ".csv")
        )
    deterministic_eval_csv = os.path.abspath(deterministic_eval_csv)
    best_model_out = os.path.abspath(best_model_out)
    best_torchscript_out = os.path.abspath(best_torchscript_out)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(os.path.dirname(history_csv), exist_ok=True)
    os.makedirs(os.path.dirname(step_csv), exist_ok=True)
    os.makedirs(os.path.dirname(deterministic_eval_csv), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_out), exist_ok=True)
    os.makedirs(os.path.dirname(best_torchscript_out), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_file = open(history_csv, "w", newline="", encoding="utf-8")
    step_file = open(step_csv, "w", newline="", encoding="utf-8")
    deterministic_eval_file = open(
        deterministic_eval_csv, "w", newline="", encoding="utf-8"
    )
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "Episode",
            "EpisodeReward",
            "AvgStepReward",
            "MinStepReward",
            "MaxStepReward",
            "StdStepReward",
            "RewardGnbPower",
            "RewardUePower",
            "RewardSystemPower",
            "RewardLatency",
            "RewardBacklog",
            "TerminalReward",
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
            "TotalNetworkEnergyJ",
            "NetworkEnergyEfficiencyMbitPerJ",
            "MinOfferedRateMbps",
            "MaxOfferedRateMbps",
            "StdOfferedRateMbps",
            "AvgCqi",
            "AvgP10Cqi",
            "AvgMinCqi",
            "AvgDlMcs",
            "AvgActiveWindowMs",
            "WideBwpRatePct",
            "PerUeWideBwpRatePct",
            "MeanWideBwpRatePct",
            "AnyUeWideBwpRatePct",
            "AllUeWideBwpRatePct",
            "BwpSwitchCount",
            "PerUeBwpSwitchCount",
            "SlaViolationRate",
            "QosSatisfactionRatePct",
            "AvgQosMarginMs",
            "AvgSlaViolationMs",
            "AvgJainFairness",
            "Loss",
            "PolicyLoss",
            "ValueLoss",
            "Entropy",
            "ApproxKl",
            "ClipFraction",
            "ExplainedVariance",
            "ValueTargetMean",
            "ValueTargetStd",
            "ValuePredictionMean",
            "ValuePredictionStd",
            "ValueReturnCorrelation",
            "NormalizedValueError",
            "ActorGradNorm",
            "CriticGradNorm",
            "FeatureGradNorm",
            "EpochApproxKlTrace",
            "EpochClipFractionTrace",
            "UpdateActionCounts",
            "UpdateActionFrequencies",
            "PpoUpdated",
            "PpoUpdateSamples",
            "PpoEpochsCompleted",
            "RollingReward",
            "RollingQosSatisfactionRatePct",
            "DeterministicEvalPerformed",
            "DeterministicEvalMeanReward",
            "DeterministicEvalMinQosPct",
            "DeterministicEvalMeanGnbPowerW",
            "DeterministicEvalMeanUePowerMw",
            "DeterministicEvalMeanActiveWindowMs",
            "DeterministicEvalActions",
            "IsBestCheckpoint",
            "EnergyModel",
            "FrequencyControlMode",
            "PhysicalBandwidthMHz",
            "GnbEnergyBsCategory",
            "GnbEnergyRefConfig",
            "GnbEnergyReferenceTxPowerDbm",
            "GnbEnergyEtaMode",
            "GnbEnergyAntennaRatio",
            "GnbEnergyPowerUnitW",
            "UeEnergyPowerUnitMw",
            "QosTargetMs",
            "EnableDciDynamic",
            "ControlIntervalMs",
            "NumUes",
            "UdpRateMbpsPerUe",
            "OfferedRateNormalizationMbps",
            "Seed",
            "LearningRate",
            "ValueLossCoefficient",
            "ConfiguredPpoEpochs",
            "BatchSize",
            "TargetKl",
            "EpisodesPerUpdate",
            "ActionEncoding",
            "ActionSpaceSize",
            "SafePriorActions",
            "SafePriorMass",
            "RewardMode",
            "WeightSystemPower",
            "WeightLatency",
            "WeightBacklog",
            "BacklogSafeBytes",
            "TotalSystemPowerNormalizationW",
        ],
    )
    step_writer = csv.DictWriter(
        step_file,
        fieldnames=[
            "Episode", "Step", "GlobalStep", "ControlElapsedSec",
            "UdpRateMbpsPerUe", "OfferedRateNormalizationMbps",
            "Action", "ActionEncoding", "TargetActiveWindowMs",
            "DeltaWindowMs", "ActionBwpState",
            "Reward", "RewardGnbPower", "RewardUePower", "RewardSystemPower",
            "RewardLatency", "RewardBacklog",
            "RewardBwpSwitch", "BwpSwitchPenaltyCount",
            "GnbPowerNormalized", "UePowerNormalized", "SystemPowerNormalized",
            "TotalUePowerW", "TotalSystemPowerW",
            "LatencyPenalty", "LatencyExcessRatio", "BacklogNormalized",
            "RawBacklogNormalized", "BacklogExcessNormalized",
            "BacklogSafeBytes", "RewardMode",
            "ValueEstimate", "ActionLogProb",
            "P99LatencyMs", "QosTargetMs", "QosSatisfied",
            "QosMarginMs", "SlaViolationMs",
            "BacklogBytes", "MaxUeBacklogBytes",
            "OfferedRateMbps", "DeliveredRateMbps", "StepDeliveredMbits",
            "GnbPowerW", "StepGnbEnergyJ", "GnbEnergyEfficiencyMbitPerJ",
            "UePowerMwPerUe", "StepUeEnergyJAllUes",
            "TotalNetworkEnergyJ", "NetworkEnergyEfficiencyMbitPerJ",
            "AvgCqi", "P10Cqi", "MinCqi", "AvgDlMcs", "JainFairness",
            "ActiveWindowMs", "BwpState", "IsWideBwp", "PerUeBwpState",
            "Terminated", "Truncated", "Backend",
        ],
    )
    deterministic_eval_writer = csv.DictWriter(
        deterministic_eval_file,
        fieldnames=[
            "TrainingEpisode", "UdpRateMbpsPerUe", "Seed", "Steps",
            "EpisodeReward", "QosSatisfactionRatePct", "AvgP99LatencyMs",
            "AvgGnbPowerW", "AvgUePowerMw", "AvgSystemPowerW",
            "AvgActiveWindowMs",
            "Actions", "PerUeBwpStates", "ActionProbabilities", "ActionEntropies",
            "IsFeasible", "IsBestCheckpoint",
        ],
    )
    history_writer.writeheader()
    step_writer.writeheader()
    deterministic_eval_writer.writeheader()
    history_file.flush()
    step_file.flush()
    print(f"📁 訓練紀錄 CSV: {history_csv}")
    print(f"🔬 每步實驗紀錄 CSV: {step_csv}")
    print(f"🎯 Deterministic evaluation CSV: {deterministic_eval_csv}")

    start_time = time.time()
    # Deployment checkpoint selection is based on the deterministic evaluation
    # objective: among QoS-feasible candidates, keep the lowest total system
    # power (gNB plus all UE power). Reward remains useful for diagnostics, but
    # must not select a higher-energy checkpoint.
    best_score = -float("inf")
    best_deterministic_system_power = float("inf")
    best_episode = None
    next_checkpoint_episode = checkpoint_interval
    next_deterministic_eval_episode = deterministic_eval_interval

    def evaluate_policy_deterministically(training_episode):
        """Evaluate the current policy without sampling or touching PPO rollout data."""
        eval_rows = []
        policy_was_training = agent.policy.training
        agent.policy.eval()
        try:
            for eval_rate in sorted(set(traffic_rates)):
                for eval_seed in deterministic_eval_seeds:
                    eval_ns3_csv = os.path.join(
                        checkpoint_dir,
                        f"det_eval_ep{training_episode:04d}_rate{eval_rate:g}_seed{eval_seed}.csv",
                    )
                    eval_env = Ns3MultiUserEnv(
                        num_ues=num_ues,
                        udp_rate=eval_rate,
                        offered_rate_max_mbps=offered_rate_max_mbps,
                        enable_burst_traffic=enable_burst_traffic,
                        synchronize_burst_traffic=synchronize_burst_traffic,
                        udp_burst_rate=udp_burst_rate,
                        burst_mean_on_ms=burst_mean_on_ms,
                        burst_mean_off_ms=burst_mean_off_ms,
                        gnb_tx_power_dbm=gnb_tx_power_dbm,
                        gnb_energy_bs_category=gnb_energy_bs_category,
                        gnb_energy_ref_config=gnb_energy_ref_config,
                        gnb_energy_reference_tx_power_dbm=gnb_energy_reference_tx_power_dbm,
                        gnb_energy_eta_mode=gnb_energy_eta_mode,
                        gnb_energy_antenna_ratio=gnb_energy_antenna_ratio,
                        gnb_energy_power_unit_w=gnb_energy_power_unit_w,
                        ue_energy_power_unit_mw=ue_energy_power_unit_mw,
                        enable_true_bwp=enable_true_bwp,
                        enable_mock=False,
                        sim_time=episode_sim_time,
                        csv_filename=eval_ns3_csv,
                        qos_target_ms=qos_target_ms,
                        controller_warmup_ms=warmup_ms,
                        controller_max_active_ms=max_active_ms,
                        controller_initial_active_ms=initial_active_ms,
                        active_window_options=active_window_options,
                        fixed_bwp_state=fixed_bwp_state,
                        enable_dci_dynamic=enable_dci_dynamic,
                        rng_seed=eval_seed,
                        port=env.port + 1,
                        weight_gnb_power=weight_gnb_power,
                        weight_ue_power=weight_ue_power,
                        weight_system_power=weight_system_power,
                        weight_latency=weight_latency,
                        weight_backlog=weight_backlog,
                        reward_mode=reward_mode,
                        backlog_safe_bytes=backlog_safe_bytes,
                        bwp_switch_penalty=bwp_switch_penalty,
                        enable_per_ue_bwp_action=enable_per_ue_bwp_action,
                    )
                    eval_env.max_steps = max_steps_per_episode
                    rewards = []
                    p99_values = []
                    gnb_powers = []
                    ue_powers = []
                    active_windows = []
                    actions = []
                    per_ue_bwp_states = []
                    action_probability_samples = []
                    action_entropy_samples = []
                    try:
                        eval_obs, _ = eval_env.reset()
                        for _ in range(max_steps_per_episode):
                            (
                                eval_action,
                                eval_probabilities,
                                eval_entropies,
                            ) = agent.deterministic_action_distribution(eval_obs)
                            eval_obs, eval_reward, terminated, truncated, eval_info = (
                                eval_env.step(eval_action)
                            )
                            rewards.append(float(eval_reward))
                            p99_values.append(float(eval_info["p99_latency_ms"]))
                            gnb_powers.append(float(eval_info["gnb_power_w"]))
                            ue_powers.append(float(eval_info["ue_power_mw"]))
                            active_windows.append(float(eval_info["w_active_ms"]))
                            actions.append(str(
                                eval_action.tolist()
                                if isinstance(eval_action, np.ndarray) else eval_action
                            ))
                            per_ue_bwp_states.append(str(eval_info.get("per_ue_bwp", [])))
                            action_probability_samples.append(eval_probabilities)
                            action_entropy_samples.append(eval_entropies)
                            if terminated or truncated:
                                break
                    finally:
                        eval_env.close()

                    qos_pct = 100.0 * float(np.mean(
                        np.asarray(p99_values) <= qos_target_ms
                    ))
                    mean_action_probabilities = [
                        np.mean([
                            sample[head]
                            for sample in action_probability_samples
                        ], axis=0)
                        for head in range(len(action_probability_samples[0]))
                    ]
                    mean_action_entropies = np.mean(
                        np.asarray(action_entropy_samples), axis=0
                    )
                    eval_rows.append({
                        "TrainingEpisode": training_episode,
                        "UdpRateMbpsPerUe": eval_rate,
                        "Seed": eval_seed,
                        "Steps": len(rewards),
                        "EpisodeReward": float(np.sum(rewards)),
                        "QosSatisfactionRatePct": qos_pct,
                        "AvgP99LatencyMs": float(np.mean(p99_values)),
                        "AvgGnbPowerW": float(np.mean(gnb_powers)),
                        "AvgUePowerMw": float(np.mean(ue_powers)),
                        "AvgSystemPowerW": float(
                            np.mean(gnb_powers)
                            + num_ues * np.mean(ue_powers) / 1000.0
                        ),
                        "AvgActiveWindowMs": float(np.mean(active_windows)),
                        "Actions": ";".join(sorted(set(actions))),
                        "PerUeBwpStates": ";".join(sorted(set(per_ue_bwp_states))),
                        "ActionProbabilities": "|".join(
                            ",".join(f"{value:.6f}" for value in probabilities)
                            for probabilities in mean_action_probabilities
                        ),
                        "ActionEntropies": ";".join(
                            f"{value:.6f}" for value in mean_action_entropies
                        ),
                        "IsFeasible": int(qos_pct >= best_qos_threshold_pct),
                        "IsBestCheckpoint": 0,
                    })
        finally:
            agent.policy.train(policy_was_training)
        return eval_rows

    print(f"\n{'Episode':<8} | {'Load/UE':<8} | {'Reward':<11} | {'P99(ms)':<9} | {'QoS(%)':<8} | {'EE(Mb/J)':<10} | {'gNB(W)':<8} | {'UE(mW)':<8} | {'Loss':<10}")
    print("-" * 118)

    try:
        global_step = 0
        for episode in range(1, num_episodes + 1):
            episode_udp_rate = float(traffic_schedule[episode - 1])
            env.udp_rate = episode_udp_rate
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
            ep_per_ue_wide_bwp = [[] for _ in range(num_ues)]
            ep_sla_violations = []
            ep_rewards = []
            ep_reward_gnb = []
            ep_reward_ue = []
            ep_reward_system = []
            ep_reward_latency = []
            ep_reward_backlog = []
            ep_qos_margins = []
            ep_sla_violation_ms = []
            ep_jain_fairness = []
            previous_bwp_state = int(info["bwp_state"])
            previous_per_ue_bwp = list(info.get("per_ue_bwp", [previous_bwp_state]))
            per_ue_bwp_switch_count = [0] * num_ues
            bwp_switch_count = 0

            for step in range(max_steps_per_episode):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                global_step += 1

                agent.buffer.add(
                    obs, action, log_prob, reward, value, done,
                    context=episode_udp_rate,
                )

                obs = next_obs
                episode_reward += reward
                reward_breakdown = step_info["reward_breakdown"]
                ep_rewards.append(reward)
                ep_reward_gnb.append(reward_breakdown["reward_gnb_power"])
                ep_reward_ue.append(reward_breakdown["reward_ue_power"])
                ep_reward_system.append(reward_breakdown["reward_system_power"])
                ep_reward_latency.append(reward_breakdown["reward_latency"])
                ep_reward_backlog.append(reward_breakdown["reward_backlog"])
                ep_qos_margins.append(reward_breakdown["qos_margin_ms"])
                ep_sla_violation_ms.append(reward_breakdown["sla_violation_ms"])
                ep_jain_fairness.append(step_info["jain_fairness"])
                if int(step_info["bwp_state"]) != previous_bwp_state:
                    bwp_switch_count += 1
                previous_bwp_state = int(step_info["bwp_state"])
                current_per_ue_bwp = list(
                    step_info.get("per_ue_bwp", [step_info["bwp_state"]])
                )
                for ue_id in range(min(num_ues, len(current_per_ue_bwp))):
                    current_value = int(current_per_ue_bwp[ue_id])
                    ep_per_ue_wide_bwp[ue_id].append(float(current_value == 1))
                    if (ue_id < len(previous_per_ue_bwp)
                            and current_value != int(previous_per_ue_bwp[ue_id])):
                        per_ue_bwp_switch_count[ue_id] += 1
                previous_per_ue_bwp = current_per_ue_bwp

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

                control_interval_sec = 0.040
                step_delivered_mbits = (
                    step_info["delivered_rate_mbps"] * control_interval_sec
                )
                step_gnb_energy_j = (
                    step_info["gnb_power_w"] * control_interval_sec
                )
                step_ue_energy_j = (
                    step_info["ue_power_mw"] / 1000.0
                    * control_interval_sec * num_ues
                )
                step_total_energy_j = step_gnb_energy_j + step_ue_energy_j
                step_writer.writerow({
                    "Episode": episode,
                    "Step": step + 1,
                    "GlobalStep": global_step,
                    "ControlElapsedSec": (step + 1) * control_interval_sec,
                    "UdpRateMbpsPerUe": episode_udp_rate,
                    "OfferedRateNormalizationMbps": offered_rate_max_mbps,
                    "Action": action,
                    "ActionEncoding": step_info["action_encoding"],
                    "TargetActiveWindowMs": step_info["target_w_active_ms"],
                    "DeltaWindowMs": step_info["delta_w_ms"],
                    "ActionBwpState": step_info["action_bwp_state"],
                    "Reward": reward,
                    "RewardGnbPower": reward_breakdown["reward_gnb_power"],
                    "RewardUePower": reward_breakdown["reward_ue_power"],
                    "RewardSystemPower": reward_breakdown["reward_system_power"],
                    "RewardLatency": reward_breakdown["reward_latency"],
                    "RewardBacklog": reward_breakdown["reward_backlog"],
                    "RewardBwpSwitch": reward_breakdown["reward_bwp_switch"],
                    "BwpSwitchPenaltyCount": reward_breakdown["bwp_switch_count"],
                    "GnbPowerNormalized": reward_breakdown["p_gnb_norm"],
                    "UePowerNormalized": reward_breakdown["p_ue_norm"],
                    "SystemPowerNormalized": reward_breakdown["p_system_norm"],
                    "TotalUePowerW": reward_breakdown["total_ue_power_w"],
                    "TotalSystemPowerW": reward_breakdown["total_system_power_w"],
                    "LatencyPenalty": reward_breakdown["penalty_p99"],
                    "LatencyExcessRatio": reward_breakdown["latency_excess_ratio"],
                    "BacklogNormalized": reward_breakdown["backlog_norm"],
                    "RawBacklogNormalized": reward_breakdown["raw_backlog_norm"],
                    "BacklogExcessNormalized": reward_breakdown["backlog_excess_norm"],
                    "BacklogSafeBytes": reward_breakdown["backlog_safe_bytes"],
                    "RewardMode": reward_breakdown["reward_mode"],
                    "ValueEstimate": value,
                    "ActionLogProb": log_prob,
                    "P99LatencyMs": step_info["p99_latency_ms"],
                    "QosTargetMs": qos_target_ms,
                    "QosSatisfied": reward_breakdown["qos_satisfied"],
                    "QosMarginMs": reward_breakdown["qos_margin_ms"],
                    "SlaViolationMs": reward_breakdown["sla_violation_ms"],
                    "BacklogBytes": step_info["backlog_bytes"],
                    "MaxUeBacklogBytes": step_info["max_ue_backlog_bytes"],
                    "OfferedRateMbps": step_info["offered_rate_mbps"],
                    "DeliveredRateMbps": step_info["delivered_rate_mbps"],
                    "StepDeliveredMbits": step_delivered_mbits,
                    "GnbPowerW": step_info["gnb_power_w"],
                    "StepGnbEnergyJ": step_gnb_energy_j,
                    "GnbEnergyEfficiencyMbitPerJ": (
                        step_delivered_mbits / step_gnb_energy_j
                        if step_gnb_energy_j > 0.0 else float("nan")
                    ),
                    "UePowerMwPerUe": step_info["ue_power_mw"],
                    "StepUeEnergyJAllUes": step_ue_energy_j,
                    "TotalNetworkEnergyJ": step_total_energy_j,
                    "NetworkEnergyEfficiencyMbitPerJ": (
                        step_delivered_mbits / step_total_energy_j
                        if step_total_energy_j > 0.0 else float("nan")
                    ),
                    "AvgCqi": step_info["avg_cqi"],
                    "P10Cqi": step_info["p10_cqi"],
                    "MinCqi": step_info["min_cqi"],
                    "AvgDlMcs": step_info["avg_dl_mcs"],
                    "JainFairness": step_info["jain_fairness"],
                    "ActiveWindowMs": step_info["w_active_ms"],
                    "BwpState": step_info["bwp_state"],
                    "IsWideBwp": int(step_info["bwp_state"] == 1),
                    "PerUeBwpState": ";".join(str(int(value)) for value in current_per_ue_bwp),
                    "Terminated": int(terminated),
                    "Truncated": int(truncated),
                    "Backend": step_info["backend"],
                })

                if done:
                    break

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
            per_ue_wide_bwp_rate_pct = [
                100.0 * np.mean(values) if values else 0.0
                for values in ep_per_ue_wide_bwp
            ]
            if enable_per_ue_bwp_action:
                wide_bwp_rate_pct = float(np.mean(per_ue_wide_bwp_rate_pct))
                active_steps = min(
                    (len(values) for values in ep_per_ue_wide_bwp),
                    default=0,
                )
                any_ue_wide_bwp_rate_pct = (
                    100.0 * np.mean([
                        any(ep_per_ue_wide_bwp[ue_id][step] > 0.5
                            for ue_id in range(num_ues))
                        for step in range(active_steps)
                    ]) if active_steps else 0.0
                )
                all_ue_wide_bwp_rate_pct = (
                    100.0 * np.mean([
                        all(ep_per_ue_wide_bwp[ue_id][step] > 0.5
                            for ue_id in range(num_ues))
                        for step in range(active_steps)
                    ]) if active_steps else 0.0
                )
            else:
                wide_bwp_rate_pct = 100.0 * np.mean(ep_wide_bwp)
                any_ue_wide_bwp_rate_pct = wide_bwp_rate_pct
                all_ue_wide_bwp_rate_pct = wide_bwp_rate_pct
            sla_violation_rate = np.mean(ep_sla_violations)
            qos_satisfaction_rate_pct = 100.0 * (1.0 - sla_violation_rate)
            terminal_reward = (
                float(terminal_qos_bonus)
                if qos_satisfaction_rate_pct >= best_qos_threshold_pct
                else 0.0
            )
            # Attach the episode-level outcome to the final transition. GAE
            # then propagates it backward while preserving dense per-step
            # energy/QoS shaping rewards.
            if terminal_reward != 0.0 and agent.buffer.rewards:
                agent.buffer.rewards[-1] += terminal_reward
            episode_reward += terminal_reward
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
            total_network_energy_j = (
                gnb_control_energy_j + ue_control_energy_j_all_ues
            )
            network_energy_efficiency = (
                total_delivered_mbits / total_network_energy_j
                if total_network_energy_j > 0.0 else float("nan")
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

            rolling_count = min(best_window, len(history["episode_rewards"]))
            rolling_reward = float(np.mean(
                history["episode_rewards"][-rolling_count:]
            ))
            rolling_qos_pct = float(100.0 * (1.0 - np.mean(
                history["sla_violation_rates"][-rolling_count:]
            )))
            is_best_checkpoint = False
            deterministic_eval_performed = False
            deterministic_eval_mean_reward = float("nan")
            deterministic_eval_min_qos_pct = float("nan")
            deterministic_eval_mean_gnb_power = float("nan")
            deterministic_eval_mean_ue_power = float("nan")
            deterministic_eval_mean_active_window = float("nan")
            deterministic_eval_actions = ""

            should_update = (
                len(agent.buffer.rewards) >= rollout_samples_target
                or episode == num_episodes
            )
            if should_update:
                update_metrics = agent.update()
            else:
                update_metrics = {
                    "loss": float("nan"),
                    "policy_loss": float("nan"),
                    "value_loss": float("nan"),
                    "entropy": float("nan"),
                    "approx_kl": float("nan"),
                    "clip_fraction": float("nan"),
                    "explained_variance": float("nan"),
                    "value_target_mean": float("nan"),
                    "value_target_std": float("nan"),
                    "value_prediction_mean": float("nan"),
                    "value_prediction_std": float("nan"),
                    "value_return_correlation": float("nan"),
                    "normalized_value_error": float("nan"),
                    "actor_grad_norm": float("nan"),
                    "critic_grad_norm": float("nan"),
                    "feature_grad_norm": float("nan"),
                    "epoch_kl_trace": "",
                    "epoch_clip_fraction_trace": "",
                    "action_counts": "",
                    "action_frequencies": "",
                    "epochs_completed": 0,
                    "samples": 0,
                }
            loss_val = update_metrics["loss"]
            history["losses"].append(loss_val)

            should_deterministic_eval = should_update and (
                episode >= next_deterministic_eval_episode
                or episode == num_episodes
            )
            if should_deterministic_eval:
                deterministic_eval_performed = True
                eval_rows = evaluate_policy_deterministically(episode)
                deterministic_eval_mean_reward = float(np.mean([
                    row["EpisodeReward"] for row in eval_rows
                ]))
                deterministic_eval_min_qos_pct = float(min(
                    row["QosSatisfactionRatePct"] for row in eval_rows
                ))
                deterministic_eval_mean_gnb_power = float(np.mean([
                    row["AvgGnbPowerW"] for row in eval_rows
                ]))
                deterministic_eval_mean_ue_power = float(np.mean([
                    row["AvgUePowerMw"] for row in eval_rows
                ]))
                deterministic_eval_mean_system_power = float(np.mean([
                    row["AvgSystemPowerW"] for row in eval_rows
                ]))
                deterministic_eval_mean_active_window = float(np.mean([
                    row["AvgActiveWindowMs"] for row in eval_rows
                ]))
                deterministic_eval_actions = "|".join(sorted(set(
                    row["Actions"] for row in eval_rows
                )))
                candidate_is_feasible = (
                    deterministic_eval_min_qos_pct >= best_qos_threshold_pct
                )
                is_best_checkpoint = candidate_is_feasible and (
                    deterministic_eval_mean_system_power
                    < best_deterministic_system_power - 1e-9
                    or (
                        abs(
                            deterministic_eval_mean_system_power
                            - best_deterministic_system_power
                        )
                        <= 1e-9
                        and deterministic_eval_mean_reward > best_score
                    )
                )
                if is_best_checkpoint:
                    best_score = deterministic_eval_mean_reward
                    best_deterministic_system_power = (
                        deterministic_eval_mean_system_power
                    )
                    best_episode = episode
                    torch.save(agent.policy.state_dict(), best_model_out)
                    try:
                        torch.jit.script(agent.policy).save(best_torchscript_out)
                    except Exception as exc:
                        print(f"⚠️ Best TorchScript 匯出提示: {exc}")
                for eval_row in eval_rows:
                    eval_row["IsBestCheckpoint"] = int(is_best_checkpoint)
                    deterministic_eval_writer.writerow(eval_row)
                deterministic_eval_file.flush()
                print(
                    f"🎯 Deterministic eval @ episode {episode}: "
                    f"reward={deterministic_eval_mean_reward:.4f}, "
                    f"min QoS={deterministic_eval_min_qos_pct:.1f}%, "
                    f"gNB={deterministic_eval_mean_gnb_power:.2f}W, "
                    f"active={deterministic_eval_mean_active_window:.2f}ms, "
                    f"actions={deterministic_eval_actions}"
                )
                if is_best_checkpoint:
                    print(
                        f"🏆 Best deterministic model @ episode {episode}: "
                        f"system={best_deterministic_system_power:.2f}W, "
                        f"reward={best_score:.4f}, "
                        f"min QoS={deterministic_eval_min_qos_pct:.1f}%"
                    )
                while next_deterministic_eval_episode <= episode:
                    next_deterministic_eval_episode += deterministic_eval_interval

            if should_update and (
                episode >= next_checkpoint_episode or episode == num_episodes
            ):
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_episode_{episode:04d}.pt"
                )
                torch.save({
                    "episode": episode,
                    "policy_state_dict": agent.policy.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "best_episode": best_episode,
                    "best_deterministic_reward": best_score,
                    "best_deterministic_system_power_w": best_deterministic_system_power,
                    "episodes_per_update": episodes_per_update,
                    "learning_rate": learning_rate,
                    "value_loss_coefficient": value_loss_coefficient,
                    "ppo_epochs": ppo_epochs,
                    "batch_size": batch_size,
                    "target_kl": target_kl,
                    "action_encoding": (
                        S5_ACTION_ENCODING if enable_per_ue_bwp_action
                        else ACTION_ENCODING
                    ),
                    "action_space_size": (
                        env.action_space.nvec.tolist()
                        if enable_per_ue_bwp_action else env.action_space.n
                    ),
                    "safe_prior_actions": safe_prior_actions,
                    "safe_prior_mass": safe_prior_mass if safe_prior_actions else 0.0,
                    "active_window_options_ms": env.ACTIVE_WINDOW_OPTIONS,
                    "udp_rate_choices_mbps_per_ue": traffic_rates,
                    "offered_rate_normalization_mbps": offered_rate_max_mbps,
                    "reward_mode": reward_mode,
                    "weight_system_power": weight_system_power,
                    "weight_latency": weight_latency,
                    "weight_backlog": weight_backlog,
                    "backlog_safe_bytes": backlog_safe_bytes,
                    "total_system_power_normalization_w": env.P_TOTAL_MAX_W,
                }, checkpoint_path)
                while next_checkpoint_episode <= episode:
                    next_checkpoint_episode += checkpoint_interval

            history_writer.writerow({
                "Episode": episode,
                "EpisodeReward": episode_reward,
                "AvgStepReward": np.mean(ep_rewards),
                "MinStepReward": np.min(ep_rewards),
                "MaxStepReward": np.max(ep_rewards),
                "StdStepReward": np.std(ep_rewards),
                "RewardGnbPower": np.sum(ep_reward_gnb),
                "RewardUePower": np.sum(ep_reward_ue),
                "RewardSystemPower": np.sum(ep_reward_system),
                "RewardLatency": np.sum(ep_reward_latency),
                "RewardBacklog": np.sum(ep_reward_backlog),
                "TerminalReward": terminal_reward,
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
                "TotalNetworkEnergyJ": total_network_energy_j,
                "NetworkEnergyEfficiencyMbitPerJ": network_energy_efficiency,
                "MinOfferedRateMbps": min_offered_rate,
                "MaxOfferedRateMbps": max_offered_rate,
                "StdOfferedRateMbps": std_offered_rate,
                "AvgCqi": avg_cqi,
                "AvgP10Cqi": avg_p10_cqi,
                "AvgMinCqi": avg_min_cqi,
                "AvgDlMcs": avg_dl_mcs,
                "AvgActiveWindowMs": avg_active_window,
                "WideBwpRatePct": wide_bwp_rate_pct,
                "PerUeWideBwpRatePct": ";".join(
                    f"{value:.6g}" for value in per_ue_wide_bwp_rate_pct
                ),
                "MeanWideBwpRatePct": wide_bwp_rate_pct,
                "AnyUeWideBwpRatePct": any_ue_wide_bwp_rate_pct,
                "AllUeWideBwpRatePct": all_ue_wide_bwp_rate_pct,
                "BwpSwitchCount": bwp_switch_count,
                "PerUeBwpSwitchCount": ";".join(
                    str(int(value)) for value in per_ue_bwp_switch_count
                ),
                "SlaViolationRate": sla_violation_rate,
                "QosSatisfactionRatePct": qos_satisfaction_rate_pct,
                "AvgQosMarginMs": np.mean(ep_qos_margins),
                "AvgSlaViolationMs": np.mean(ep_sla_violation_ms),
                "AvgJainFairness": np.mean(ep_jain_fairness),
                "Loss": loss_val,
                "PolicyLoss": update_metrics["policy_loss"],
                "ValueLoss": update_metrics["value_loss"],
                "Entropy": update_metrics["entropy"],
                "ApproxKl": update_metrics["approx_kl"],
                "ClipFraction": update_metrics["clip_fraction"],
                "ExplainedVariance": update_metrics["explained_variance"],
                "ValueTargetMean": update_metrics["value_target_mean"],
                "ValueTargetStd": update_metrics["value_target_std"],
                "ValuePredictionMean": update_metrics["value_prediction_mean"],
                "ValuePredictionStd": update_metrics["value_prediction_std"],
                "ValueReturnCorrelation": update_metrics["value_return_correlation"],
                "NormalizedValueError": update_metrics["normalized_value_error"],
                "ActorGradNorm": update_metrics["actor_grad_norm"],
                "CriticGradNorm": update_metrics["critic_grad_norm"],
                "FeatureGradNorm": update_metrics["feature_grad_norm"],
                "EpochApproxKlTrace": update_metrics["epoch_kl_trace"],
                "EpochClipFractionTrace": update_metrics["epoch_clip_fraction_trace"],
                "UpdateActionCounts": update_metrics["action_counts"],
                "UpdateActionFrequencies": update_metrics["action_frequencies"],
                "PpoUpdated": int(should_update),
                "PpoUpdateSamples": update_metrics["samples"],
                "PpoEpochsCompleted": update_metrics["epochs_completed"],
                "RollingReward": rolling_reward,
                "RollingQosSatisfactionRatePct": rolling_qos_pct,
                "DeterministicEvalPerformed": int(deterministic_eval_performed),
                "DeterministicEvalMeanReward": deterministic_eval_mean_reward,
                "DeterministicEvalMinQosPct": deterministic_eval_min_qos_pct,
                "DeterministicEvalMeanGnbPowerW": deterministic_eval_mean_gnb_power,
                "DeterministicEvalMeanUePowerMw": deterministic_eval_mean_ue_power,
                "DeterministicEvalMeanActiveWindowMs": deterministic_eval_mean_active_window,
                "DeterministicEvalActions": deterministic_eval_actions,
                "IsBestCheckpoint": int(is_best_checkpoint),
                "EnergyModel": "5G-LENA-TR38.864-TR38.840",
                "FrequencyControlMode": (
                    "physical-20-100-MHz-BWP"
                    if enable_true_bwp else "legacy-PRB-profile"
                ),
                "PhysicalBandwidthMHz": 100,
                "GnbEnergyBsCategory": gnb_energy_bs_category,
                "GnbEnergyRefConfig": gnb_energy_ref_config,
                "GnbEnergyReferenceTxPowerDbm": gnb_energy_reference_tx_power_dbm,
                "GnbEnergyEtaMode": gnb_energy_eta_mode,
                "GnbEnergyAntennaRatio": gnb_energy_antenna_ratio,
                "GnbEnergyPowerUnitW": gnb_energy_power_unit_w,
                "UeEnergyPowerUnitMw": ue_energy_power_unit_mw,
                "QosTargetMs": qos_target_ms,
                "EnableDciDynamic": int(enable_dci_dynamic),
                "ControlIntervalMs": 40.0,
                "NumUes": num_ues,
                "UdpRateMbpsPerUe": episode_udp_rate,
                "OfferedRateNormalizationMbps": offered_rate_max_mbps,
                "Seed": seed,
                "LearningRate": learning_rate,
                "ValueLossCoefficient": value_loss_coefficient,
                "ConfiguredPpoEpochs": ppo_epochs,
                "BatchSize": batch_size,
                "TargetKl": target_kl,
                "EpisodesPerUpdate": episodes_per_update,
                "ActionEncoding": (
                    S5_ACTION_ENCODING if enable_per_ue_bwp_action
                    else ACTION_ENCODING
                ),
                "ActionSpaceSize": (
                    ",".join(str(int(value)) for value in env.action_space.nvec)
                    if enable_per_ue_bwp_action else env.action_space.n
                ),
                "SafePriorActions": ";".join(str(action) for action in safe_prior_actions),
                "SafePriorMass": safe_prior_mass if safe_prior_actions else 0.0,
                "RewardMode": reward_mode,
                "WeightSystemPower": weight_system_power,
                "WeightLatency": weight_latency,
                "WeightBacklog": weight_backlog,
                "BacklogSafeBytes": backlog_safe_bytes,
                "TotalSystemPowerNormalizationW": env.P_TOTAL_MAX_W,
            })
            history_file.flush()
            step_file.flush()

            loss_text = f"{loss_val:.4f}" if np.isfinite(loss_val) else "-"
            print(
                f"{episode:<8} | {episode_udp_rate:<8.2f} | "
                f"{episode_reward:<11.4f} | {avg_p99:<9.2f} | "
                f"{qos_satisfaction_rate_pct:<8.1f} | {network_energy_efficiency:<10.4f} | "
                f"{avg_gnb:<8.2f} | {avg_ue:<8.2f} | {loss_text:<10}"
            )
            if should_update:
                print(
                    "  🔎 PPO update: "
                    f"return={update_metrics['value_target_mean']:.3f}±"
                    f"{update_metrics['value_target_std']:.3f} | "
                    f"value={update_metrics['value_prediction_mean']:.3f}±"
                    f"{update_metrics['value_prediction_std']:.3f} | "
                    f"corr={update_metrics['value_return_correlation']:.3f} | "
                    f"norm-MSE={update_metrics['normalized_value_error']:.3f} | "
                    f"grad(a/c/f)={update_metrics['actor_grad_norm']:.3g}/"
                    f"{update_metrics['critic_grad_norm']:.3g}/"
                    f"{update_metrics['feature_grad_norm']:.3g}"
                )
    finally:
        env.close()
        history_file.close()
        step_file.close()
        deterministic_eval_file.close()

    elapsed_sec = time.time() - start_time
    print("-" * 110)
    print(f"✅ 連續狀態 PPO 代理人訓練完成！總耗時: {elapsed_sec:.2f} 秒")

    model_save_path = os.path.abspath(model_out)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(agent.policy.state_dict(), model_save_path)
    print(f"💾 模型權重已成功儲存至: {model_save_path}")
    print(f"📊 訓練紀錄已儲存至: {history_csv}")
    print(f"🔬 每步實驗紀錄已儲存至: {step_csv}")
    if best_episode is not None:
        print(
            f"🏆 最佳模型: episode {best_episode}, "
            f"deterministic reward={best_score:.4f}\n"
            f"   PyTorch: {best_model_out}\n"
            f"   TorchScript: {best_torchscript_out}"
        )
    else:
        print(
            f"⚠️ 沒有 deterministic evaluation 達到 QoS >= "
            f"{best_qos_threshold_pct:.1f}%；"
            "未產生 best model。"
        )

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
    parser.add_argument(
        "--udp-rate-choices",
        type=float,
        nargs="+",
        default=None,
        help=(
            "randomize the per-UE Mbps rate per episode using balanced "
            "shuffled blocks (for example: 0.5 1 2 4)"
        ),
    )
    parser.add_argument(
        "--udp-rate-block-size", type=int, default=1,
        help="hold each --udp-rate-choices context for this many episodes; 1 keeps shuffled schedule",
    )
    parser.add_argument("--enable-burst-traffic", action="store_true")
    parser.add_argument("--synchronize-burst-traffic", action="store_true")
    parser.add_argument("--udp-burst-rate", type=float, default=3.0)
    parser.add_argument("--burst-mean-on-ms", type=float, default=80.0)
    parser.add_argument("--burst-mean-off-ms", type=float, default=120.0)
    parser.add_argument("--gnb-tx-power-dbm", type=float, default=30.0)
    parser.add_argument("--gnb-energy-bs-category", choices=("BsCat1", "BsCat2"), default="BsCat1")
    parser.add_argument(
        "--gnb-energy-ref-config",
        choices=("Custom", "Set1", "Set2", "Set3"),
        default="Custom",
    )
    parser.add_argument(
        "--gnb-energy-reference-tx-power-dbm", type=float, default=30.0,
        help="reference Tx power used by the Custom gNB energy configuration",
    )
    parser.add_argument("--gnb-energy-eta-mode", choices=("Single", "Dual"), default="Dual")
    parser.add_argument("--gnb-energy-antenna-ratio", type=float, default=0.4)
    parser.add_argument("--gnb-energy-power-unit-w", type=float, default=1.0)
    parser.add_argument("--ue-energy-power-unit-mw", type=float, default=1.0)
    parser.add_argument(
        "--true-bwp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use physical 20/100 MHz BWPs (default); --no-true-bwp uses the legacy PRB profile",
    )
    parser.add_argument(
        "--enable-per-ue-bwp-action",
        action="store_true",
        help="Use S5 MultiDiscrete action: active-window index plus one BWP action per UE",
    )
    parser.add_argument("--history-csv", default="ppo_training_history.csv")
    parser.add_argument("--step-csv", default="ppo_training_steps.csv")
    parser.add_argument("--ns3-csv", default="multi_user_rl_step.csv")
    parser.add_argument("--model-out", default="ppo_5g_joint2d_model.pt")
    parser.add_argument("--torchscript-out", default="ppo_5g_joint2d_torchscript.pt")
    parser.add_argument("--best-model-out", default="ppo_5g_joint2d_best_model.pt")
    parser.add_argument(
        "--best-torchscript-out", default="ppo_5g_joint2d_best_torchscript.pt"
    )
    parser.add_argument("--checkpoint-dir", default="ppo_checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument(
        "--rollout-samples", type=int, default=0,
        help="target on-policy samples per PPO update; 0 derives from episodes-per-update",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--context-conditioned-actor", action="store_true",
        help="concatenate normalized offered-rate observation to actor heads",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--best-qos-threshold-pct", type=float, default=95.0)
    parser.add_argument("--best-window", type=int, default=4)
    parser.add_argument(
        "--deterministic-eval-interval", type=int, default=4,
        help="evaluate the greedy policy every N training episodes after an update",
    )
    parser.add_argument(
        "--deterministic-eval-seeds", type=int, nargs="+", default=[101],
        help="fixed ns-3 seeds used for deterministic best-model selection",
    )
    parser.add_argument(
        "--deterministic-eval-csv", default=None,
        help="evaluation CSV path; default derives from --history-csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qos-target-ms", type=float, default=30.0)
    parser.add_argument("--warmup-ms", type=float, default=200.0)
    parser.add_argument("--max-active-ms", type=float, default=30.0)
    parser.add_argument("--initial-active-ms", type=float, default=30.0)
    parser.add_argument(
        "--active-window-options", type=float, nargs="+", default=None,
        help="explicit active-window action options, e.g. 10 12 14 16",
    )
    parser.add_argument(
        "--fixed-bwp-state", type=int, choices=(0, 1), default=None,
        help="diagnostic mode: fix all per-UE BWP actions to 0 (20 MHz) or 1 (100 MHz)",
    )
    parser.add_argument(
        "--enable-dci-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="couple Cell-DTX active-window actions to UE C-DRX on-duration",
    )
    parser.add_argument("--weight-gnb-power", type=float, default=0.15)
    parser.add_argument("--weight-ue-power", type=float, default=0.15)
    parser.add_argument("--weight-system-power", type=float, default=0.30)
    parser.add_argument("--weight-latency", type=float, default=0.45)
    parser.add_argument("--weight-backlog", type=float, default=0.25)
    parser.add_argument(
        "--terminal-qos-bonus", type=float, default=0.0,
        help="episode-end bonus when QoS satisfaction reaches the best-QoS threshold",
    )
    parser.add_argument(
        "--reward-mode",
        choices=(
            "total-system-threshold-v1",
            "balanced-system-threshold-v1",
            "legacy-separated-v1",
        ),
        default=REWARD_MODE,
    )
    parser.add_argument("--backlog-safe-bytes", type=float, default=50000.0)
    parser.add_argument(
        "--bwp-switch-penalty", type=float, default=0.0,
        help="per-UE BWP switch penalty added to each step reward",
    )
    parser.add_argument(
        "--safe-prior-active-ms",
        type=float,
        nargs="+",
        default=None,
        help="initialize a prior over these 100 MHz absolute active-window actions",
    )
    parser.add_argument(
        "--safe-prior-mass",
        type=float,
        default=0.7,
        help="initial total probability mass assigned to --safe-prior-active-ms",
    )
    parser.add_argument("--rerun-train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num-episodes must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.episodes_per_update <= 0:
        parser.error("--episodes-per-update must be greater than zero")
    if args.rollout_samples < 0:
        parser.error("--rollout-samples cannot be negative")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be greater than zero")
    if args.entropy_coef < 0.0:
        parser.error("--entropy-coef cannot be negative")
    if args.value_loss_coef < 0.0:
        parser.error("--value-loss-coef cannot be negative")
    if args.bwp_switch_penalty < 0.0:
        parser.error("--bwp-switch-penalty cannot be negative")
    if args.ppo_epochs <= 0 or args.batch_size <= 0:
        parser.error("--ppo-epochs and --batch-size must be greater than zero")
    if args.target_kl <= 0.0:
        parser.error("--target-kl must be greater than zero")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be greater than zero")
    if args.best_window <= 0:
        parser.error("--best-window must be greater than zero")
    if args.deterministic_eval_interval <= 0:
        parser.error("--deterministic-eval-interval must be greater than zero")
    if not args.deterministic_eval_seeds:
        parser.error("--deterministic-eval-seeds must not be empty")
    if not 0.0 <= args.best_qos_threshold_pct <= 100.0:
        parser.error("--best-qos-threshold-pct must be inside [0, 100]")
    if args.num_ues <= 0:
        parser.error("--num-ues must be greater than zero")
    if args.udp_rate <= 0:
        parser.error("--udp-rate must be greater than zero")
    if args.udp_rate_choices is not None and (
        not args.udp_rate_choices
        or any(rate <= 0.0 for rate in args.udp_rate_choices)
    ):
        parser.error("--udp-rate-choices values must be greater than zero")
    if args.udp_rate_block_size <= 0:
        parser.error("--udp-rate-block-size must be greater than zero")
    if args.enable_burst_traffic and args.udp_burst_rate <= 0:
        parser.error("--udp-burst-rate must be greater than zero")
    if args.enable_burst_traffic and (
        args.burst_mean_on_ms <= 0 or args.burst_mean_off_ms <= 0
    ):
        parser.error("burst On/Off means must be greater than zero")
    if not -50.0 <= args.gnb_tx_power_dbm <= 60.0:
        parser.error("--gnb-tx-power-dbm must be between -50 and 60")
    if not -50.0 <= args.gnb_energy_reference_tx_power_dbm <= 60.0:
        parser.error("--gnb-energy-reference-tx-power-dbm must be between -50 and 60")
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
        args.weight_system_power,
        args.weight_latency,
        args.weight_backlog,
    ]
    if any(weight < 0 for weight in reward_weights):
        parser.error("reward weights cannot be negative")
    if args.terminal_qos_bonus < 0.0:
        parser.error("terminal QoS bonus cannot be negative")
    active_reward_weights = (
        [args.weight_system_power, args.weight_latency, args.weight_backlog]
        if args.reward_mode == "total-system-threshold-v1"
        else [
            args.weight_gnb_power,
            args.weight_ue_power,
            args.weight_latency,
            args.weight_backlog,
        ]
    )
    if sum(active_reward_weights) <= 0:
        parser.error("at least one active reward weight must be positive")
    if not 0.0 <= args.backlog_safe_bytes < 1000000.0:
        parser.error("--backlog-safe-bytes must be in [0, 1000000)")
    if args.safe_prior_active_ms is not None and not 0.0 < args.safe_prior_mass < 1.0:
        parser.error("--safe-prior-mass must be inside (0, 1)")

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train_ppo_agent(
        num_episodes=cli_args.num_episodes,
        max_steps_per_episode=cli_args.max_steps,
        num_ues=cli_args.num_ues,
        udp_rate=cli_args.udp_rate,
        udp_rate_choices=cli_args.udp_rate_choices,
        udp_rate_block_size=cli_args.udp_rate_block_size,
        enable_burst_traffic=cli_args.enable_burst_traffic,
        synchronize_burst_traffic=cli_args.synchronize_burst_traffic,
        udp_burst_rate=cli_args.udp_burst_rate,
        burst_mean_on_ms=cli_args.burst_mean_on_ms,
        burst_mean_off_ms=cli_args.burst_mean_off_ms,
        gnb_tx_power_dbm=cli_args.gnb_tx_power_dbm,
        gnb_energy_bs_category=cli_args.gnb_energy_bs_category,
        gnb_energy_ref_config=cli_args.gnb_energy_ref_config,
        gnb_energy_reference_tx_power_dbm=cli_args.gnb_energy_reference_tx_power_dbm,
        gnb_energy_eta_mode=cli_args.gnb_energy_eta_mode,
        gnb_energy_antenna_ratio=cli_args.gnb_energy_antenna_ratio,
        gnb_energy_power_unit_w=cli_args.gnb_energy_power_unit_w,
        ue_energy_power_unit_mw=cli_args.ue_energy_power_unit_mw,
        enable_true_bwp=cli_args.true_bwp,
        history_csv=cli_args.history_csv,
        step_csv=cli_args.step_csv,
        ns3_csv=cli_args.ns3_csv,
        model_out=cli_args.model_out,
        torchscript_out=cli_args.torchscript_out,
        best_model_out=cli_args.best_model_out,
        best_torchscript_out=cli_args.best_torchscript_out,
        checkpoint_dir=cli_args.checkpoint_dir,
        checkpoint_interval=cli_args.checkpoint_interval,
        episodes_per_update=cli_args.episodes_per_update,
        rollout_samples=cli_args.rollout_samples,
        learning_rate=cli_args.learning_rate,
        value_loss_coefficient=cli_args.value_loss_coef,
        entropy_coefficient=cli_args.entropy_coef,
        context_conditioned_actor=cli_args.context_conditioned_actor,
        ppo_epochs=cli_args.ppo_epochs,
        batch_size=cli_args.batch_size,
        target_kl=cli_args.target_kl,
        best_qos_threshold_pct=cli_args.best_qos_threshold_pct,
        best_window=cli_args.best_window,
        deterministic_eval_interval=cli_args.deterministic_eval_interval,
        deterministic_eval_seeds=cli_args.deterministic_eval_seeds,
        deterministic_eval_csv=cli_args.deterministic_eval_csv,
        seed=cli_args.seed,
        qos_target_ms=cli_args.qos_target_ms,
        warmup_ms=cli_args.warmup_ms,
        max_active_ms=cli_args.max_active_ms,
        initial_active_ms=cli_args.initial_active_ms,
        active_window_options=cli_args.active_window_options,
        fixed_bwp_state=cli_args.fixed_bwp_state,
        enable_dci_dynamic=cli_args.enable_dci_dynamic,
        weight_gnb_power=cli_args.weight_gnb_power,
        weight_ue_power=cli_args.weight_ue_power,
        weight_system_power=cli_args.weight_system_power,
        weight_latency=cli_args.weight_latency,
        weight_backlog=cli_args.weight_backlog,
        reward_mode=cli_args.reward_mode,
        backlog_safe_bytes=cli_args.backlog_safe_bytes,
        terminal_qos_bonus=cli_args.terminal_qos_bonus,
        bwp_switch_penalty=cli_args.bwp_switch_penalty,
        safe_prior_active_ms=cli_args.safe_prior_active_ms,
        safe_prior_mass=cli_args.safe_prior_mass,
        enable_per_ue_bwp_action=cli_args.enable_per_ue_bwp_action,
    )
