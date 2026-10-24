#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ns3_environment import Ns3MultiUserEnv
from train_drl_agent import PPOAgent

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--rate', type=float, default=0.5)
ap.add_argument('--qos', type=float, default=50.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--max-steps', type=int, default=50)
ap.add_argument('--sim-time', type=float, default=2.3)
args = ap.parse_args()

env = Ns3MultiUserEnv(
    ns3_path='./ns3', num_ues=2, udp_rate=args.rate,
    enable_true_bwp=True, enable_per_ue_bwp_action=True,
    sim_time=args.sim_time, controller_warmup_ms=200.0,
    controller_max_active_ms=40.0, controller_initial_active_ms=20.0,
    qos_target_ms=args.qos, rng_seed=args.seed,
    port=5602,
)
env.max_steps = args.max_steps
agent = PPOAgent(state_dim=env.observation_dim, action_dim=env.action_space.nvec.tolist())
state = torch.load(args.model, map_location=agent.device, weights_only=True)
agent.policy.load_state_dict(state)
agent.policy.eval()
obs, info = env.reset()
rows=[]
try:
    for step in range(args.max_steps):
        with torch.no_grad():
            logits, value = agent.policy.forward(torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0))
            action = np.array([int(x.argmax(-1).item()) for x in logits], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({'step':step, 'active_ms':info.get('active_window_ms'), 'action':action.tolist(),
                     'per_ue_bwp':info.get('per_ue_bwp'), 'reward':reward,
                     'p99_ms':info.get('p99_latency_ms'), 'qos':None,
                     'gnb_power_w':info.get('gnb_power_w'), 'ue_power_mw':info.get('ue_power_mw')})
        if terminated or truncated: break
finally:
    env.close()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, 'w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('steps',len(rows))
print('actions', sorted(set(str(r['action']) for r in rows)))
print('per_ue_bwp', sorted(set(str(r['per_ue_bwp']) for r in rows)))
print('mean_gnb_w', np.mean([r['gnb_power_w'] for r in rows]))
print('mean_ue_mw', np.mean([r['ue_power_mw'] for r in rows]))
print('mean_p99_ms', np.mean([r['p99_ms'] for r in rows]))
print('qos_pct', 100*np.mean([float(r['p99_ms']) <= args.qos for r in rows]))
