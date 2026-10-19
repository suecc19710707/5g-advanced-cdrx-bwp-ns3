# PPO training baseline

- Archived: 2026-10-18 14:07 (Asia/Taipei)
- Reward latency weight (`GAMMA`): 0.5
- P99 latency SLA target: 20 ms
- History: `ppo_training_history.csv`
- Plot generated from that history: `plot_training_results.png`
- PyTorch state dictionary: `ppo_5g_joint2d_model.pt`
- TorchScript model: `ppo_5g_joint2d_torchscript.pt`
- Deterministic evaluation: `deterministic_evaluation_results.csv`

Evaluation used real ns-3 IPC with 10 UEs, 2 Mbps UDP per UE, 10 evaluation
runs, and 250 steps per run. Mean P99 latency was 45.80 ms and the 20 ms
SLA violation rate was 100%. The current evaluator does not pass its Python
seed into ns-3, so all 10 rows are identical and should not be interpreted as
independent ns-3 seeds.

These are preserved copies of the baseline results before testing a higher
latency-penalty weight. The original working files were not moved or changed.
