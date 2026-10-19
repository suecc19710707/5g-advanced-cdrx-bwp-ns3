# Phase 3 policy and IPC physical-BWP control

Phase 3 connects the previously validated physical 20/100 MHz switch to both
the built-in rule-based controller and the Python socket IPC action. It also
selects backlog/CQI/MCS observations from the currently active BWP scheduler
and applies Cell-DTX gating to both scheduler instances.

## Control behavior

- `enableTrueBwp=true, enableBwpDynamic=true, enableQueueAware=true` routes
  rule-based BWP decisions through the physical UE/gNB switch.
- `enableTrueBwp=true, enableBwpDynamic=true, enableRlIpc=true` routes the
  10-action PPO/IPC frequency dimension through the same switch.
- The legacy notched-RBG approximation remains available with
  `enableTrueBwp=false`.
- `train_drl_agent.py` and `Ns3MultiUserEnv` default to physical BWP control;
  `--no-true-bwp` retains the legacy fallback.

## Exploratory load sweep

One deterministic seed was run for 3 UEs, 2 s simulation time, constant UDP,
Cell-DTX off, and C-DRX off. Rates are Mbps per UE.

| Rate | Fixed 20 Rx (Mbit) | Fixed 100 Rx (Mbit) | Rule Rx (Mbit) | Rule final tendency |
|---:|---:|---:|---:|---|
| 0.5 | 2.767 | 2.767 | 2.767 | mostly 20 MHz |
| 1 | 5.519 | 5.519 | 5.519 | mostly 20 MHz |
| 2 | 11.023 | 11.023 | 11.023 | mixed |
| 4 | 22.057 | 22.057 | 22.057 | mixed |
| 8 | 44.098 | 44.098 | 44.098 | mixed |
| 12 | 66.140 | 66.140 | 66.140 | mostly 100 MHz |
| 20 | 110.248 | 110.248 | 110.248 | mostly 100 MHz |
| 30 | 135.790 | 165.330 | 164.809 | 100 MHz |
| 40 | 135.790 | 220.321 | 218.898 | 100 MHz |
| 60 | 135.790 | 330.377 | 327.086 | 100 MHz |

The 20 MHz path saturates near 135.79 received Mbit per 2 s, while the
100 MHz path continues to scale. At 60 Mbps/UE the rule-based controller
reaches 99.0% of fixed-100 throughput in this exploratory run.

The energy trade-off is visible throughout: fixed 20 MHz has lower UE power,
while fixed 100 MHz has lower gNB power for the tested traffic because it
uses fewer active symbols. This is a single-seed functional sweep, not the
final statistical comparison.

## Final short regression

The final working tree was rechecked with 3 UEs at 40 Mbps/UE for 1 s:

| Mode | Rx (Mbit) | P99 (ms) | gNB power (W) | UE power (mW/UE) |
|---|---:|---:|---:|---:|
| Fixed 20 MHz | 67.343 | 15.335 | 141.602 | 118.456 |
| Fixed 100 MHz | 109.389 | 11.847 | 104.208 | 296.120 |
| Rule physical BWP | 107.950 | 14.959 | 105.475 | 290.253 |

The rule switched from 20 to 100 MHz at 0.14 s and remained wide under load.

## IPC smoke test

```bash
python3 train_drl_agent.py --num-episodes 1 --max-steps 5 \
  --num-ues 3 --udp-rate 2 \
  --history-csv /tmp/v6-phase3-final-ipc-history.csv \
  --ns3-csv /tmp/v6-phase3-final-ipc-ns3.csv \
  --model-out /tmp/v6-phase3-final-ipc-model.pt \
  --torchscript-out /tmp/v6-phase3-final-ipc-torchscript.pt
```

The Python process connected to the real ns-3 backend on port 5555, did not
fall back to Mock, and applied physical BWP switches at 0.20, 0.24, 0.28, and
0.32 s. This is a connectivity/action smoke test, not evidence of PPO
convergence.

## Regression status

- target build: PASS;
- `nr-energy-integration`: PASS;
- `nr-energy-models`: PASS;
- fixed 20/100 MHz scenarios: PASS;
- scripted 20 -> 100 -> 20 MHz: PASS;
- rule-based physical switch: PASS;
- Python IPC physical switch: PASS.

The separate `phase3_dl_handover/` directory records the scheduler-buffer and
per-TB continuity test, including the explicit old-BWP K-pipeline limitation.

## Files

- `load_sweep_*.csv`: exploratory fixed/rule load-gradient results;
- `load_sweep_rule_timeline.csv`: rule decisions and backlog timeline;
- `regression_*.csv`: final short scenario checks;
- `regression_rule_timeline.csv`: final rule switch trace;
- `ipc_smoke_history.csv`, `ipc_smoke_ns3.csv`: real-backend IPC evidence.
