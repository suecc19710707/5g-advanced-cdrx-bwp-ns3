# Training-ready 2D PPO checkpoint (2026-10-18)

This directory preserves the validation and training artifacts for the first
clean scheduler-controlled implementation. It does not overwrite the earlier
baseline archive.

## Final experiment configuration

- 10 UEs, each offering a constant 2 Mbps downlink UDP flow
- 100 MHz NR carrier at 3.5 GHz, numerology 1 (30 kHz SCS)
- UE distances: 30, 42, ..., 138 m
- gNB Tx power: 30 dBm (explicit; the NrGnbPhy default was only 4 dBm)
- Cell-DTX period: 40 ms; controllable Active Window: 10-30 ms
- Frequency action: centered 20 MHz-equivalent (51 PRBs) or full 100 MHz carrier
- QoS target: 30 ms end-to-end P99 delay
- Controller warm-up: 200 ms
- PPO discount gamma: 0.99
- Reward weights: gNB power 0.15, UE power 0.15, latency 0.45, RLC backlog 0.25
- RNG seed: 42; ns-3 run number increments once per episode

The reward is bounded to approximately [-1, 0] per step. The latency term is
`1 - exp(-max(0, (P99 - SLA) / SLA))`. Backlog is read directly from the
scheduler's aggregate downlink RLC buffers, not inferred from packet UIDs.

## Validation sequence

1. The original 4 dBm maximum-resource calibration could not sustain the load:
   P99 106.86 ms with a 30 ms Active Window, and P99 80.51 ms even at 40 ms.
2. With an explicit 30 dBm gNB Tx power and the original 30 ms maximum Active
   Window, maximum-resource P99 was 19.98 ms with zero 30 ms SLA violations.
3. After replacing UID-derived backlog with true scheduler RLC backlog, the
   same calibration reported 27.0 KB mean and 31.7 KB final RLC backlog.
4. `nr-test-notching` and `nr-system-test-scheduler-many-ues` both passed.
5. The final 5x50 smoke run completed through real IPC on port 5555. Episode
   loss stayed between 0.7653 and 2.2577, versus values up to 340196 before
   bounding the reward.

## Reproduction commands

Build and regression tests:

```bash
conda activate torch_cu118
cd /home/suecc3/workspace/ns-3-dev
./ns3 build
./test.py -s nr-test-notching
./test.py -s nr-system-test-scheduler-many-ues
```

Final smoke run:

```bash
python3 train_drl_agent.py \
  --num-episodes 5 --max-steps 50 \
  --num-ues 10 --udp-rate 2.0 --gnb-tx-power-dbm 30 \
  --qos-target-ms 30 --warmup-ms 200 --max-active-ms 30 --seed 42 \
  --history-csv experiment_archive/training_ready_v1_2026-10-18/smoke_final_history.csv \
  --ns3-csv experiment_archive/training_ready_v1_2026-10-18/smoke_final_ns3.csv \
  --model-out experiment_archive/training_ready_v1_2026-10-18/smoke_final_model.pt \
  --torchscript-out experiment_archive/training_ready_v1_2026-10-18/smoke_final_torchscript.pt
```

Pilot run:

```bash
python3 train_drl_agent.py \
  --num-episodes 30 --max-steps 125 \
  --num-ues 10 --udp-rate 2.0 --gnb-tx-power-dbm 30 \
  --qos-target-ms 30 --warmup-ms 200 --max-active-ms 30 --seed 42 \
  --history-csv experiment_archive/training_ready_v1_2026-10-18/pilot_30x125_history.csv \
  --ns3-csv experiment_archive/training_ready_v1_2026-10-18/pilot_30x125_ns3.csv \
  --model-out experiment_archive/training_ready_v1_2026-10-18/pilot_30x125_model.pt \
  --torchscript-out experiment_archive/training_ready_v1_2026-10-18/pilot_30x125_torchscript.pt
```

Do not treat the smoke model as the final trained model. Promote a model only
after deterministic multi-seed evaluation and rule-based comparison.

## Deterministic 10-seed evaluation

The pilot model was evaluated with argmax actions on seeds 101-110, using 125
steps per seed and the same physical/reward configuration as training.

- Mean P99 latency: 23.98 +/- 0.57 ms
- Mean 30 ms SLA violation rate: 5.6% (94.4% compliance)
- Mean gNB power: 105.89 +/- 2.18 W
- Mean UE power: 39.52 +/- 0.12 mW per UE
- Mean DL RLC backlog: 30,085.9 bytes
- Mean Active Window: 29.28 ms
- Wide-BWP selection rate: 0.0%

The latency result is stable across seeds, but the zero wide-BWP selection rate
shows that this checkpoint learned an effectively time-only policy. It should
therefore be compared against the 1D rule-based controller before claiming a
benefit from joint 2D control.

Important metric correction: `AvgP99LatencyMs` in the Python evaluation CSV is
the mean of 125 per-window P99 values. It is not the P99 of all packets in an
episode. The packet-level episode P99 must be read from the ns-3 final-statistics
CSV (`eval_10seed_125step_ns3.csv`).

Artifacts:

- `eval_10seed_125step.csv`
- `eval_10seed_125step_ns3.csv`
- `eval_10seed_125step.png`
- `0005-add-deterministic-evaluation.patch`
- `0006-add-evaluation-plot.patch`

## Fair 1D rule-based versus DRL comparison

The 1D baseline was rerun on matched seeds 101-110 with the same 5.2-second
effective duration, 10 UEs, 2 Mbps per UE, 30 dBm gNB Tx power, and a fixed
20 MHz BWP. A fixed-BWP bug in the UE energy model was corrected before the
comparison; it had previously charged 100 MHz UE power whenever dynamic BWP was
disabled.

| Metric | 1D Rule-Based | DRL checkpoint | DRL change |
|---|---:|---:|---:|
| Packet-level episode P99 | 52.11 +/- 1.15 ms | 52.84 +/- 0.95 ms | +1.40% |
| 30 ms packet compliance | 81.56 +/- 0.42% | 94.86 +/- 0.05% | +16.31% |
| gNB power | 99.11 +/- 1.34 W | 104.40 +/- 2.10 W | +5.33% |
| UE power | 34.16 +/- 0.18 mW | 38.81 +/- 0.13 mW | +13.64% |
| Energy efficiency | 0.2051 +/- 0.0028 Mbit/J | 0.1952 +/- 0.0039 Mbit/J | -4.83% |
| Delivered traffic | 105.67 +/- 0.05 Mbit | 105.91 +/- 0.06 Mbit | +0.23% |

Every one of the 10 matched seeds showed the same direction for all six
differences. The DRL checkpoint trades energy for substantially better packet
compliance, but it does not improve the extreme packet-level P99 and it never
selects the wide BWP. It should be described as an effectively 1D learned
policy, not as demonstrated 2D joint-control gain.

Comparison artifacts:

- `rule_1d_10seed_ns3.csv`
- `rule_1d_10seed_timeline.csv`
- `fair_rule_vs_drl_summary.csv`
- `fair_rule_vs_drl.png`

## Observation V2 (requires retraining)

The next controller version replaces the hard-coded CQI value with statistics
read from the NR scheduler. Its input remains fixed-size when the UE count
changes:

1. total DL RLC backlog
2. maximum per-UE DL RLC backlog
3. window P99 delay
4. mean wideband CQI
5. P10 wideband CQI
6. minimum wideband CQI
7. mean DL MCS
8. Active Window
9. BWP state

The scheduler only includes CQI reports with a live CQI timer, so the channel
features represent actual CSI feedback instead of the initial fallback value.
The IPC message version is now `S2`; Python rejects an old message instead of
silently shifting fields. UDP clients also start at the controller warm-up
boundary, preventing pre-control packets from contaminating the delay metric.

The prior 5D checkpoints in this archive are preserved as baselines but cannot
be loaded into the 9D network. A fresh training run is required.

Validation performed on 2026-10-18:

- full `./ns3 build`: passed (108/108 targets)
- Python syntax checks: passed
- real ns-3 IPC smoke run: 1 episode x 5 steps, passed
- smoke mean CQI / P10 CQI / minimum CQI: 11.56 / 9.60 / 9.60
- smoke mean DL MCS: 21.52

The smoke outputs were written under `/tmp/ppo_v2_smoke_*` and are not promoted
as experiment results.
