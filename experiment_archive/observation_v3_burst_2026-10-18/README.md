# Observation V3 and burst-traffic calibration

Date: 2026-10-18

This version uses a fixed-size 10D observation:

1. total DL RLC backlog
2. maximum per-UE DL RLC backlog
3. offered traffic rate over the latest 40 ms window
4. window P99 delay
5. mean wideband CQI
6. P10 wideband CQI
7. minimum wideband CQI
8. mean DL MCS
9. Active Window
10. BWP state

The IPC protocol identifier is `S3`. Old 5D and 9D checkpoints are not
compatible and remain archived only as baselines.

## Traffic model

The original constant source remains the default. Burst traffic is enabled
explicitly with `--enable-burst-traffic` in Python or
`--enableBurstTraffic=true` in C++.

Every UE has two independent downlink UDP sources:

- constant base source: 1 Mbps
- additional burst source: 8 Mbps while On
- exponential mean On duration: 80 ms
- exponential mean Off duration: 120 ms
- expected duty cycle: 40%
- expected mean per UE: 4.2 Mbps
- per-UE peak: 9 Mbps

The ten burst sources use ns-3 RNG streams and are not synchronized. UDP and
the controller start together after the 200 ms channel/attachment warm-up.

## Matched-seed fixed-BWP calibration

Seed 42 / run 1, 10 UEs, 30 ms Active Window, 30 ms SLA:

| Load | BWP | Packet P99 | SLA compliance | Delivered | gNB power | UE power |
|---|---:|---:|---:|---:|---:|---:|
| base 1 + burst 3 Mbps/UE | 20 MHz | 23.25 ms | 100.00% | 113.70 Mbit | 137.73 W | 38.01 mW |
| base 1 + burst 3 Mbps/UE | 100 MHz | 21.58 ms | 100.00% | 113.70 Mbit | 129.00 W | 76.58 mW |
| base 1 + burst 8 Mbps/UE | 20 MHz | 31.42 ms | 97.76% | 122.81 Mbit | 179.75 W | 50.19 mW |
| base 1 + burst 8 Mbps/UE | 100 MHz | 21.62 ms | 100.00% | 129.11 Mbit | 168.89 W | 107.68 mW |

The 3 Mbps burst is too light because both BWPs satisfy the SLA. The 8 Mbps
burst provides a useful 2D control trade-off: 100 MHz improves delay and
delivered traffic, but roughly doubles UE power.

The gNB power model is still traffic-driven and not parameterized directly by
BWP state. No arbitrary BWP power constants were added; that model requires a
separate, documented calibration.

## Intended short training command

```bash
python3 train_drl_agent.py \
  --num-episodes 5 --max-steps 50 \
  --num-ues 10 --udp-rate 1 \
  --enable-burst-traffic --udp-burst-rate 8 \
  --burst-mean-on-ms 80 --burst-mean-off-ms 120 \
  --gnb-tx-power-dbm 30 --qos-target-ms 30 \
  --warmup-ms 200 --max-active-ms 30 --seed 42 \
  --history-csv experiment_archive/observation_v3_burst_2026-10-18/pilot_5x50_history.csv \
  --ns3-csv experiment_archive/observation_v3_burst_2026-10-18/pilot_5x50_ns3.csv \
  --model-out experiment_archive/observation_v3_burst_2026-10-18/pilot_5x50_model.pt \
  --torchscript-out experiment_archive/observation_v3_burst_2026-10-18/pilot_5x50_torchscript.pt
```

## Synchronized cell-level burst extension

The optional `--synchronize-burst-traffic` Python flag (or
`--synchronizeBurstTraffic=true` in C++) gives all ten UEs the same constant
On/Off burst phase. The independent exponential model remains the default.
With base 1 Mbps/UE, burst 8 Mbps/UE, 80 ms On and 120 ms Off, the S3
observation measured 8.352--91.872 Mbps over 40 ms windows (standard deviation
39.943 Mbps). This avoids the aggregation smoothing seen with ten independent
sources.

Matched seed 42/run 1, 10 UEs, 30 ms SLA, 3.2 s simulation:

| Controller | Packet P99 | SLA compliance | Delivered | gNB power | UE power | Wide BWP |
|---|---:|---:|---:|---:|---:|---:|
| fixed 20 MHz, Active 30 ms | 36.49 ms | 82.72% | 90.73 Mbit | 127.00 W | 43.49 mW | 0% |
| fixed 100 MHz, Active 30 ms | 21.67 ms | 100.00% | 121.30 Mbit | 116.16 W | 91.26 mW | 100% |
| original 2D rule, 2 KB threshold | 31.06 ms | 98.21% | 102.04 Mbit | 94.82 W | 80.74 mW | 98.67% |
| 2D rule, 30 KB threshold | 37.20 ms | 86.62% | 87.95 Mbit | 82.29 W | 60.95 mW | 81.33% |
| BWP-only rule, Active 30 ms, 30 KB threshold | 29.13 ms | 99.36% | 114.68 Mbit | 117.76 W | 78.78 mW | 57.33% |

The BWP-only ablation proves that dynamic frequency control is effective: it
reduces UE power by 13.7% relative to fixed 100 MHz while retaining 99.36% SLA
compliance. The original 2 KB rule threshold is below the roughly 21 KB
low-load aggregate RLC floor for ten UEs, so it stays wide almost all the time.
The full 30 KB rule also shortens the Active Window too aggressively; that is
why its latency is worse than the BWP-only ablation.

## Synchronized PPO pilot

The `sync_safe_init_15x50_*` files contain a 15-episode, 50-step PPO pilot and
a three-seed deterministic evaluation. The deterministic policy achieved
19.20 +/- 0.01 ms mean window-P99 and zero 30 ms window violations, but used a
30 ms Active Window and 100 MHz BWP for every low- and high-load step. It is a
QoS-safe maximum-resource policy, not a converged dynamic-energy policy.

The pilot has only 750 transitions. In addition, P99 is currently normalized
by 500 ms, mapping the relevant 20--40 ms range into only 0.04--0.08. Further
PPO work should first use an SLA-relevant delay scale and then run a longer,
multi-seed training campaign. Training loss alone must not be used as evidence
of convergence.

