# gNB energy reporting and BWP-aware model v4

Date: 2026-10-18

## Scope

This version keeps the DRL observation at 10 dimensions but advances the IPC
protocol from `S3` to `S4`. The additional delivered-rate field is reporting
metadata, not a new observation dimension.

Training and deterministic evaluation CSVs now report:

- average delivered rate (Mbps)
- delivered traffic during controlled windows (Mbit)
- gNB energy during controlled windows (J)
- gNB energy efficiency (Mbit/J)
- normalized gNB energy (kWh/TB)
- aggregate UE energy during controlled windows (J)

The Python values cover only the 40 ms decision windows. The final C++ CSV
continues to report whole-process energy, including warm-up and shutdown, so
the two energy columns intentionally have different scopes.

## BWP-aware gNB energy tracker

`GnbSlotEnergyTracker` now stores the current BWP and is updated whenever the
rule-based or IPC controller changes BWP. The following C++/Python parameters
are exposed instead of hiding assumptions in constants:

- active baseline power
- micro-sleep power
- RF transmit-power term and load slope
- narrow/wide reference capacity
- additional wide-BWP active and sleep power

Both reference capacities default to the legacy 100 Mbps and both wide-BWP
overheads default to zero. These defaults reproduce the previous tracker and
must not be interpreted as a calibrated 20/100 MHz hardware-power model.
Measured or cited coefficients can be supplied later without another code
change.

## Validation

- Python syntax and `git diff --check`: passed
- full `./ns3 build`: passed
- real S4 IPC training smoke test: passed
- real S4 deterministic evaluation smoke test: passed

The evaluation smoke test reported 96.64 J over 20 controlled windows,
0.3136 Mbit/J, and 7.09 kWh/TB.

Matched seed 42/run 1, 10 UEs, synchronized base 1 + burst 8 Mbps/UE,
30 ms Active Window, zero wide-BWP power overhead:

| BWP | Delivered | gNB power | gNB energy | Mbit/J | kWh/TB | P99 | SLA compliance | UE power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 MHz | 90.73 Mbit | 127.00 W | 406.29 J | 0.2233 | 9.95 | 36.49 ms | 82.72% | 43.49 mW |
| 100 MHz | 121.30 Mbit | 116.16 W | 371.58 J | 0.3264 | 6.81 | 21.67 ms | 100.00% | 91.26 mW |

The lower gNB energy at 100 MHz is currently caused by delivered-throughput
and sleep-slot effects. It is not evidence that a real 100 MHz radio consumes
less hardware power. A rejected diagnostic run used 20 Mbps as the narrow
reference capacity; that assumption artificially saturated the load term and
is retained only as `fixed20_overhead0.csv`, not as a baseline. The official
backward-compatible 20 MHz result is `fixed20_reference100_overhead0.csv`.

## Next step

Calibrate or sweep the wide-BWP active/sleep overheads, then change the P99
observation scale from the legacy 500 ms to an SLA-relevant scale before the
next PPO training campaign.
