# True physical BWP v6 — fixed and scripted-switch checkpoints

This directory records both the fixed-mode validation and the deterministic
`20 -> 100 -> 20 MHz` runtime-switch validation used before connecting the
policy/DRL controller.  Policy-driven switching is deliberately still
disabled.

## Provenance

- Parent ns-3 checkpoint: `44d7e76599f43f0b9e948343a808291004748be0`
  (`official-energy-v5`).
- Nested 5G-LENA commits:
  - `fb0ca0378fd5d547ae06d12bc32135470e911d1c` (controlled RRC switch);
  - `b7b3ae7b90792a1ad6484c6e10fb521aeeb8d96f` (one UE Energy model
    observing every runtime BWP).
- Reproduction patch series:
  - `0001-nr-expose-controlled-same-cell-BWP-switching.patch`;
  - `0002-nr-account-UE-energy-across-runtime-BWP-switches.patch`.

## Physical configuration

With `--enableTrueBwp=true`, the scenario creates two PHYs on both the gNB and
each UE:

- BWP 0: 20 MHz, centre 3.46 GHz, range 3.45–3.47 GHz;
- BWP 1: 100 MHz, centre 3.50 GHz, range 3.45–3.55 GHz.

The distinct centre ARFCNs let the current 5G-LENA RRC map and configure both
PHYs. The 20 MHz BWP is nested in the wider carrier range. The stock
`CcBwpCreator` only creates non-overlapping partitions, so the nested pair is
described explicitly by the scenario.

## RRC extension

The nested patch adds:

- `NrUeRrc::RequestPrimaryBwpSwitch()`, which uses the existing guarded
  same-cell switch and informs the gNB per-UE BWP routing;
- serving `(cellId, ARFCN)` bookkeeping during `ReconfigureFromSib1()`;
- `EnableAutomaticBwpSwitch`, allowing the local RSRP policy to be disabled
  when an external controller owns BWP selection.

For controlled v6 experiments the automatic RSRP policy is disabled. Without
this, it changed BWP 1 back to BWP 0 at traffic start and invalidated the
experiment.

## Energy accounting

In fixed mode, the official 5G-LENA UE Energy listener is attached to the
selected physical BWP. A trace confirmed:

- fixed 20 MHz: `ApplyBwpScaling(20)` on the first DL TB;
- fixed 100 MHz: `ApplyBwpScaling(100)` on the first DL TB, followed by
  `DlTbReceivedCallback` for subsequent transport blocks.

For v6 true-BWP runs, cumulative energy baselines are reset at the common
traffic-start time. Attachment and the initial BWP-1 switch therefore do not
bias the fixed-mode comparison.

For scripted runtime switching, `NrEnergyHelper::InstallUeAllBwps()` creates
exactly one `NrUeEnergyModel` per UE and one listener subscribed to every BWP
PHY. Each callback is bound to its source PHY, so its bandwidth, cached slot
duration, and UL transmit power come from the BWP that carried the TB. All
BWPs share one `activeUntil` deadline, preventing a stale timer from an old
BWP from ending a new BWP's active interval. The switch event also calls the
official `ApplyBwpScaling()` API immediately, so monitoring power changes at
the switch rather than waiting for the first TB.

## Fixed-mode validation

Command shape:

```bash
./ns3 run "scratch/joint-energy-mvp --enableTrueBwp=true \
  --fixedBwpMHz=20 --simTime=0.8 --numUes=3 --udpDataRate=2 \
  --enableGating=false --enableCdrx=false --enableBwpDynamic=false"
```

Repeat with `--fixedBwpMHz=100`.

| Physical BWP | Rx (Mbit) | P99 (ms) | QoS | gNB power (W) | UE power (mW/UE) |
|---|---:|---:|---:|---:|---:|
| 20 MHz | 3.62544 | 11.9531 | 100% | 73.0157 | 58.4452 |
| 100 MHz | 3.62544 | 11.7389 | 100% | 68.3324 | 121.901 |

At equal offered load, the narrow BWP lowers UE active power but needs more
active OFDM resources; the wide BWP finishes the same traffic with fewer
active symbols, so the official gNB model reports lower average power here.

## Scripted runtime-switch validation

Command shape:

```bash
./ns3 run "scratch/joint-energy-mvp --enableTrueBwp=true \
  --enableTrueBwpSwitchScript=true --enableBwpValidationTraces=true \
  --simTime=2.0 --numUes=3 --udpDataRate=2 \
  --enableGating=false --enableCdrx=false \
  --bwpSwitchToWideSec=0.8 --bwpSwitchBackToNarrowSec=1.4"
```

The built-in UE PHY DCI trace reported:

| Time region | Expected BWP | DL DCI count |
|---|---:|---:|
| before 0.8 s | 0 (20 MHz) | 447 |
| 0.8–1.4 s | 1 (100 MHz) | 450 |
| after 1.4 s | 0 (20 MHz) | 438 |

Three additional BWP-0 DCI records arrived at 0.801036 s, one per UE. They
were already in the k-delay pipeline before the 0.8 s switch; no later BWP-0
DL DCI occurred in the wide interval. The return to BWP 0 had no stale BWP-1
DL DCI.

The official UE energy trace applied 100 MHz to all three UE models at exactly
0.800000 s and 20 MHz at exactly 1.400000 s. End-to-end results were 11.0231
Mbit received, 11.9451 ms P99, 100% QoS, 71.4095 W average gNB power, and
81.0778 mW average UE power per UE. Offered/application counters remained
continuous across both switches (`1374948` offered bytes, `1339392` received,
`9396` pending).

Full build, `nr-energy-integration`, and `nr-energy-models` passed after the
change.

## Remaining work and limitation

- Deterministic runtime switching is enabled; rule-based and Python/DRL action
  integration are not enabled yet.
- `NrChannelHelper` creates one `SpectrumChannel` object per BWP. The nested
  spectral ranges are therefore a simulator-level BWP abstraction rather than
  two simultaneously interacting overlapping RF transmissions. Only the
  selected BWP may carry user data in the controlled experiment.
- The runtime validation changes primary BWP through the existing ideal RRC
  indication and 5G-LENA `DciInfoElementTdma` BWP ID. It does not bit-encode a
  standards DCI message over a modeled control channel.
- The next acceptance stage is to route rule-based and IPC/DRL actions through
  the validated switch event, then run the offered-load sweep.
