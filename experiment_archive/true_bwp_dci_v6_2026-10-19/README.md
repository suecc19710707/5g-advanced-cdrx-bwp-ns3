# True physical BWP v6 — phase 1 checkpoint

This checkpoint is the fixed-mode validation stage for replacing the v5
scheduler PRB profile with distinct physical gNB/UE PHY bandwidth parts.
Dynamic DRL switching is deliberately still disabled.

## Provenance

- Parent ns-3 checkpoint: `44d7e76599f43f0b9e948343a808291004748be0`
  (`official-energy-v5`).
- Nested 5G-LENA commit: `fb0ca0378fd5d547ae06d12bc32135470e911d1c`.
- Reproduction patch:
  `0001-nr-expose-controlled-same-cell-BWP-switching.patch`.

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

The official 5G-LENA UE Energy listener is attached to the selected physical
BWP. A trace confirmed:

- fixed 20 MHz: `ApplyBwpScaling(20)` on the first DL TB;
- fixed 100 MHz: `ApplyBwpScaling(100)` on the first DL TB, followed by
  `DlTbReceivedCallback` for subsequent transport blocks.

For v6 true-BWP runs, cumulative energy baselines are reset at the common
traffic-start time. Attachment and the initial BWP-1 switch therefore do not
bias the fixed-mode comparison.

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

## Remaining work and limitation

- Dynamic switching and Python/DRL integration are not enabled yet.
- `NrChannelHelper` creates one `SpectrumChannel` object per BWP. The nested
  spectral ranges are therefore a simulator-level BWP abstraction rather than
  two simultaneously interacting overlapping RF transmissions. Only the
  selected BWP may carry user data in the controlled experiment.
- The official UE helper attaches one energy listener/model to the selected
  BWP. Phase 2 must make that binding follow every runtime switch without
  double-counting inactive BWPs.
- Phase 2 acceptance requires DCI traces with changing BWP IDs, continuous
  per-UE traffic, synchronized gNB/UE primary BWP state, and official energy
  callbacks on the newly active BWP.
