# First clean scheduler-controlled version

Date: 2026-10-18

This archive records the first version in which the application traffic is
independent of the DRL action, the Cell-DTX active window controls downlink MAC
scheduling, the 20/100 MHz action changes usable scheduler resources, and P99
uses measured packet delay without an added synthetic C-DRX penalty.

## Git bases

- ns-3 root (`main`): `d696d664076708008478af7c18a43c523e6efa4a`
- `contrib/nr` (`master`): `9b830f9adfc28bb559f4eaa5fc22fd4518e1cdfa`

`scratch/joint-energy-mvp.cc` already contained uncommitted IPC, lifecycle, and
10-UE experiment work before this change. Patch 0001 is therefore a complete
snapshot of that scenario relative to the root commit above, not only the last
Cell-DTX edits. This is intentional so that the experiment can be reconstructed
from a clean checkout.

## Patches

1. `0001-joint-energy-first-version.patch`
   - SHA-256: `45f0f81b511264e2c9cf0f7cd66938904f6a4fe3b34ebe36de27adcb22237cd0`
   - Applies to the ns-3 root repository.
2. `0002-nr-scheduler-dl-gating.patch`
   - SHA-256: `5dff3c0183e9ffd5dd480299d6203c75e398162281c13c2a255a62eb6970bd7a`
   - Applies to the nested `contrib/nr` repository.

Apply from the ns-3 root:

```bash
git apply --check experiment_archive/first_version_validation_2026-10-18/0001-joint-energy-first-version.patch
git apply experiment_archive/first_version_validation_2026-10-18/0001-joint-energy-first-version.patch

git -C contrib/nr apply --check ../../experiment_archive/first_version_validation_2026-10-18/0002-nr-scheduler-dl-gating.patch
git -C contrib/nr apply ../../experiment_archive/first_version_validation_2026-10-18/0002-nr-scheduler-dl-gating.patch

./ns3 build
```

## Behavior implemented

- Every UE continuously offers the configured UDP rate. Active Window no longer
  changes `OnTime` or `OffTime`.
- Inactive Cell-DTX time disables new DL data scheduling. Control signaling and
  required HARQ retransmissions remain enabled so the NR connection is not
  destroyed.
- Narrow mode exposes the contiguous scheduler RBG set corresponding to 51 PRBs.
- Wide mode exposes the full 100 MHz carrier, up to the configured 273 PRBs.
- The default numerology is 1 (30 kHz SCS), matching the 51/273 PRB profiles.
- The delay tracker records only `rxTime - txTime`; no synthetic wake-up delay is
  added.
- Pending bytes are tracked per packet at a consistent UDP-payload size instead
  of subtracting counters from different protocol layers.
- This is an effective scheduler-bandwidth implementation, not yet two separate
  PHY BWPs with RRC BWP switching.

## Validation

Build completed successfully. The following tests passed:

```text
PASS: nr-test-notching
PASS: nr-system-test-scheduler-many-ues
```

All three final 1-second fixed-policy runs generated exactly `2,526,480` offered
bytes, which confirms that Active Window and BWP no longer alter source traffic.

| Case | App RX bytes | Pending bytes | P99 ms |
|---|---:|---:|---:|
| 10 ms, narrow | 1,374,208 | 1,153,620 | 211.670 |
| 30 ms, narrow | 2,107,392 | 598,212 | 88.374 |
| 30 ms, wide | 2,285,568 | 467,712 | 128.775 |

These short runs are smoke/capacity tests, not final performance results. Their
P99 values include startup transients and an undrained final queue. A new model
must be trained before drawing Rule-Based versus DRL conclusions.

The Rule-Based dynamic run created 24 decision rows plus the header and reached
30 ms / 100 MHz. A final 5-step deterministic evaluation also completed over real
socket IPC on port 5555. Its score is not meaningful because it used the old
model only to verify protocol compatibility.

## Files

- `gating10_narrow_mu1.csv`
- `gating30_narrow_mu1.csv`
- `gating30_wide_mu1.csv`
- `rule_dynamic_v2.csv`
- `rule_dynamic_timeline_v2.csv`
- `ipc_eval_summary_mu1.csv`
- `ipc_final_stats_mu1.csv`

