# Official Energy v5 integration

This directory records the first project version that replaces the custom
throughput-based energy accounting with the 5G-LENA NR energy framework.

## Upstream provenance

The framework is under review in the official CTTC 5G-LENA GitLab project.  It
is not part of the v5.1 release yet, so this project pins the reviewed candidate
commits rather than describing it as a released API.

- Energy models/listeners, MR !408, source head `d61be032131c24463efcff35298d17152d9e9264`
- `NrEnergyHelper`, MR !417, source head `470cf7cc48c80fb86799cf97a287df78f8a6cb50`
- Local 5G-LENA base: `9b830f9adfc28bb559f4eaa5fc22fd4518e1cdfa`
- Integrated local 5G-LENA head: `0563f802`

The numbered patch files in this directory reproduce the eight upstream Energy
commits on the local v5.1-27 base, followed by the project's scheduler
observation extension.

## Scenario integration

`scratch/joint-energy-mvp.cc` now installs:

- `BasicEnergySource` per gNB and UE;
- `NrGnbEnergyModel` and `NrGnbPhyEnergyListener` (TR 38.864);
- `NrUeEnergyModel`, `NrUePhyEnergyListener`, and optional `NrUeDrxModel`
  (TR 38.840);
- `OfficialEnergyMetrics`, a read-only adapter that derives control-window
  averages from the official cumulative model energy.

The old one-millisecond custom tracker is excluded from compilation and no
custom energy event is scheduled.  Consequently the official gNB listener uses
the PHY's real symbol duration (for numerology 1: 0.5 ms slots / 14 symbols)
without creating a second simulation clock.

Default configuration:

- gNB: `BsCat1`, `Set1`, dual PA efficiency, antenna ratio `A=0.4`;
- gNB reference RB count: the actual 100 MHz PHY RB count;
- UE: FR1, 100 MHz reference BWP;
- gNB `PowerUnit=1 W`, UE `PowerUnit=1 mW`;
- C-DRX: custom 40 ms long cycle with the scenario's OnDuration and inactivity
  timer.

`PowerUnit` is intentionally explicit in the C++ and Python CLIs.  TR 38.864
and TR 38.840 provide relative power tables, not a universally calibrated
absolute watt value, so absolute-energy claims must state these assumptions or
replace them with measured calibration values.

## Validation performed

- Full `./ns3 build`: passed.
- `nr-energy-models`: passed.
- `nr-test-gnb-phy-energy-listener`: passed.
- `nr-test-ue-phy-energy-listener`: passed.
- `nr-energy-integration`: passed (model total, analytical reconstruction, and
  `BasicEnergySource` drain agree in the upstream test).
- 3-UE, 0.8 s rule-based scenario smoke test: passed; gNB and UE energies were
  non-zero and the exported CSV had 21 matching columns.
- Real IPC PPO smoke test (`1 episode x 2 steps`): passed without Mock fallback
  or port-5555 timeout.  Observed window averages were 71.47 W gNB and
  91.47 mW per UE for that deterministic short run.

## Current limitation

`NrEnergyHelper` starts every UE DRX model immediately and does not expose a
per-UE start offset.  Official v5 therefore rejects non-zero `phaseOffsetMs`
when C-DRX is enabled instead of silently reporting the wrong phase behavior.
A later version can manually wire the official UE components or extend the
helper after upstream settles the API.
