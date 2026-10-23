# Slot-start eligibility enforcement

This directory contains the production `contrib/nr` changes for Cell-DTX and
per-UE C-DRX eligibility enforcement.

`0001` implements slot-start eligibility. `0002` upgrades DL data grants to
grant-span-aware eligibility using the actual DCI symbol interval. Together,
they intentionally do **not** implement a fixed full-slot or two-slot guard.
The existing PHY listener remains as a consistency assertion for a TB that
still arrives outside UE Active Time.

## Apply

From the ns-3 repository root:

```bash
cd contrib/nr
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0001-nr-slot-start-eligibility.patch
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0002-nr-grant-span-eligibility.patch
cd ../..
git apply experiment_archive/slot_start_eligibility_2026-10-23/0003-joint-energy-scenario-grant-span.patch
```

The patches add the scheduler Cell-DTX gate, the per-UE DL eligibility
callback, the shared `NrUeDrxModel` time-interval query, and the PHY
consistency check. New DL DCIs are rejected before RLC bytes are assigned when
their exact grant span crosses the UE's C-DRX boundary.

`0003` updates `scratch/joint-energy-mvp.cc` so the scenario passes the actual
grant start/end interval to `SetUeDlEligibilityCallback()`.
