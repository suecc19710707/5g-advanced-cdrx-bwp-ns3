# Slot-start eligibility enforcement

This patch contains the production `contrib/nr` changes for slot-start
Cell-DTX and per-UE C-DRX eligibility enforcement.

It intentionally does **not** implement full-slot or two-slot look-ahead
eligibility. The scheduler evaluates the start time of the target slot; the
existing PHY listener remains as a consistency assertion for a TB that still
arrives outside UE Active Time.

## Apply

From the ns-3 repository root:

```bash
cd contrib/nr
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0001-nr-slot-start-eligibility.patch
```

The patch adds the scheduler Cell-DTX gate, the per-UE DL eligibility callback,
the shared `NrUeDrxModel::IsActiveAt()` query, and the PHY consistency check.

The scenario-side callback wiring must call
`NrMacSchedulerNs3::SetUeDlEligibilityCallback()` with the target slot start
time. Full-slot eligibility remains a separate follow-up decision.
