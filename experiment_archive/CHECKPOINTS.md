# Experiment checkpoints

## `custom-energy-v4-checkpoint-2026-10-19`

This checkpoint is the final version before replacing the custom gNB energy
accounting with the official 5G-LENA energy model.

Included runtime sources:

- `scratch/joint-energy-mvp.cc`
- `ns3_environment.py`
- `train_drl_agent.py`
- `eval_drl_agent.py`
- `plot_training_results.py`
- `plot_evaluation_results.py`

The local `contrib/nr` checkout is a separate upstream Git repository and is
not tracked by the parent ns-3 repository. Its scheduler modifications are
preserved as:

- `gnb_energy_v4_2026-10-18/0002-nr-scheduler-observation.patch`

The archived experiment directories contain the corresponding patches, CSV
results, plots, and small model artifacts needed to reproduce the progression
from the first validation and rule-based comparisons through observation-v3
and custom-energy-v4.

Future official-energy work starts after this checkpoint and must not rewrite
the checkpoint tag.

## `official-energy-v5` development branch

Development after the custom-energy checkpoint replaces the custom slot
trackers with the candidate 5G-LENA TR 38.864/TR 38.840 framework from official
GitLab merge requests !408 and !417.  Provenance, assumptions, validation, and
the complete nested `contrib/nr` patch series are stored in:

- `official_energy_v5_2026-10-19/`

The v5 branch also records an important semantic correction: the current
frequency action is a schedulable-PRB profile on one physical 100 MHz BWP, not
a physical 20/100 MHz BWP switch.  Official UE energy follows the actual PHY
bandwidth; the former manual 20 MHz energy scaling override was removed.

## `true-bwp-dci-v6` phase 1

The first v6 checkpoint validates fixed physical 20 MHz and 100 MHz gNB/UE
PHY BWPs before dynamic switching is connected to the DRL agent.  The scenario,
RRC extension patch, fixed-mode measurements, and remaining simulator
limitations are documented in:

- `true_bwp_dci_v6_2026-10-19/`

## `true-bwp-dci-v6` phase 2

The deterministic `20 -> 100 -> 20 MHz` test now switches the UE RRC primary
BWP, gNB per-UE routing, DCI BWP ID, and official UE Energy bandwidth scaling
as one controlled operation. One official UE model observes all BWP PHYs
without double-counting the UE. DCI interval counts, traffic continuity,
energy timing, regression results, and the second nested 5G-LENA patch are
recorded in the same directory. Policy/IPC integration remains the next phase.
