# Full 2D Rule-Based vs 2D DRL comparison

Configuration: 10 UEs, nominal 2 Mbps UDP per UE, 20 ms SLA, C-DRX/DCI and
dynamic BWP enabled. The DRL model is the archived `GAMMA=0.5` baseline.

This run exposed a workload-coupling problem. `W_active` is applied directly to
the UDP OnOff applications, so changing the controller action also changes the
offered traffic. The Rule-Based run ended at a 30 ms active window and received
129.569 Mbit, while DRL stayed at 10 ms and received only 50.233 Mbit. The two
rows therefore do not represent equal delivered workloads and must not be used
to claim that one controller is better.

The packet-level final statistics are in `rule_based_default.csv` and
`drl_final_stats.csv`. `drl_eval_summary.csv` contains the step-averaged
evaluation statistics.

The six-panel visual comparison is in `rule_vs_drl_comparison.png`.

Before controller ablations or latency-weight tuning, decouple UDP traffic
generation from the Cell-DTX active-window control, then repeat this comparison.
