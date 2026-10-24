# S5 PPO multi-seed training summary

Configuration: 2 UEs, 0.5 Mbps/UE, QoS target 50 ms, 5.2 s episode, Custom energy reference 30 dBm and antenna ratio 0.4.

| Training seed | Best deterministic gNB power (W) | QoS | p99 delay (ms) | Active window (ms) |
|---:|---:|---:|---:|---:|
| 42 | 41.26584 | 100% | 33.36072 | 10.256 |
| 101 | 46.84400 | 100% | 33.29440 | 14.560 |
| 202 | 43.41016 | 100% | 33.37272 | 12.000 |
| 303 | 51.02008 | 100% | 33.37272 | 18.064 |

The environment-seed evaluation of the seed-42 checkpoint (seeds 101, 202, 303, 404, 505) varied by less than 0.001 W. The larger spread above is therefore training-seed variance, not a special channel/environment seed effect.
