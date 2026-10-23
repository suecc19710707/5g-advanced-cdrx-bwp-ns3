# 5G-Advanced Joint Cell-DTX, UE C-DRX & Dynamic 2D BWP Switching Framework

This repository contains an advanced **5G-Advanced discrete-event network simulation framework** built on top of **ns-3 (5G-LENA `nr` module)**. It implements a **Joint Time-Frequency 2D Alignment Architecture**, coordinating Base Station **Cell-DTX**, User Equipment **C-DRX**, and DCI-based **Dynamic Bandwidth Part (BWP) Switching (20 MHz ↔ 100 MHz)**.

> The reproducible setup below is the current grant-span eligibility
> environment. The older result tables and the historical 9-experiment
> commands further down are retained for reference.

## Rebuild the current environment from GitHub

The outer repository intentionally does not vendor `contrib/nr`; that module
is a separate Git repository. From a clean machine:

```bash
git clone --branch true-bwp-dci-v6 \
  https://github.com/suecc0707/5g-advanced-cdrx-bwp-ns3.git ns-3-dev
cd ns-3-dev

# Fetch the ns-3 NR module used by this experiment.
git clone https://gitlab.com/cttc-lena/nr.git contrib/nr
cd contrib/nr
git checkout 9b830f9adfc28bb559f4eaa5fc22fd4518e1cdfa
cd ../..

# Reconstruct the energy/BWP history, then apply grant-span eligibility.
cd contrib/nr
git am ../../experiment_archive/official_energy_v5_2026-10-19/000*.patch
git am ../../experiment_archive/true_bwp_dci_v6_2026-10-19/000*.patch
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0001-*.patch
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0002-*.patch
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0004-*.patch
cd ../..
git apply experiment_archive/slot_start_eligibility_2026-10-23/0003-*.patch

# Configure and build ns-3 (CMake-based ns-3.48 environment).
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

The final patches implement **grant-span-aware** Cell-DTX and per-UE C-DRX
eligibility. The scheduler checks the actual DCI symbol interval before
assigning RLC bytes, clips a partially overlapping grant when possible, and
defers only when the clipped TB is below the minimum size. It deliberately
does not implement a fixed two-slot guard.
Patch `0003` installs the matching scenario-side callback wiring.

To run the main scenario after building:

```bash
./ns3 run joint-energy-mvp
```

The exact scenario arguments and experiment matrices are kept in the relevant
`experiment_archive/` run scripts.

## Scenario parameters

`joint-energy-mvp` uses ns-3 `CommandLine` parameters. Values in the source
are defaults; a reproducible experiment should pass the important values
explicitly. The following are the parameters used by the current rule-based
baseline/grid experiments.

| Parameter | Meaning | Current baseline value |
|---|---|---:|
| `numUes` | Number of UEs | `10` |
| `udpDataRate` | Constant DL UDP rate per UE (Mbps) | `2` |
| `simTime` | Simulation duration (s) | `5` or `10` |
| `trafficStopSec` | Stop traffic before simulation end; `-1` disables | `-1` |
| `cellDtxPeriodMs` | Cell-DTX cycle length | `40` ms |
| `cellDtxActiveMs` | Initial Cell-DTX active window | `20` ms for the current baseline |
| `cdrxOnDurationMs` | UE C-DRX on-duration | `10` ms |
| `cdrxInactivityMs` | Common UE C-DRX inactivity timer | `10` ms |
| `ueCdrxInactivityMs` | Comma-separated per-UE inactivity timers; overrides common value | empty |
| `phaseOffsetMs` | Common UE/Cell C-DRX phase offset | `0` ms |
| `ueCdrxOffsetsMs` | Comma-separated per-UE offsets; overrides common phase | empty |
| `qosTargetDelayMs` | QoS delay target used by controller/statistics | `50` ms in the current baseline |
| `enableGating` | Enable Cell-DTX scheduler gating | `true` |
| `enableCdrx` | Enable UE C-DRX energy/eligibility model | `true` |
| `enableQueueAware` | Enable closed-loop rule controller | `true` for rule baselines |
| `controllerEvalIntervalMs` | Controller decision interval | `40` ms |
| `controllerWarmupMs` | Delay before first controller decision | `200` ms |
| `controllerMinActiveMs` | Lower active-window bound | `10` ms |
| `controllerMaxActiveMs` | Upper active-window bound | `30` ms |
| `controllerInitialActiveMs` | Rule controller initial active window | `20` ms |
| `controllerStepUpMs` | Active-window increase per decision | `2` ms |
| `controllerStepDownMs` | Active-window decrease per decision | `2` ms |
| `controllerQueueWarnRatio` | Queue warning occupancy ratio | `0.50` |
| `controllerQueueHighRatio` | Queue high occupancy ratio | `0.60` |
| `controllerBwpDwellMs` | Minimum time between rule BWP switches | `100` ms |
| `enableBwpDynamic` | Enable policy/rule BWP selection | `false` for 1D, `true` for 2D |
| `enableTrueBwp` | Use physical 20/100 MHz BWPs | `false` for legacy/1D, `true` for physical 2D |
| `fixedBwpMHz` | Fixed BWP profile when dynamic BWP is off | `20` or `100` |
| `gnbTxPowerDbm` | gNB transmit power | `30` dBm |
| `gnbEnergyRefConfig` | Energy reference row (`Custom` uses Set1 values) | `Custom` |
| `gnbEnergyReferenceTxPowerDbm` | Reference power for `sp` normalization | `30` dBm |
| `gnbEnergyAntennaRatio` | TR 38.864 antenna dynamic-power fraction `A` | `0.4` |
| `gnbEnergyEtaMode` | PA efficiency mode | `Dual` |
| `rngSeed` / `rngRun` | ns-3 seed and independent run | `42` / `1` |
| `csvFileName` | Main result CSV | `joint_energy_results.csv` |
| `timelineCsvName` | Controller timeline CSV | `active_window_timeline.csv` |

Additional diagnostic options include `rlcBearerSampleMs`,
`bwpHandoverTraceCsv`, `bwpHandoverSampleMs`, and `tbContinuityTraceCsv`.
Traffic-burst options are `enableBurstTraffic`, `synchronizeBurstTraffic`,
`udpBurstRate`, `burstMeanOnMs`, and `burstMeanOffMs`. The complete option list
is also visible with:

```bash
./ns3 run "joint-energy-mvp --PrintHelp"
```

### 1D rule baseline (fixed BWP)

```bash
./ns3 run "joint-energy-mvp \
  --numUes=10 --udpDataRate=2 --simTime=5 \
  --enableGating=true --enableQueueAware=true --enableCdrx=true \
  --enableBwpDynamic=false --enableTrueBwp=false --fixedBwpMHz=100 \
  --cellDtxPeriodMs=40 --cellDtxActiveMs=20 \
  --cdrxOnDurationMs=10 --cdrxInactivityMs=10 \
  --qosTargetDelayMs=50 --controllerInitialActiveMs=20 \
  --controllerStepUpMs=2 --controllerStepDownMs=2 \
  --controllerQueueWarnRatio=0.5 --controllerQueueHighRatio=0.6 \
  --controllerBwpDwellMs=100 --gnbTxPowerDbm=30 \
  --gnbEnergyRefConfig=Custom --gnbEnergyReferenceTxPowerDbm=30 \
  --gnbEnergyAntennaRatio=0.4 --rngSeed=42 --rngRun=1"
```

### 2D rule baseline (dynamic physical 20/100 MHz BWP)

Use the same command, but change the BWP controls:

```bash
--enableBwpDynamic=true --enableTrueBwp=true
```

For a fixed-BWP comparison, use `--enableBwpDynamic=false` and run once with
`--fixedBwpMHz=20` and once with `--fixedBwpMHz=100`. The current grant-span
eligibility patches are active whenever `enableCdrx=true`; they check the
actual DL DCI interval and do not add a fixed full-slot/two-slot guard.

### Parallel greedy-search example (grant-span enforcement)

For a quick rule-parameter search, evaluate candidate configurations in
parallel and keep the lowest-power candidate that satisfies the QoS and loss
constraints. The archived search script starts at most four simulations at a
time and writes one result CSV plus one status log per candidate:

```bash
SIM_TIME=5 bash experiment_archive/run_static_grid_expanded.sh \
  experiment_archive/slot_start_eligibility_2026-10-23/greedy_candidates
```

This sweeps candidate BWP, Cell-DTX active window, C-DRX on-duration,
inactivity timer, and phase offset while keeping grant-span eligibility active
(`enableCdrx=true`). A candidate that aborts with the PHY consistency assertion
is marked failed and must not enter the feasible set.

The formal baseline and comparison runs use `SIM_TIME=5` seconds. A shorter
`SIM_TIME=1.6` is only a fast diagnostic/smoke-search setting and must not be
mixed with the 5-second result tables.

The greedy selection criterion is:

```text
feasible := RlcTxDropPackets == 0
            AND FlowLostPackets == 0
            AND QosSatisfactionRatePct >= 95%
choose argmin(AvgPowerWatts) over feasible candidates
```

For one traffic context, the lowest-power feasible result can be selected with:

```bash
python3 - <<'PY'
import csv, glob

best = None
for path in glob.glob("experiment_archive/slot_start_eligibility_2026-10-23/greedy_candidates/*.csv"):
    if path.endswith(".rlc-bearer.csv"):
        continue
    with open(path, newline="") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        continue
    if int(row["RlcTxDropPackets"]) != 0 or int(row["FlowLostPackets"]) != 0:
        continue
    if float(row["QosSatisfactionRatePct"]) < 95.0:
        continue
    candidate = (float(row["AvgPowerWatts"]), path, row)
    if best is None or candidate[0] < best[0]:
        best = candidate

if best is None:
    raise SystemExit("no feasible slot-start candidate")
print(f"best_avg_power_w={best[0]:.6f}")
print(f"csv={best[1]}")
print(f"cell_dtx_period_ms={best[2]['CellDtxPeriodMs']}")
print(f"cell_dtx_active_ms={best[2]['CellDtxActiveMs']}")
PY
```

For a true greedy refinement rather than a broad sweep, first search coarse
active windows (`10, 20, 30, 40` ms), retain the best feasible candidate, then
search neighbouring values (for example ±2 ms) with the same seed, traffic,
UE count, and QoS target. This keeps the comparison on one time axis and does
not introduce a fixed full-slot guard.

---

## 🌟 Key Features

1. **Orthogonal 5G NR Stack & Topology**: Built with `NrHelper`, `CcBwpCreator`, and 3GPP UMi channel model at 3.5 GHz (\\(n78\\) band, 30 kHz SCS).
2. **3GPP Rel-16/17 DCI-level Alignment**: Broadcasts DCI format 2_6 / Wake-Up Signals (WUS) to dynamically scale On-Duration windows (\\(10\text{ ms} \leftrightarrow 30\text{ ms}\\)) with a \\(2\text{ KB}\\) hysteresis buffer threshold.
3. **Dynamic 2D BWP Switching (TS 38.104 / TS 38.133)**:
   - **Narrow BWP (20 MHz, 51 PRBs)**: Low power decoding (\\(120\text{ mW}\\) Active Rx / \\(20\text{ mW}\\) Active Idle).
   - **Wide BWP (100 MHz, 273 PRBs)**: Full capacity processing (\\(250\text{ mW}\\) Active Rx / \\(45\text{ mW}\\) Active Idle).
   - **Switching Penalty**: Includes a \\(1.0\text{ ms}\\) interruption gap during BWP transition.
4. **Zero-Penalty Power Saving**: Proves that 2D BWP dynamic switching cuts UE power consumption by **11.4% to 16.7%** under light/medium workloads with **0% degradation in P99 latency and QoS SLA compliance**.

---

## 📊 Experimental Results Summary

| Alignment Mode | Traffic Rate | UE Avg Power (mW) | gNB Energy (kWh/TB) | P99 Latency (ms) | QoS Satisfaction (\\(\le 20\text{ ms}\\)) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static-RRC (Static 10ms)** | 2.0 Mbps | 26.63 mW | 50.91 kWh/TB | 24.88 ms | 70.42% |
| **Time-Only (DCI Dynamic)** | 2.0 Mbps | 32.70 mW | 46.38 kWh/TB | 25.80 ms | 68.21% |
| **Joint 2D (DCI + BWP)** | **2.0 Mbps** | **27.24 mW** | **46.38 kWh/TB** | **25.80 ms** | **68.21%** |
| **Static-RRC (Static 10ms)** | 5.0 Mbps | 47.79 mW | 39.59 kWh/TB | 31.72 ms | 68.52% |
| **Time-Only (DCI Dynamic)** | 5.0 Mbps | 74.28 mW | 30.13 kWh/TB | 50.24 ms | 70.72% |
| **Joint 2D (DCI + BWP)** | **5.0 Mbps** | **65.80 mW** | **30.13 kWh/TB** | **50.24 ms** | **70.72%** |
| **Static-RRC (Static 10ms)** | 50.0 Mbps | 64.99 mW | 9.96 kWh/TB | 52.41 ms | 90.70% |
| **Time-Only (DCI Dynamic)** | 50.0 Mbps | 135.84 mW | 7.35 kWh/TB | 53.68 ms | 98.35% |
| **Joint 2D (DCI + BWP)** | **50.0 Mbps** | **134.98 mW** | **7.35 kWh/TB** | **53.68 ms** | **98.35%** |

---

## 🛠️ Repository Structure

```text
├── scratch/
│   └── joint-energy-mvp.cc       # Main 5G NR C++ discrete-event simulation script
├── gen_config_bwp.py             # Generates JSON matrix for batch simulations
├── run_experiments_bwp.py        # Automated batch scheduler & summary printer
├── plot_bwp_2x2.py               # Generates publication-quality 2x2 performance charts
├── joint_bwp_2x2_results.png     # Exported 2x2 performance visualization chart
└── Multi_User_AI_Research_Blueprint_HackMD.md # Multi-User RL scale-up blueprint

🚀 How to Run
# 1. Generate matrix configuration
python3 gen_config_bwp.py

# 2. Execute 9-experiment simulation matrix
python3 run_experiments_bwp.py

# 3. Generate 2x2 performance comparison chart
python3 plot_bwp_2x2.py

---

### 💻 步驟三：在本地端 Terminal 執行 Git 推送指令

在本地端的 `~/workspace/ns-3-dev/` 目錄下依次執行以下命令：

```bash
# 1. 初始化 Git 倉庫（如果尚未初始化）
git init

# 2. 將專案核心檔案納入追蹤
git add scratch/joint-energy-mvp.cc gen_config_bwp.py run_experiments_bwp.py plot_bwp_2x2.py Multi_User_AI_Research_Blueprint_HackMD.md README.md .gitignore joint_bwp_2x2_results.png

# 3. 提交第一個版本 (Commit)
git commit -m "feat: complete 5G-Advanced Cell-DTX, C-DRX, and 2D BWP alignment simulation framework"

# 4. 設定主要分支名稱為 main
git branch -M main

# 5. 綁定您剛建立的 GitHub 遠端倉庫 (請替換為您的真實 GitHub URL)
git remote add origin https://github.com/您的 GitHub 帳號/5g-advanced-cdrx-bwp-ns3.git

# 6. 推送至 GitHub
git push -u origin main
