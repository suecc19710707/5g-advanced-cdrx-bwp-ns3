# 5G-Advanced Joint Cell-DTX, UE C-DRX & Dynamic 2D BWP Switching Framework

This repository contains an advanced **5G-Advanced discrete-event network simulation framework** built on top of **ns-3 (5G-LENA `nr` module)**. It implements a **Joint Time-Frequency 2D Alignment Architecture**, coordinating Base Station **Cell-DTX**, User Equipment **C-DRX**, and DCI-based **Dynamic Bandwidth Part (BWP) Switching (20 MHz ↔ 100 MHz)**.

> The reproducible setup below is the current slot-start eligibility
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

# Reconstruct the energy/BWP history, then apply slot-start eligibility.
cd contrib/nr
git am ../../experiment_archive/official_energy_v5_2026-10-19/000*.patch
git am ../../experiment_archive/true_bwp_dci_v6_2026-10-19/000*.patch
git am ../../experiment_archive/slot_start_eligibility_2026-10-23/0001-*.patch
cd ../..

# Configure and build ns-3 (CMake-based ns-3.48 environment).
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

The final patch implements **slot-start** Cell-DTX and per-UE C-DRX
eligibility. It deliberately does not implement the deferred full-slot or
two-slot guard. The scenario must install the callback described in
`experiment_archive/slot_start_eligibility_2026-10-23/README.md`.

To run the main scenario after building:

```bash
./ns3 run joint-energy-mvp
```

The exact scenario arguments and experiment matrices are kept in the relevant
`experiment_archive/` run scripts.

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
