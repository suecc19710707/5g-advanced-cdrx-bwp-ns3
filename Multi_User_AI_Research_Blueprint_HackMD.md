# 5G-Advanced Joint Cell-DTX + UE C-DRX + BWP 2D 演進藍圖：多用戶規模擴展與 AI 強化學習整合計畫 (HackMD Draft)

> **研究主題**：5G-Advanced 雙重動態（時域 Active Window ✕ 頻域 Dynamic BWP）於多用戶異質場景下之規模擴展與 AI/RL 智慧控制架構  
> **日期**：2026-10-16  
> **目標**：從單區少數用戶（3 UEs）推進至多用戶（10~20+ UEs）資源競爭場景，並導入強化學習（Reinforcement Learning, RL）實現超智慧 2D 動態資源調度。

---

## 📋 階段一：多用戶規模擴展模擬 (Multi-User Scale-Up Simulation)

### 1.1 空間與頻道異質性配置 (Spatial & Channel Heterogeneity)
* **拓樸生成**：使用 `ns3::UniformDiskPositionAllocator`，將 10～20 個 UE 隨機均勻分佈於 gNB 周圍 **30m ~ 200m** 的圓形涵蓋區內。
* **CQI/MCS 階梯分佈**：
  * **近點 UE (30~70m)**：高 CQI / 256-QAM，傳送位元所需 PRB 數少。
  * **中點 UE (70~130m)**：中等 CQI / 64-QAM。
  * **遠點 UE (130~200m)**：低 CQI / QPSK，傳送相同位元需佔用大量 PRB，容易引發 RLC Buffer 滯留與 Queue Drift。

### 1.2 異質業務流量模型 (Heterogeneous Traffic Profile)
將 20 個 UE 劃分為三種典型 5G-Advanced 業務類型：
1. **eMBB 影音重載群 (20% UEs)**：10 ~ 20 Mbps 高速連續下行串流。
2. **Web / 社交應用中載群 (50% UEs)**：2 ~ 5 Mbps 突發型（Bursty On-Off）流量。
3. **Periodic IoT / 語音輕載群 (30% UEs)**：200 ~ 500 kbps 週期性小封包。

### 1.3 多用戶 PRB 資源爭奪與 BWP 觸發機制
* **20 MHz (51 PRBs) 容量瓶頸**：多用戶併發時，51 個 PRB 將快速飽和，驗證頻域 BWP 2D 控制器即時升頻至 **100 MHz (273 PRBs)** 全力衝刺之必要性。
* **公平性指標 (Jain's Fairness Index)**：
  $$\text{Jain's Index} = \frac{(\sum_{i=1}^{N} x_i)^2}{N \cdot \sum_{i=1}^{N} x_i^2}$$
  評估 Active Window 縮短時，遠點 UE 是否會面臨過度的延遲懲罰。

### 1.4 C++ 原始碼 (`joint-energy-mvp.cc`) 修改要點
* 升級 `NodeContainer` 支援動態 `numUes` (10~20)。
* 全區總積壓 (Global RLC Backlog) 與個別 UE 最長滯留延遲雙重監控。
* 輸出多用戶平均 UE 功耗變異數（Power Variance）與 Jain's 公平性係數。

---

## 🧠 階段二：AI 強化學習 (RL) 演算法整合藍圖 (AI/RL Integration)

### 2.1 馬爾可夫決策過程 (MDP) 數學模型

#### **1. 狀態空間 (State Space $S_t$)**
* $B_{\text{global}}$：全區 RLC Buffer 總積壓量 (Bytes)。
* $Q_{\text{max\_delay}}$：當前佇列中最長滯留封包之等待時間 (ms)。
* $\bar{C}_{\text{CQI}}$：全區 UE 之平均 CQI / MCS 階數。
* $W_{\text{active}}$：當前 Active Window 時間 (10.0 ~ 30.0 ms)。
* $B_{\text{state}}$：當前 BWP 頻寬狀態 ($0: 20\text{ MHz}, 1: 100\text{ MHz}$)。

#### **2. 動作空間 (Action Space $A_t$)**
採 2D 離散雙維度動作組合 $a_t = (\Delta W, B_{\text{next}})$：
* 時域階梯：$\Delta W \in \{-4, -2, 0, +2, +4\}\text{ ms}$
* 頻域切換：$B_{\text{next}} \in \{0: 20\text{ MHz}, 1: 100\text{ MHz}\}$

#### **3. 獎勵函數 (Reward Function $R_t$)**
$$R_t = - \Big[ \alpha \cdot P_{\text{gNB}} + \beta \cdot \bar{P}_{\text{UE}} + \gamma \cdot \text{Penalty}(\text{P99 Delay} > 20\text{ms}) + \delta \cdot (1 - \text{Fairness}) \Big]$$
* $\alpha, \beta$：權衡基站與手機端功耗。
* $\gamma$：對超越 20ms QoS SLA Bound 給予指數級重度懲罰。
* $\delta$：確保多用戶間的傳輸公平性。

### 2.2 架構串接與部署方式
* **內部 C++ LibTorch 內嵌推理**：直接將 PyTorch 編譯之微型 Actor-Critic / DQN 神經網路嵌入 `DynamicDtxController`，實現微秒級超低延遲控制。
* **外部 Python Socket / Gym 介面**：透過 Socket 將 ns-3 狀態同步至 Python `Stable-Baselines3` 進行訓練。

---

## 📅 執行里程碑 (Milestones)

* [ ] **Milestone 1**：完成 `joint-energy-mvp.cc` 10~20 UEs 多用戶拓樸與異質流量擴展。
* [ ] **Milestone 2**：跑出 10~20 UEs 多用戶規模下之 Baseline 與 2D 規則式控制器數據。
* [ ] **Milestone 3**：完成 MDP 狀態與獎勵函數設定，並建立 LibTorch/Python 串接介面。
* [ ] **Milestone 4**：完成 RL 代理人訓練，並與 Static-RRC 及 Rule-Based 進行 2x2 多維度數據比較。
