#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. 設定繪圖主題與字型
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 2. 檢查並讀取 CSV 模擬數據
csv_file = "joint_bwp_energy_results.csv"
if not os.path.exists(csv_file):
    print(f"❌ 找不到數據檔案：{csv_file}")
    exit(1)

df = pd.read_csv(csv_file)

# 🛠️ 正確 3GPP 標準單位換算 (1 TB = 8,000,000 Mbits, 1 kWh = 3,600,000 Joules)
# kWh/TB = (Joules / 3,600,000) / (Mbits / 8,000,000) = (20 / 9) / EnergyEfficiency_MbitsPerJoule
if 'EnergyEfficiency_MbitsPerJoule' in df.columns and (df['EnergyEfficiency_MbitsPerJoule'] > 0).all():
    df['kWhPerTB'] = (20.0 / 9.0) / df['EnergyEfficiency_MbitsPerJoule']
elif 'TotalEnergyJoules' in df.columns and 'TotalRxMbits' in df.columns:
    df['kWhPerTB'] = (df['TotalEnergyJoules'] * 20.0) / (df['TotalRxMbits'] * 9.0)
else:
    df['kWhPerTB'] = 0.0

# 3. 定義對齊模式標籤
def get_mode_label(row):
    dci = row.get('EnableDciDynamic', 0)
    bwp = row.get('EnableBwpDynamic', 0)
    qa = row.get('EnableQueueAware', 0)
    if qa == 1 and dci == 1 and bwp == 1:
        return "Joint 2D (DCI + BWP)"
    elif qa == 1 and dci == 1 and bwp == 0:
        return "Time-Only (DCI Dynamic)"
    else:
        return "Static-RRC (Static 10ms)"

df['Mode'] = df.apply(get_mode_label, axis=1)

# 4. 建立 2x2 子圖看板
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("5G-Advanced C-DRX (Time) x BWP (Freq) 2D Alignment Performance (2M / 5M / 50M)", fontsize=16, fontweight='bold', y=0.98)

rates = sorted(df['UdpDataRateMbps'].unique())
x = np.arange(len(rates))
width = 0.25

modes = ["Static-RRC (Static 10ms)", "Time-Only (DCI Dynamic)", "Joint 2D (DCI + BWP)"]
colors = ['#7f7f7f', '#1f77b4', '#2ca02c']

# [子圖 1] UE 平均功耗 (mW)
ax1 = axes[0, 0]
for idx, mode in enumerate(modes):
    sub = df[df['Mode'] == mode].sort_values('UdpDataRateMbps')
    vals = sub['UeAvgPowerMw'].values if 'UeAvgPowerMw' in sub.columns else np.zeros(len(rates))
    rects = ax1.bar(x + idx*width - width, vals, width, label=mode, color=colors[idx])
    ax1.bar_label(rects, fmt='%.1f', padding=3, fontsize=9)

ax1.set_title("UE Average Power Consumption (mW)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Power (mW)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{r} Mbps" for r in rates])
ax1.legend(fontsize=9)
if len(df['UeAvgPowerMw']) > 0 and max(df['UeAvgPowerMw']) > 0:
    ax1.set_ylim(0, max(df['UeAvgPowerMw']) * 1.25)

# [子圖 2] gNB 單位能耗 (kWh/TB)
ax2 = axes[0, 1]
for idx, mode in enumerate(modes):
    sub = df[df['Mode'] == mode].sort_values('UdpDataRateMbps')
    vals = sub['kWhPerTB'].values
    rects = ax2.bar(x + idx*width - width, vals, width, label=mode, color=colors[idx])
    ax2.bar_label(rects, fmt='%.1f', padding=3, fontsize=9)

ax2.set_title("gNB Energy Efficiency (kWh / TB)", fontsize=12, fontweight='bold')
ax2.set_ylabel("kWh / TB")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{r} Mbps" for r in rates])
ax2.legend(fontsize=9)
if len(df['kWhPerTB']) > 0 and max(df['kWhPerTB']) > 0:
    ax2.set_ylim(0, max(df['kWhPerTB']) * 1.25)

# [子圖 3] P99 尾端延遲 (ms)
ax3 = axes[1, 0]
for idx, mode in enumerate(modes):
    sub = df[df['Mode'] == mode].sort_values('UdpDataRateMbps')
    vals = sub['P99LatencyMs'].values if 'P99LatencyMs' in sub.columns else np.zeros(len(rates))
    rects = ax3.bar(x + idx*width - width, vals, width, label=mode, color=colors[idx])
    ax3.bar_label(rects, fmt='%.1f', padding=3, fontsize=9)

ax3.axhline(20.0, color='red', linestyle='--', linewidth=1.5, label='QoS Target Bound (20ms)')
ax3.set_title("P99 Tail Latency (ms)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Latency (ms)")
ax3.set_xticks(x)
ax3.set_xticklabels([f"{r} Mbps" for r in rates])
ax3.legend(fontsize=9)
if len(df['P99LatencyMs']) > 0 and max(df['P99LatencyMs']) > 0:
    ax3.set_ylim(0, max(df['P99LatencyMs']) * 1.25)

# [子圖 4] QoS SLA 達標率 (%)
ax4 = axes[1, 1]
for idx, mode in enumerate(modes):
    sub = df[df['Mode'] == mode].sort_values('UdpDataRateMbps')
    vals = sub['QosSatisfactionRatePct'].values if 'QosSatisfactionRatePct' in sub.columns else np.zeros(len(rates))
    rects = ax4.bar(x + idx*width - width, vals, width, label=mode, color=colors[idx])
    ax4.bar_label(rects, fmt='%.1f%%', padding=3, fontsize=9)

ax4.set_title("QoS SLA Bound Satisfaction Rate (<= 20ms %)", fontsize=12, fontweight='bold')
ax4.set_ylabel("QoS Rate (%)")
ax4.set_xticks(x)
ax4.set_xticklabels([f"{r} Mbps" for r in rates])
ax4.legend(fontsize=9)
ax4.set_ylim(0, 115)

# 5. 導出高解析度圖表
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_img = "joint_bwp_2x2_results.png"
plt.savefig(output_img, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ 成功繪製並儲存 2x2 圖表至：{output_img}")
