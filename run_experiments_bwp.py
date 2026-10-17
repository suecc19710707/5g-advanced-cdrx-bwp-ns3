#!/usr/bin/env python3
import json
import subprocess
import os
import pandas as pd

pd.set_option('display.unicode.east_asian_width', True)

def main():
    config_file = "config_bwp.json"
    if not os.path.exists(config_file):
        import gen_config_bwp
        gen_config_bwp.generate_config()

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    defaults = config.get("simulation_defaults", {})
    sim_time = defaults.get("simTime", 5.0)
    csv_file = defaults.get("csvFileName", "joint_bwp_energy_results.csv")

    if os.path.exists(csv_file):
        os.remove(csv_file)

    experiments = config.get("experiments", [])
    print(f"🚀 開始執行 5G-Advanced C-DRX + BWP 2D 動態批次實驗 (共 {len(experiments)} 組)...")

    for idx, exp in enumerate(experiments, 1):
        name = exp["name"]
        rate = exp["udpDataRate"]
        gating = exp["enableGating"]
        queue_aware = exp["enableQueueAware"]
        cdrx = exp["enableCdrx"]
        dci_dyn = exp["enableDciDynamic"]
        bwp_dyn = exp["enableBwpDynamic"]
        offset = exp.get("phaseOffsetMs", 0.0)

        cmd = (
            f'./ns3 run "scratch/joint-energy-mvp '
            f'--simTime={sim_time} '
            f'--udpDataRate={rate} '
            f'--enableGating={"true" if gating else "false"} '
            f'--enableQueueAware={"true" if queue_aware else "false"} '
            f'--enableCdrx={"true" if cdrx else "false"} '
            f'--enableDciDynamic={"true" if dci_dyn else "false"} '
            f'--enableBwpDynamic={"true" if bwp_dyn else "false"} '
            f'--phaseOffsetMs={offset} '
            f'--csvFileName={csv_file}"'
        )

        print(f"\n[{idx}/{len(experiments)}] 正在執行: {name} (DataRate={rate}Mbps)")
        subprocess.run(cmd, shell=True)

    print("\n" + "=" * 105)
    print(" 📊 5G-Advanced C-DRX (時域) ✕ BWP (頻域) 雙重動態對齊實測總覽表")
    print("=" * 105)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        def get_mode_label(row):
            dci = row.get('EnableDciDynamic', 0)
            bwp = row.get('EnableBwpDynamic', 0)
            qa = row.get('EnableQueueAware', 0)
            if qa == 1 and dci == 1 and bwp == 1:
                return "Joint 2D (DCI + BWP 20/100MHz)"
            elif qa == 1 and dci == 1 and bwp == 0:
                return "Time-Only (DCI Dynamic, 100MHz)"
            else:
                return "Static-RRC (Static 10ms, 100MHz)"

        df['對齊模式'] = df.apply(get_mode_label, axis=1)
        display_cols = ['對齊模式', 'UdpDataRateMbps', 'UeAvgPowerMw', 'kWhPerTB', 'P99LatencyMs', 'QosSatisfactionRatePct']
        summary_df = df[[c for c in display_cols if c in df.columns]].copy()
        
        summary_df.rename(columns={
            'UdpDataRateMbps': '流量(Mbps)',
            'UeAvgPowerMw': 'UE功耗(mW)',
            'kWhPerTB': 'gNB能耗(kWh/TB)',
            'P99LatencyMs': 'P99延遲(ms)',
            'QosSatisfactionRatePct': 'QoS達標率(%)'
        }, inplace=True)

        print(summary_df.to_string(index=False))
        print("=" * 105 + "\n")

if __name__ == "__main__":
    main()
