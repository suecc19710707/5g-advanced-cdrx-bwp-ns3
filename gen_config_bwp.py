#!/usr/bin/env python3
import json

def generate_config():
    config = {
        "simulation_defaults": {
            "simTime": 5.0,
            "csvFileName": "joint_bwp_energy_results.csv"
        },
        "experiments": [
            # =========================================================================
            # 1. 2.0 Mbps 輕載 (Total 6.0 Mbps) - 3組對稱模式
            # =========================================================================
            {
                "name": "Exp1_2M_Static_RRC",
                "udpDataRate": 2.0,
                "enableGating": True,
                "enableQueueAware": False,
                "enableCdrx": True,
                "enableDciDynamic": False,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp2_2M_Time_Only",
                "udpDataRate": 2.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp3_2M_Joint_2D",
                "udpDataRate": 2.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": True,
                "phaseOffsetMs": 0.0
            },

            # =========================================================================
            # 2. 5.0 Mbps 中載 (Total 15.0 Mbps) - 3組對稱模式
            # =========================================================================
            {
                "name": "Exp4_5M_Static_RRC",
                "udpDataRate": 5.0,
                "enableGating": True,
                "enableQueueAware": False,
                "enableCdrx": True,
                "enableDciDynamic": False,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp5_5M_Time_Only",
                "udpDataRate": 5.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp6_5M_Joint_2D",
                "udpDataRate": 5.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": True,
                "phaseOffsetMs": 0.0
            },

            # =========================================================================
            # 3. 50.0 Mbps 重載 (Total 150.0 Mbps) - 3組對稱模式
            # =========================================================================
            {
                "name": "Exp7_50M_Static_RRC",
                "udpDataRate": 50.0,
                "enableGating": True,
                "enableQueueAware": False,
                "enableCdrx": True,
                "enableDciDynamic": False,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp8_50M_Time_Only",
                "udpDataRate": 50.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": False,
                "phaseOffsetMs": 0.0
            },
            {
                "name": "Exp9_50M_Joint_2D",
                "udpDataRate": 50.0,
                "enableGating": True,
                "enableQueueAware": True,
                "enableCdrx": True,
                "enableDciDynamic": True,
                "enableBwpDynamic": True,
                "phaseOffsetMs": 0.0
            }
        ]
    }

    with open("config_bwp.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("✅ 已成功寫入 config_bwp.json (包含 9 組實驗對照矩陣)")

if __name__ == "__main__":
    generate_config()
