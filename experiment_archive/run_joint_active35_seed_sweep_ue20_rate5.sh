#!/usr/bin/env bash
set -euo pipefail
cd /home/suecc3/workspace/ns-3-dev
out="${1:-experiment_archive/torch_cu118_joint_active35_ue20_rate5_bwp100_5s_2026-10-25}"
sim="${SIM_TIME:-5}"; max_jobs="${MAX_JOBS:-4}"
mkdir -p "$out/logs"
run_one() {
  local seed="$1"
  local tag="joint_active35_bwp100_ue20_rate5_qos50_seed${seed}"
  local csv="$out/${tag}.csv" log="$out/logs/${tag}.log" status="$out/logs/${tag}.status"
  [[ -f "$csv" && -f "$status" ]] && return 0
  if ./build/scratch/ns3-dev-joint-energy-mvp-default \
      --simTime="$sim" --numUes=20 --udpDataRate=5 \
      --enableGating=true --enableQueueAware=false --enableCdrx=true \
      --enableDciDynamic=true --enableBwpDynamic=false --enableTrueBwp=true \
      --bwpInactivityTimerMs=40 --cellDtxPeriodMs=40 --cellDtxActiveMs=35 \
      --cdrxOnDurationMs=35 --cdrxInactivityMs=10 --phaseOffsetMs=0 \
      --qosTargetDelayMs=50 --fixedBwpMHz=100 \
      --rngSeed="$seed" --rngRun=1 --gnbTxPowerDbm=30 \
      --gnbEnergyRefConfig=Custom --gnbEnergyReferenceTxPowerDbm=30 \
      --gnbEnergyAntennaRatio=0.4 --csvFileName="$csv" \
      --timelineCsvName="$out/${tag}.timeline.csv" >"$log" 2>&1; then
    echo exit=0 >"$status"
  else
    echo exit=1 >"$status"; return 1
  fi
}
for seed in 101 202 303 404 505 606 707; do
  run_one "$seed" &
  while (( $(jobs -rp | wc -l) >= max_jobs )); do wait -n || true; done
done
wait || true
echo "completed=7" >"$out/COMPLETE"
