#!/usr/bin/env bash
set -euo pipefail
cd /home/suecc3/workspace/ns-3-dev
out="${1:-experiment_archive/torch_cu118_onon_bwp20_capacity_sweep_5s_2026-10-26}"
sim="${SIM_TIME:-5}"; max_jobs="${MAX_JOBS:-4}"
mkdir -p "$out/logs"
run_one() {
  local active="$1" rate="$2" seed="$3" rate_tag="${rate/./p}"
  local tag="onon_active${active}_bwp20_ue20_rate${rate_tag}_qos50_seed${seed}"
  local csv="$out/${tag}.csv" log="$out/logs/${tag}.log" status="$out/logs/${tag}.status"
  [[ -f "$csv" && -f "$status" ]] && return 0
  if ./build/scratch/ns3-dev-joint-energy-mvp-default \
      --simTime="$sim" --numUes=20 --udpDataRate="$rate" \
      --enableGating=true --enableQueueAware=false --enableCdrx=true \
      --enableDciDynamic=true --enableBwpDynamic=false --enableTrueBwp=true \
      --bwpInactivityTimerMs=40 --cellDtxPeriodMs=40 --cellDtxActiveMs="$active" \
      --cdrxOnDurationMs="$active" --cdrxInactivityMs=10 --phaseOffsetMs=0 \
      --qosTargetDelayMs=50 --fixedBwpMHz=20 \
      --rngSeed="$seed" --rngRun=1 --gnbTxPowerDbm=30 \
      --gnbEnergyRefConfig=Custom --gnbEnergyReferenceTxPowerDbm=30 \
      --gnbEnergyAntennaRatio=0.4 --csvFileName="$csv" \
      --timelineCsvName="$out/${tag}.timeline.csv" >"$log" 2>&1; then
    echo exit=0 >"$status"
  else
    echo exit=1 >"$status"; return 1
  fi
}
for active in 35 40; do
  for rate in 0.5 1 2 3 4 5; do
    for seed in 101 202 303 404 505 606 707; do
      run_one "$active" "$rate" "$seed" &
      while (( $(jobs -rp | wc -l) >= max_jobs )); do wait -n || true; done
    done
  done
done
wait || true
echo "completed=84" >"$out/COMPLETE"
