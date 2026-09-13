#!/bin/bash
# Everything after the PDM-Closed run, strictly one job at a time.
cd /home/kongwoang/research/risk-aware-perception
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/supervise.log; }
gate () { while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done; sync; sleep 10; }
attempt () { local l="$1"; shift
  for t in 1 2; do gate; say "$l attempt $t"; "$@" && { say "$l ok"; return 0; }
    say "$l attempt $t FAILED"; sleep 60; done; say "$l GAVE UP"; return 1; }

say "final: waiting for the PDM-Closed run"
while ! grep -q "TRACKB BIG DONE" logs/supervise.log; do sleep 60; done
say "final: Track B compute finished"

CM=$(ls -d results/raw/*core_matrix_postreview 2>/dev/null | tail -1)
[ -n "$CM" ] || CM=results/raw/20260912_071140_core_matrix

attempt "B7B8" ./scripts/py scripts/83_external_transfer.py --nboot 400
for v in oracle mono; do
  sfx=""; [ "$v" = mono ] && sfx="_mono"
  attempt "cross-tierobust/$v" ./scripts/py scripts/75_cross_target_measurement.py \
      --brake_table "$CM/nuScenes__YOLOv8s__ns_cheap_320tons_full_640__$v.pkl" \
      --plan_csv "data/cache/planner_d/planC_vs_truth$sfx.csv" --label $v --nboot 400 \
      --tag cross_target_tierobust_$v
done
for v in oracle mono; do
  attempt "eta-fde/$v" ./scripts/py scripts/62_planning_metric_eta.py --variant $v --nboot 400 \
      --coverage per_metric --tag phase0g_eta_fde_$v
done
attempt "deployable-gate" ./scripts/py scripts/84_deployable_gate.py --nboot 400
say "FINAL COMPUTE DONE"
