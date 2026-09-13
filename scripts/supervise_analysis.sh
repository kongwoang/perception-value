#!/bin/bash
# Everything that runs on top of the rebuilt compute, then the first untouched nuPlan simulation.
# Waits for the rebuild to finish rather than racing it: this board has room for one job.
cd /home/kongwoang/research/risk-aware-perception
PAUSE=logs/metrics.pause
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/supervise.log; }
competing () { ./scripts/busy.sh; }
gate () { while :; do
    [ -e "$PAUSE" ] && { say "analysis: paused"; sleep 20; continue; }
    c=$(competing); [ -n "$c" ] && { say "analysis: waiting on pid $c"; sleep 30; continue; }
    u=$(free -m | awk '/^Mem:/ {print $3}')
    [ "$u" -gt 5000 ] && { say "analysis: waiting, ${u}MB resident"; sleep 30; continue; }
    sync; sleep 10; return 0; done; }
attempt () { local l="$1"; shift
  for t in 1 2 3; do gate; say "$l attempt $t"; "$@" && { say "$l ok"; return 0; }
    say "$l attempt $t FAILED"; sleep $((30*t)); done; say "$l GAVE UP"; return 1; }

say "analysis: waiting for the rebuild"
while ! grep -q "REBUILD COMPUTE DONE" logs/supervise.log; do sleep 120; done
say "analysis: rebuild finished"

# Phase 0F headline table across all eight cells, with the tie-break averaged out
attempt "core_matrix" ./scripts/py scripts/52_core_matrix.py --tag core_matrix_postreview

# eta tables, both geometry variants, including Planner C's two costs
for v in oracle mono; do
  attempt "eta/$v" ./scripts/py scripts/62_planning_metric_eta.py --variant $v --nboot 400 \
      --coverage per_metric --tag phase0g_eta_$v
done

# the score-free cross-target measurement, both variants
CM=results/raw/20260912_071140_core_matrix
attempt "cross/oracle" ./scripts/py scripts/75_cross_target_measurement.py \
    --brake_table $CM/nuScenes__YOLOv8s__ns_cheap_320tons_full_640__oracle.pkl \
    --plan_csv data/cache/planner_d/planC_vs_truth.csv --label oracle --nboot 400 \
    --tag cross_target_oracle
attempt "cross/mono" ./scripts/py scripts/75_cross_target_measurement.py \
    --brake_table $CM/nuScenes__YOLOv8s__ns_cheap_320tons_full_640__mono.pkl \
    --plan_csv data/cache/planner_d/planC_vs_truth_mono.csv --label mono --nboot 400 \
    --tag cross_target_mono

# Planner D, for the limitations sentence only: D-F4 fired, so no eta of its is interpreted
for f in data/cache/planner_d/planner_d_*_best.pt; do
  attempt "planD-eval/$(basename $f)" ./scripts/py scripts/72_planner_d_eval.py --ckpt "$f"
done

attempt "figures" ./scripts/py scripts/63_phase0f_figures.py
say "ANALYSIS DONE"

# Track B step B2: one untouched official simulation per external planner
attempt "nuplan-smoke" bash scripts/80_nuplan_smoke.sh
say "ALL DONE"
