#!/bin/bash
# Phase 0G geometry check: everything needed to answer the cross-target question under mono.
# One process at a time, resumable, three attempts per step.
cd /home/kongwoang/research/risk-aware-perception
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/supervise.log; }
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

competing () { ./scripts/busy.sh; }
gate () {
  while :; do
    [ -e logs/metrics.pause ] && { say "mono: paused"; sleep 20; continue; }
    c=$(competing); [ -n "$c" ] && { say "mono: waiting on pid $c"; sleep 30; continue; }
    u=$(free -m | awk '/^Mem:/ {print $3}')
    [ "$u" -gt 5000 ] && { say "mono: waiting, ${u}MB"; sleep 30; continue; }
    sync; sleep 10; return 0
  done
}
attempt () { local l="$1"; shift
  for t in 1 2 3; do gate; say "$l attempt $t"; "$@" && { say "$l ok"; return 0; }
    say "$l attempt $t FAILED"; sleep $((30*t)); done; say "$l GAVE UP"; return 1; }

# mono test rasters, needed for the truth-referenced Planner C cost
for c in 0 1 2 3 4 5; do
  attempt "pdata/test-mono/c$c" ./scripts/py scripts/70_planner_d_data.py \
      --split test --variant mono --nchunks 6 --chunk $c --skip_existing --tag pdata_test_mono_c$c
done
# Planner C scored against the real trajectory, under mono
attempt "planC-truth/mono" ./scripts/py scripts/74_plannerC_vs_truth.py --variant mono \
    --tag plannerC_vs_truth_mono
# Planner C path-deviation cost under mono (the Phase 0F construction, for completeness)
for c in 0 1 2 3 4 5; do
  attempt "planC/mono/c$c" ./scripts/py scripts/66_planner_c_pkl_planner.py --variant mono \
      --bsz 1 --nworkers 1 --nchunks 6 --chunk $c --skip_existing --tag planner_c_mono_c$c
done
say "MONO GEOMETRY SWEEP DONE"
