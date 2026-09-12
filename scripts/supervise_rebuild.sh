#!/bin/bash
# Full rebuild after the 2026-09-13 code review.  Everything downstream of the detection
# submissions is recomputed, because the monocular lift wrote the near-face range as the box
# centre (review D3) and that contaminated every submission -- fully for the mono variant, and
# in the false-positive boxes for the oracle variant.
#
# Pre-review artefacts are kept under data/cache/stale_pre_review/ so the old numbers stay
# auditable rather than being silently replaced.
#
# One process at a time, three attempts each, resumable: chunks whose output exists are skipped.
cd /home/kongwoang/research/risk-aware-perception
PAUSE=logs/metrics.pause
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/supervise.log; }
competing () { pgrep -f "61_run_planning|62_planning|66_planner_c|70_planner_d|71_planner_d|74_plannerC|run_simulation|git (push|gc|repack)" | grep -v "^$$\$" | head -1; }
gate () { while :; do
    [ -e "$PAUSE" ] && { say "rebuild: paused"; sleep 20; continue; }
    c=$(competing); [ -n "$c" ] && { say "rebuild: waiting on pid $c"; sleep 30; continue; }
    u=$(free -m | awk '/^Mem:/ {print $3}')
    [ "$u" -gt 5000 ] && { say "rebuild: waiting, ${u}MB"; sleep 30; continue; }
    sync; sleep 10; return 0; done; }
attempt () { local l="$1"; shift
  for t in 1 2 3; do gate; say "$l attempt $t"; "$@" && { say "$l ok"; return 0; }
    say "$l attempt $t FAILED"; sleep $((30*t)); done; say "$l GAVE UP"; return 1; }

# 1. submissions, with the corrected monocular lift
attempt "submissions" ./scripts/py scripts/60_build_submissions.py --tag submissions_postreview

# 2. the two published planning-aware metrics, both geometry variants
for v in oracle mono; do for m in pkl tip; do for c in 0 1 2 3 4 5; do
  attempt "$m/$v/c$c" ./scripts/py scripts/61_run_planning_metric.py --metric $m --variant $v \
      --bsz 1 --nworkers 1 --nchunks 6 --chunk $c --skip_existing --tag ${m}_${v}_c${c}
done; done; done

# 3. Planner C: the cached rasters it and 74 need, then both of its costs
for v in oracle mono; do for c in 0 1 2 3 4 5; do
  attempt "pdata/test-$v/c$c" ./scripts/py scripts/70_planner_d_data.py --split test \
      --variant $v --nchunks 6 --chunk $c --skip_existing --tag pdata_test_${v}_c$c
done; done
for v in oracle mono; do
  attempt "planC-truth/$v" ./scripts/py scripts/74_plannerC_vs_truth.py --variant $v \
      --tag plannerC_vs_truth_$v
  for c in 0 1 2 3 4 5; do
    attempt "planC/$v/c$c" ./scripts/py scripts/66_planner_c_pkl_planner.py --variant $v \
        --bsz 1 --nworkers 1 --nchunks 6 --chunk $c --skip_existing --tag planner_c_${v}_c$c
  done
done
say "REBUILD COMPUTE DONE"
