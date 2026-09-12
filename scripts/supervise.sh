#!/bin/bash
# Supervised runner for the PKL / TIP / Planner C sweeps.
#
# This board serves GPU allocations from the same physical RAM as the host, and the driver
# needs a large contiguous block.  Twice a chunk died with "CUDA out of memory" while asking
# for 20-128 MiB with gigabytes nominally free, each time because another process was running
# alongside (a Planner B sweep once, a 48 MB git push the other).  Neither was a fault in the
# metric code: both chunks succeeded unchanged once the board was idle.
#
# So every chunk runs behind three gates:
#   1. a pause lock, so foreground work (a push, an analysis run) can stop the sweep cleanly
#      rather than racing it -- `touch logs/metrics.pause` to hold, remove it to resume;
#   2. an idleness gate -- no competing heavy process, and resident memory below a threshold;
#   3. three attempts with escalating backoff, each preceded by sync and a settle delay.
#
# Chunks whose CSV already exists are skipped, so an interrupted sweep resumes.

cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise.log
PAUSE=logs/metrics.pause
GATE_MB=5000          # resident megabytes above which we do not start a new chunk
SETTLE=15             # seconds to let the kernel settle after the gate opens
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

used_mb () { free -m | awk '/^Mem:/ {print $3}'; }

competing () {
  # heavy processes that are not this sweep: other python jobs, git packing, compressors
  pgrep -f "65_planner_b|62_planning_metric|git (push|gc|repack)|git-pack|xz|zstd" \
    | grep -v "^$$\$" | head -1
}

gate () {
  while :; do
    if [ -e "$PAUSE" ]; then say "paused (remove $PAUSE to resume)"; sleep 20; continue; fi
    c=$(competing)
    if [ -n "$c" ]; then say "waiting: competing pid $c"; sleep 20; continue; fi
    u=$(used_mb)
    if [ "$u" -gt "$GATE_MB" ]; then say "waiting: ${u}MB resident > ${GATE_MB}MB"; sleep 20; continue; fi
    sync; sleep "$SETTLE"
    return 0
  done
}

# attempt <label> <command...>
attempt () {
  local label="$1"; shift
  for try in 1 2 3; do
    gate
    say "$label attempt $try"
    if "$@"; then say "$label ok"; return 0; fi
    say "$label attempt $try FAILED"
    sleep $((30 * try))
  done
  say "$label GAVE UP after 3 attempts"
  return 1
}

NC=6
metric_chunk () {  # metric variant chunk
  attempt "$1/$2/c$3" ./scripts/py scripts/61_run_planning_metric.py \
      --metric "$1" --variant "$2" --bsz 1 --nworkers 1 \
      --nchunks $NC --chunk "$3" --tag "$1_$2_c$3" --skip_existing
}

dataset_chunk () {  # split chunk nchunks
  attempt "pdata/$1/c$2" ./scripts/py scripts/70_planner_d_data.py \
      --split "$1" --nchunks "$3" --chunk "$2" --skip_existing --tag pdata_$1_c$2
}

planc_chunk () {   # variant chunk
  attempt "planC/$1/c$2" ./scripts/py scripts/66_planner_c_pkl_planner.py \
      --variant "$1" --bsz 1 --nworkers 1 \
      --nchunks $NC --chunk "$2" --tag planner_c_$1_c$2 --skip_existing
}

case "${1:-all}" in
  oracle_metrics) for m in pkl tip; do for c in $(seq 0 $((NC-1))); do metric_chunk $m oracle $c; done; done ;;
  planc)          for c in $(seq 0 $((NC-1))); do planc_chunk oracle $c; done ;;
  mono_metrics)   for m in pkl tip; do for c in $(seq 0 $((NC-1))); do metric_chunk $m mono $c; done; done ;;
  planner_d_data) for c in $(seq 0 4);  do dataset_chunk val   $c 5;  done
                  for c in $(seq 0 9);  do dataset_chunk train $c 10; done
                  for c in $(seq 0 5);  do dataset_chunk test  $c 6;  done ;;
  all)            "$0" oracle_metrics; "$0" planc; "$0" mono_metrics ;;
esac
say "SWEEP ${1:-all} DONE"
