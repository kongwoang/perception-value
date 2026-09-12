#!/bin/bash
# Supervised Planner D pipeline: wait for the raster cache, then train the six models.
#
# A separate file from supervise.sh on purpose: bash reads a script incrementally, so editing
# supervise.sh while an instance of it is mid-loop risks corrupting the parse.  It also carries
# its own guard against the mistake that already happened once -- clearing the pause lock woke a
# still-alive mono loop, and two jobs ran at once on a board that has room for one.
cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise.log
PAUSE=logs/metrics.pause
GATE_MB=5000
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

competing () {
  pgrep -f "61_run_planning|66_planner_c|70_planner_d_data|62_planning_metric|git (push|gc|repack)" \
    | grep -v "^$$\$" | head -1
}

gate () {
  while :; do
    if [ -e "$PAUSE" ]; then say "planner_d: paused"; sleep 20; continue; fi
    c=$(competing); if [ -n "$c" ]; then say "planner_d: waiting on pid $c"; sleep 30; continue; fi
    u=$(free -m | awk '/^Mem:/ {print $3}')
    if [ "$u" -gt "$GATE_MB" ]; then say "planner_d: waiting, ${u}MB resident"; sleep 30; continue; fi
    sync; sleep 10; return 0
  done
}

attempt () {
  local label="$1"; shift
  for try in 1 2 3; do
    gate
    say "$label attempt $try"
    if "$@"; then say "$label ok"; return 0; fi
    say "$label attempt $try FAILED"; sleep $((30 * try))
  done
  say "$label GAVE UP"; return 1
}

# 1. wait for the raster cache: 5 val + 10 train + 6 test chunks
say "planner_d: waiting for the raster cache (21 chunks)"
while [ "$(ls data/cache/planner_d/*/chunk_*.npz 2>/dev/null | wc -l)" -lt 21 ]; do
  if grep -q "GAVE UP" <(tail -n 60 logs/supervise.log); then
    say "planner_d: a data chunk gave up -- stopping before training"; exit 1
  fi
  sleep 120
done
say "planner_d: raster cache complete"

# 2. six models: two pre-registered variants x three seeds
for v in gt aug; do
  for s in 0 1 2; do
    attempt "trainD/$v/seed$s" ./scripts/py scripts/71_planner_d_train.py \
        --variant "$v" --seed "$s" || true
  done
done
say "PLANNER D TRAINING DONE"
