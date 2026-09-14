#!/bin/bash
# Task 2 continuation (RESEARCH_LOG 2026-09-14): R1 crashed on a file name containing nuPlan's "n/a"
# geometry after the supervisor had started R2's image cache.  Wait for that cache to finish, then R1,
# R2 training and the router budget tables, one heavy job at a time.  Logs to logs/supervise_tasks.log.
cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise_tasks.log
say () { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
wait_free () { while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done; sync; sleep 5; }
step () {
  local name=$1; shift
  wait_free
  say "$name start"
  if "$@" >> logs/$name.log 2>&1; then say "$name ok"; return 0; else say "$name FAILED (exit $?)"; return 1; fi
}

say "task 2 continuation started; waiting for the R2 image cache"
wait_free
if [ -f data/cache/router_r2/KITTI.npz ] && [ -f data/cache/router_r2/nuScenes.npz ]; then
  say "router_r2_cache ok (both caches present)"
  R2=1
else
  step router_r2_cache ./scripts/py scripts/107_router_r2.py --stage cache && R2=1 || R2=0
fi
step router_r1 ./scripts/py scripts/103_routers_r1.py
[ "$R2" = 1 ] && step router_r2_train ./scripts/py scripts/107_router_r2.py --stage train
step router_budget ./scripts/py scripts/93_budget_allocation.py --routers --tag benchmark_budget_routers
say "TASK2 COMPUTE DONE"
