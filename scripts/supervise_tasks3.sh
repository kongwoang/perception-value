#!/bin/bash
# Task 2 after the reboot (RESEARCH_LOG 12:55): R2 per dataset and per step, then the router budget.
# One heavy job at a time; a step starts only with at least 6 GB of memory available.
cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise_tasks.log
PT=results/raw/20260914_114453_router_r2/r2_nuScenes.pt
say () { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
wait_free () {
  while [ -n "$(./scripts/busy.sh)" ] || [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 6000 ]; do sleep 30; done
  sync; sleep 5
}
step () {
  local name=$1; shift
  wait_free
  say "$name start"
  if "$@" >> logs/$name.log 2>&1; then say "$name ok"; return 0; else say "$name FAILED (exit $?)"; return 1; fi
}

say "task 2 resumed after the reboot"
RUN=$(./scripts/py -c "import sys; sys.path.insert(0,'src'); from rap import runmeta; print(runmeta.new_run('router_r2', {'resumed_after_reboot': True}))" | tail -1)
say "router_r2 run dir $RUN"
step r2_train_nuscenes ./scripts/py scripts/107_router_r2.py --stage train --dataset nuScenes --run "$RUN" --reuse_pt "$PT" \
  && step r2_export_nuscenes ./scripts/py scripts/107_router_r2.py --stage export --dataset nuScenes --run "$RUN"
step r2_train_kitti ./scripts/py scripts/107_router_r2.py --stage train --dataset KITTI --run "$RUN" \
  && step r2_export_kitti ./scripts/py scripts/107_router_r2.py --stage export --dataset KITTI --run "$RUN"
step router_budget ./scripts/py scripts/93_budget_allocation.py --routers --tag benchmark_budget_routers
say "TASK2 COMPUTE DONE"
