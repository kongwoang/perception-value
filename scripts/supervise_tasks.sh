#!/bin/bash
# Task 1 then Task 2 compute (RESEARCH_LOG 2026-09-14), strictly one heavy job at a time.
#
# Runs detached from the Claude session, because the previous session ended while 100 was running.
# Waits for the calibration outcome run, refuses to continue Task 1 unless its equivalence checks
# passed, and records every step's start, end and exit status in logs/supervise_tasks.log.
# Task 2 does not depend on Task 1's results, so it runs even if Task 1 stops.
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

say "supervisor started; waiting for 100_calibration_outcomes"
wait_free
OC=$(ls -d results/raw/*_calibration_outcomes | tail -1)
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OC/checks.json'))['all_pass'] else 1)" 2>/dev/null; then
  say "task1 checks 1-2 PASS ($OC)"
  step calib_submissions ./scripts/py scripts/60_build_submissions.py --op_conf 0.10 --op_conf_full 0.10 \
       --out data/cache/nusc_submissions_calib \
    && step calib_plan ./scripts/py scripts/101_calibration_plan.py --outcomes "$OC" \
    && step calib_cells ./scripts/py scripts/102_calibration_cells.py
else
  say "task1 checks 1-2 FAILED or missing in $OC -- Task 1 stopped before any scheme"
fi
say "TASK1 COMPUTE DONE"

step router_tracks ./scripts/pynuplan scripts/104_nuplan_track_lists.py
step router_r1 ./scripts/py scripts/103_routers_r1.py
step router_r2_cache ./scripts/py scripts/107_router_r2.py --stage cache \
  && step router_r2_train ./scripts/py scripts/107_router_r2.py --stage train
step router_budget ./scripts/py scripts/93_budget_allocation.py --routers --tag benchmark_budget_routers
say "TASK2 COMPUTE DONE"
