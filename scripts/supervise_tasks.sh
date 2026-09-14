#!/bin/bash
# Task 1 then Task 2 compute (RESEARCH_LOG 2026-09-14), strictly one heavy job at a time.
#
# Runs detached from the Claude session.  Resumes Task 1 after the q_plan out-of-memory
# (RESEARCH_LOG 11:10): the outcome run with Planner B fixed has passed checks 1-2, the 0.10
# submissions are built, and q_plan now runs as a CPU box stage and a GPU planner stage.
# Task 2 does not depend on Task 1's results, so it runs even if Task 1 stops.
# Every step's start, end and exit status goes to logs/supervise_tasks.log.
cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise_tasks.log
OC=results/raw/20260914_103252_calibration_outcomes
say () { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
wait_free () { while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done; sync; sleep 5; }
step () {
  local name=$1; shift
  wait_free
  say "$name start"
  if "$@" >> logs/$name.log 2>&1; then say "$name ok"; return 0; else say "$name FAILED (exit $?)"; return 1; fi
}

say "supervisor (re)started: Task 1 from q_plan"
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OC/checks.json'))['all_pass'] else 1)" 2>/dev/null; then
  step calib_plan_boxes ./scripts/py scripts/101_calibration_plan.py --stage boxes \
    && step calib_plan ./scripts/py scripts/101_calibration_plan.py --stage plan --outcomes "$OC" \
    && step calib_cells ./scripts/py scripts/102_calibration_cells.py
else
  say "task1 checks 1-2 not passing in $OC -- Task 1 stopped"
fi
say "TASK1 COMPUTE DONE"

wait_free
if ls results/raw/*_nuplan_track_lists/nuplan_track_features.npz > /dev/null 2>&1; then
  say "router_tracks already produced -- skipped"
else
  step router_tracks ./scripts/pynuplan scripts/104_nuplan_track_lists.py
fi
step router_r1 ./scripts/py scripts/103_routers_r1.py
step router_r2_cache ./scripts/py scripts/107_router_r2.py --stage cache \
  && step router_r2_train ./scripts/py scripts/107_router_r2.py --stage train
step router_budget ./scripts/py scripts/93_budget_allocation.py --routers --tag benchmark_budget_routers
say "TASK2 COMPUTE DONE"
