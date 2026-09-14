#!/bin/bash
# Task 5 chain, strictly one heavy job at a time.  Part 1 stops after the reference/identity checks so the
# overlays and projection checks are reviewed before any CHEAP/FULL branch is scored; part 2 scores.
#   scripts/supervise_task5.sh part1     (waits for the fetch, then detect -> project -> checks -> ref idm/pdm)
#   scripts/supervise_task5.sh part2     (branches idm/pdm -> cells)
cd /home/kongwoang/research/risk-aware-perception
LOG=logs/supervise_task5.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
fan () { curl -s -X POST -H 'content-type: application/json' -d "$1" localhost:8765/api/jetson/fan > /dev/null; }
gate () { while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done; sync; sleep 5; }
step () { local l="$1"; shift; gate; say "$l start"; "$@" >> logs/task5_$l.log 2>&1; local rc=$?; say "$l exit $rc"; return $rc; }
quit () { fan '{"mode":"auto","profile":"quiet"}'; say "$1"; exit 1; }

if [ "$1" = part1 ]; then
  say "part1: waiting for the fetch"
  while pgrep -f "112_nuplan_fetch_cam_f0.py" > /dev/null; do sleep 30; done
  tail -3 logs/task5_fetch.log | grep -q "^  done" || quit "part1: fetch did not finish cleanly, STOP"
  fan '{"mode":"manual","pwm":255}'
  step detect scripts/py scripts/113_nuplan_detect.py || quit "part1: detection FAILED"
  step project scripts/pynuplan scripts/114_nuplan_project_match.py || quit "part1: projection FAILED"
  step checks scripts/py scripts/116_nuplan_real_cells.py --stage checks || quit "part1: checks FAILED"
  step ref_idm scripts/pynuplan scripts/115_nuplan_real_counterfactual.py --planner idm --phase ref \
    || quit "part1: IDM reference/identity check FAILED, STOP"
  step ref_pdm scripts/pynuplan scripts/115_nuplan_real_counterfactual.py --planner pdm_closed --phase ref \
    || quit "part1: PDM-Closed reference/identity check FAILED, STOP"
  step checks2 scripts/py scripts/116_nuplan_real_cells.py --stage checks || quit "part1: checks FAILED"
  fan '{"mode":"auto","profile":"quiet"}'
  say "TASK5 PART1 DONE"
elif [ "$1" = part2 ]; then
  fan '{"mode":"manual","pwm":255}'
  step br_idm scripts/pynuplan scripts/115_nuplan_real_counterfactual.py --planner idm --phase branches \
    || quit "part2: IDM branches FAILED"
  step br_pdm scripts/pynuplan scripts/115_nuplan_real_counterfactual.py --planner pdm_closed --phase branches \
    || quit "part2: PDM-Closed branches FAILED"
  step cells scripts/py scripts/116_nuplan_real_cells.py --stage cells || quit "part2: cells FAILED"
  fan '{"mode":"auto","profile":"quiet"}'
  say "TASK5 PART2 DONE"
fi
