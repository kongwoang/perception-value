#!/bin/bash
# Task 9 (target swap), pre-registered in RESEARCH_LOG.md.  One heavy job; fan at 100% while it runs.
cd /home/kongwoang/research/risk-aware-perception
fan () { curl -s -X POST -H 'content-type: application/json' -d "$1" localhost:8765/api/jetson/fan > /dev/null; }
rm -f logs/task9.done
while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done
fan '{"mode":"manual","pwm":255}'
echo "[$(date +%H:%M:%S)] task9 start" > logs/task9_target_swap.log
scripts/py scripts/122_target_swap.py >> logs/task9_target_swap.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] task9 exit $rc" >> logs/task9_target_swap.log
fan '{"mode":"auto","profile":"quiet"}'
echo "rc=$rc" > logs/task9.done
