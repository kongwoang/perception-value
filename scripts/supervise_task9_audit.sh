#!/bin/bash
# Task 9 audit (RESEARCH_LOG.md, audit plan).  One heavy job; fan at 100% while it runs.
cd /home/kongwoang/research/risk-aware-perception
fan () { curl -s -X POST -H 'content-type: application/json' -d "$1" localhost:8765/api/jetson/fan > /dev/null; }
rm -f logs/task9_audit.done
while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done
fan '{"mode":"manual","pwm":255}'
echo "[$(date +%H:%M:%S)] audit start" > logs/task9_audit.log
scripts/py scripts/129_target_swap_audit.py >> logs/task9_audit.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] audit exit $rc" >> logs/task9_audit.log
fan '{"mode":"auto","profile":"quiet"}'
echo "rc=$rc" > logs/task9_audit.done
