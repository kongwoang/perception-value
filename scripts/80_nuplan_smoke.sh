#!/bin/bash
# Phase 0G Track B, step B2: reproduce ONE untouched official simulation for each external
# planner before anything is modified.  If this does not run, Track B stops here.
#
# Nothing below is our code: the entry point, configs and planners are the released ones, and
# the only arguments that differ from the tuPlan Garage README are the scenario builder (mini
# instead of the full split, which we do not have) and a small scenario limit for the smoke run.
set -e
export NUPLAN_DEVKIT_ROOT=/home/kongwoang/research/risk-aware-perception/third_party/nuplan_devkit
export NUPLAN_DATA_ROOT=/home/kongwoang/datasets/nuplan
export NUPLAN_MAPS_ROOT=/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/home/kongwoang/research/risk-aware-perception/data/cache/nuplan_exp
export NUPLAN_SIMULATION_ALLOW_ANY_BUILDER=1
mkdir -p "$NUPLAN_EXP_ROOT"
PY="/home/kongwoang/research/risk-aware-perception/scripts/pynuplan"
SEARCH='[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]'
LIMIT=${LIMIT:-4}

run () {  # planner_name extra_searchpath
  echo "=============== $1 ==============="
  $PY "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
    +simulation=closed_loop_nonreactive_agents \
    planner="$1" \
    scenario_builder=nuplan_mini \
    scenario_filter=one_continuous_log \
    scenario_filter.limit_total_scenarios=$LIMIT \
    worker=sequential \
    hydra.searchpath="$SEARCH" \
    experiment_name="smoke_$1"
}

run idm_planner
run pdm_closed_planner
echo "NUPLAN SMOKE DONE"
