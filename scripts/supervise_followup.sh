#!/bin/bash
# After the final chain: regenerate the transfer artefacts with the 1e-9 tie tolerance so every
# reported pair count matches, then commit and push. Waits for the final chain's completion line;
# its own messages deliberately never contain that line, so the wait cannot match itself.
cd /home/kongwoang/research/risk-aware-perception
say () { echo "[$(date +%H:%M:%S)] followup: $*" | tee -a logs/supervise.log; }
gate () { while [ -n "$(./scripts/busy.sh)" ]; do sleep 30; done; sync; sleep 10; }
say "waiting for the final analysis chain"
until grep -q "^\[[0-9:]*\] FINAL COMPUTE DONE" logs/supervise.log; do sleep 60; done
CM=$(ls -d results/raw/*core_matrix_postreview 2>/dev/null | tail -1)
[ -n "$CM" ] || CM=results/raw/20260912_071140_core_matrix
gate; say "rerun B7/B8 with tie tolerance"
./scripts/py scripts/83_external_transfer.py --nboot 400 >> logs/metrics_all.log 2>&1 && say "B7/B8 regenerated" || say "B7/B8 rerun failed"
for v in oracle mono; do
  sfx=""; [ "$v" = mono ] && sfx="_mono"
  gate; say "rerun tie-robust cross-target $v"
  ./scripts/py scripts/75_cross_target_measurement.py \
      --brake_table "$CM/nuScenes__YOLOv8s__ns_cheap_320tons_full_640__$v.pkl" \
      --plan_csv "data/cache/planner_d/planC_vs_truth$sfx.csv" --label $v --nboot 400 \
      --tag cross_target_tierobust_tol_$v >> logs/metrics_all.log 2>&1 && say "cross-target $v regenerated" || say "cross-target $v rerun failed"
done
gate; say "committing and pushing"
git add -f results/final/phase0g_external_planner_summary.csv results/final/phase0g_external_planner_transfer.csv \
    results/final/phase0g_external_planner_stats.json results/final/phase0g_cross_target_oracle.csv \
    results/final/phase0g_cross_target_mono.csv results/final/phase0g_deployable_gate.csv \
    results/final/phase0f_planning_metric_eta.csv 2>/dev/null
git commit -q -m "Regenerate transfer artefacts with the 1e-9 tie tolerance; FDE and deployable-gate results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013mc5wwW2wUCLR2Er6CpgEX" && say "committed"
git push origin exp/phase0g-planner-conditionality >> logs/metrics_all.log 2>&1 && say "pushed" || say "push failed"
say "all follow-up steps finished"
