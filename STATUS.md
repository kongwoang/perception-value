# Session status (resume notes)

Written 2026-09-11 ~15:35. Update or delete freely.

## Running in the background, detached (survives session death)
- `scripts/06_sensitivity.py`, log at `logs/sensitivity.log`, writes
  `results/raw/20260911_145752_sensitivity/sensitivity.csv` **incrementally**, one row
  per config as it completes. 28 configs, ~10 min each, started 15:00.
  Nothing is lost if it is interrupted — completed rows are already on disk.

## Done
- KITTI tracking, 21 sequences, 8008 frames, in `~/datasets/kitti_tracking/`.
- FP16 TensorRT engines for 5 modes in `~/models/yolo/engines/`.
- Detection cache for cheap_320 / cheap_384 / full_640 / full_960: `data/cache/det/`.
- Profile: `results/raw/20260911_143838_profile`
- Mechanism: `results/raw/20260911_143838_mechanism`
- Main analysis: `results/raw/20260911_150044_analysis`
- Figures: `results/figures/` (fig0..fig7b)

## Left to do
1. Wait for sensitivity sweep, summarise into the findings doc.
2. Write `docs/cvpr_phase0_findings.md` (answers A-G + GO/NO-GO).
3. Optional: robustness pairs cheap_384/full_640 and cheap_320/full_960
   (detection cache already exists for both, so only 03 + 04 need re-running).

## Headline numbers so far
- Compute gap cheap_320 -> full_640: 10.1 -> 16.0 ms e2e, 3.2 -> 8.0 ms GPU inference.
- 36% of frames gain exactly zero from FULL; top 10% carry 60% of the gain.
- Out-of-fold Spearman with Value_task: uncertainty 0.254, all-visual 0.207, all 0.293.
- Adding a total-stakes scalar to all-visual: null (p = 0.55). Adding full criticality
  structure on top of that: +0.123, 17/21 sequences, p = 0.002.
- Budget at 20% quota: uncertainty eta 0.435, uncertainty+criticality 0.553, 19/21 seqs.
- Same selections on the standard detection metric: uncertainty 0.338 *beats*
  uncertainty+criticality 0.280 — the routing is task-specific, not generically better.
