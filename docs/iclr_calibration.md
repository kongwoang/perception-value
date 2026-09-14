# Does sign-varying decision value survive per-mode operating points?

Pre-registration: `RESEARCH_LOG.md`, 2026-09-14 "Task 1 pre-registration" (commit `bb1a270`, before any
run), with two documented corrections before any scheme was scored (10:40 Planner B preset; 11:10 q_plan
split). Every table at every scheme, split and sweep point: `docs/calibration_tables.md`, generated from
`results/final/calibration_*.csv` by `scripts/109_calibration_markdown.py`.

## Verdict: **survives**

The reviewer concern was that harm comes from extra false positives when FULL and CHEAP share one
threshold (0.25). Each fidelity now gets its own operating point, chosen on train ∪ val only:

* S1: FULL emits as many boxes as CHEAP.
* S2: each mode at its F1 optimum.
* S3: FULL at least as precise as CHEAP.

Under **each** of S1, S2 and S3, harm rate ≥ 20% and ρ ≥ 0.20 hold in **4 of 4 nuScenes cells and all
6 KITTI moderate-gap cells**. So the pre-registered "survives" condition is met, including its literal
all-six reading. The cells under the collapse line are 2 of 14 under S1 and 3 of 14 under S2 and S3. All
of them are KITTI Y8 320→640 cells whose ρ was already below 0.10 at the shared threshold.

## What was verified before anything was scored

| check | result |
|---|---|
| 1 · S0 reproduces the recorded tables (braking, lateral, 9 perception gains, detection counts, Planner B, Planner C ADE/FDE) | exact on every frame of all 7 pair × geometry combinations; Planner C max \|Δ\| 7e-15 |
| 2 · direct asymmetric runs (0.15, 0.45) and (0.45, 0.15) equal the per-mode composition | exact, nuScenes mono and KITTI 384→640 |
| 3 · 0.10 submissions filtered at 0.25 equal the recorded submissions, box for box | 4 / 4 |
| 4 · rasters rebuilt at 0.25 equal the cached CHEAP/FULL rasters | 0 frames differ, both geometries, both modes |

Check 1 first failed on Planner B, because the runner used the wrong preset. Braking and perception
columns passed on that first run. The failure was fixed and logged before any scheme was scored.

## The operating points actually remove the false-positive explanation

Braking cell, all units. Precision and recall are pooled against GT (IoU 0.5).

| cell · scheme | CHEAP thr · boxes/frame · P · R | FULL thr · boxes/frame · P · R | harm rate | ρ |
|---|---|---|---|---|
| nuScenes mono · S0 | 0.25 · 2.56 · 0.654 · 0.344 | 0.25 · 4.18 · 0.579 · 0.499 | 45.5% | 0.54 |
| nuScenes mono · **S1** | 0.25 · 2.56 · 0.654 · 0.344 | **0.498 · 2.59 · 0.764** · 0.410 | **38.3%** [33, 44] | **0.40** [0.28, 0.59] |
| nuScenes mono · S3 | 0.25 · 2.56 · 0.654 · 0.344 | 0.340 · 3.44 · 0.660 · 0.470 | 41.9% [35, 48] | 0.45 [0.30, 0.67] |
| KITTI Y8 384→640 · S0 | 0.25 · 4.50 · 0.701 · 0.566 | 0.25 · 6.13 · 0.654 · 0.717 | 44.6% | 0.55 |
| KITTI Y8 384→640 · **S1** | 0.25 · 4.50 · 0.701 · 0.566 | **0.434 · 4.45 · 0.793** · 0.634 | **38.7%** [32, 45] | **0.49** [0.21, 0.85] |

Under S1, FULL emits no more boxes than CHEAP and is **more** precise: 0.764 vs 0.654 on nuScenes, and
0.793 vs 0.701 on KITTI. Yet more than a third of affected frames are still harmed, and harm still
cancels 40–49% of the benefit. The same pattern holds in every other nuScenes and moderate-gap cell.

## Harm rate and ρ, all units (95% unit-bootstrap CI in `calibration_tables.md`)

| cell | S0 harm · ρ | S1 | S2 | S3 |
|---|---|---|---|---|
| nuScenes oracle · q_brake | 48.5% · 0.60 | 43.8% · 0.59 | 45.8% · 0.51 | 45.8% · 0.53 |
| nuScenes oracle · q_plan | 51.4% · 0.82 | 51.5% · 0.89 | 51.1% · 0.83 | 51.7% · 0.86 |
| nuScenes mono · q_brake | 45.5% · 0.54 | 38.3% · 0.40 | 40.5% · 0.41 | 41.9% · 0.45 |
| nuScenes mono · q_plan | 51.6% · 0.72 | 51.5% · 0.76 | 51.4% · 0.70 | 50.9% · 0.67 |
| KITTI mono Y8 384→640 · q_traj / q_brake | 45.4% · 0.21 / 44.6% · 0.55 | 42.4% · 0.26 / 38.7% · 0.49 | 43.4% · 0.21 / 41.9% · 0.51 | 44.4% · 0.21 / 43.8% · 0.54 |
| KITTI mono Y8 512→640 · q_traj / q_brake | 40.7% · 0.44 / 45.7% · 0.68 | 38.8% · 0.41 / 43.9% · 0.67 | 39.2% · 0.32 / 43.6% · 0.62 | 40.0% · 0.41 / 45.4% · 0.68 |
| KITTI mono RT-DETR 480→640 · q_traj / q_brake | 41.4% · 0.55 / 44.5% · 0.80 | 41.1% · 0.55 / 44.4% · 0.84 | 43.3% · 0.43 / 43.4% · 0.83 | 41.4% · 0.55 / 44.5% · 0.80 |
| KITTI mono Y8 320→640 · q_traj / q_brake | 35.2% · 0.06 / 40.7% · 0.19 | 32.2% · 0.11 / 29.4% · 0.14 | 33.3% · 0.08 / 37.3% · 0.19 | 34.1% · 0.06 / 39.2% · 0.18 |
| KITTI oracle Y8 320→640 · q_traj / q_brake | 9.5% · 0.00 / 34.1% · 0.09 | 13.0% · 0.07 / 18.6% · 0.06 | 8.8% · 0.00 / 27.2% · 0.07 | 8.7% · 0.00 / 31.4% · 0.08 |

The test-split values are close, with much wider intervals. Under S1–S3, nuScenes harm rates on test are
39.6–53.2%.

**The sensitivity sweep** covers all 25 (CHEAP, FULL) pairs in {0.15, …, 0.55}², all units. It shows
harm rate ≥ 20% and ρ ≥ 0.20 at the following number of pairs per cell:

| cells | pairs passing (of 25) |
|---|---|
| all four nuScenes cells | 25 |
| RT-DETR q_brake | 25 |
| RT-DETR q_traj | 24 |
| Y8 512→640 q_brake / q_traj | 25 / 17 |
| Y8 384→640 q_brake / q_traj | 20 / 11 |
| Y8 320→640 mono q_brake / q_traj | 6 / 1 |
| KITTI oracle cells | 0 |

The lowest nuScenes harm rate anywhere in the sweep is 31.7%. The KITTI oracle-geometry cells were
harmless at the shared threshold and stay so. That is a property of the cell, not of the threshold.

## Perception gains still do not predict the sign (nuScenes, all units)

Gains considered: exact FN, FN+FP, combined E5 and E_risk, under S1–S3.

**Sign disagreement** (among frames where both the gain and V are nonzero):
* braking: 29–48%;
* planner: 49–54%.

**Spearman correlation with V:**
* braking: between −0.02 and +0.10;
* planner: between −0.02 and +0.01.

**P(V<0 | gain>0) among affected frames:**
* braking: 30–46%;
* planner: 50–53%.

These are the same magnitudes as at S0. Better perception by any of the four measures is harmful for the
decision about as often as a coin flip on the planner, and a third to a half of the time on braking.

**Braking vs PKL's planner, frames where both values are nonzero and their signs are opposite:**
* oracle geometry: S0 80 / 153, S1 58 / 120, S2 57 / 120, S3 71 / 135;
* mono geometry: S0 156 / 309, S1 129 / 261, S2 137 / 273, S3 149 / 285.

About half, under every scheme.

## Downstream-tuned thresholds (S4, not part of the reading)

S4 picks, for each mode, the threshold minimising that mode's own downstream loss on train ∪ val.

* **Boundary-limited:** on KITTI, the CHEAP optimum is the grid floor 0.10 in 7 cells, and the FULL
  optimum is 0.10 for RT-DETR q_traj. Caches stop at 0.10, so these optima are limited by the boundary.
  Locating them would need the engines re-run at conf ≥ 0.01, about 35–45 min.
* **Harm under S4:** 28.0–54.5% in every nuScenes and moderate-gap cell.
* **ρ under S4:** 0.29–0.83 in those cells, with the KITTI oracle cells again the exception.

Tuning each fidelity to its own downstream loss does not remove the sign variation either.

## What this changes for the paper

1. **The false-positive objection can be answered with a number.** Count-matched FULL is more precise
   than CHEAP and emits no more boxes. Yet nuScenes braking still has 38% harmed and ρ 0.40, and every
   nuScenes and moderate-gap cell stays above 20% / 0.20.
2. **Shared-threshold harm is not the whole effect, but part of it moves.** Under S1, harm on nuScenes
   braking falls by 5–7 points and ρ by up to 0.14. The planner cells do not move at all. State this as
   "reduced, not removed".
3. **The low-gap and oracle-geometry KITTI cells are harmless under every scheme.** The claim should be
   scoped to cells with a meaningful fidelity gap or monocular geometry. That is where the benchmark
   already puts it.

## Caveats

* Thresholds are chosen on train ∪ val, and the reading uses all-unit statistics, which include those
  frames. The choices use only precision/recall (S1–S3) or one mode's own loss (S4), never V.
  Test-split statistics are reported next to every number.
* Thresholds below 0.10 do not exist in the caches. S2 never selected the floor. S4 did, as reported
  above.
* nuPlan was not part of this test.
* The sign-agreement definitions are stated in the pre-registration, because the paper's own table
  source is not in the repository. Check them against it.
