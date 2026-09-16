# Statistics hardening: raw gain, single-unit influence, trivial baselines

**What this is.** Three stress tests of the held-out conclusions, plus a second harm reading. Do the results
survive without the prize denominator, without one dominant test unit, and against predictors that use one
number?

**Provenance.**
* Pre-registration: `RESEARCH_LOG.md`, Task 12 (commit `e627e60`), committed before the script ran.
* Code: `scripts/125_statistics_hardening.py`. Output: `results/final/statistics_hardening.csv` (2,552 rows).
* No official result file was changed. CPU only, cached scores.

**Check.** Recomputing the official nDG through this script matches the released tables to 3 decimals in
**980 of 980** comparisons.

## 1. Paired intervals on raw gain, not on nDG

For each cell, signal and quota: the realised decision value minus random's, in loss units, with a 1,000-draw
paired unit bootstrap. Every draw is kept, because the quantity is not divided by the prize; the official 25%
prize filter is reported beside it.

**The filter does not make results more conservative. It makes them look better.**

| quota | rows | beat random, every draw | beat random, 25% filter | verdicts changed |
|---|---|---|---|---|
| 10% | 294 | 18 | 28 | 10 |
| 20% | 294 | 18 | 24 | 6 |
| 30% | 294 | 28 | 34 | 6 |
| 50% | 294 | 36 | 55 | 19 |

Every changed verdict goes the same way: dropping draws turns a non-significant row into a significant one. The
rows affected are exactly those whose prize sits in one unit, for example KITTI oracle traj (85 draws dropped) and
the two PDM-Closed cells (319 draws dropped).

**At the 20% quota, of the 12 official wins among the 118 deployable cell × signal pairs:**

| outcome | count | rows |
|---|---|---|
| survive on raw gain, every draw kept | 8 | KITTI oracle brake R1_gbm_clf; KITTI oracle traj gate_gbm, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf; KITTI mono brake R1_mlp_reg, R1_gbm_reg; nuPlan PDM-Closed scalar_J gate_ridge |
| do not survive | 4 | nuPlan PDM-Closed safety gate_ridge and gate_gbm; PDM-Closed scalar_J gate_gbm; nuPlan IDM scalar_J R1_mlp_clf |
| new, raw gain only | 1 | — |

The three PDM-Closed gate rows that fail have a lower bound of exactly 0.000: in the draws that do not contain the
dominant log, the difference to random is exactly zero, so the interval touches zero and the win disappears.

**Sizes of the surviving wins, as a share of the all-cheap loss:**

| cell | signal | Δ vs random | as % of all-cheap loss |
|---|---|---|---|
| KITTI oracle traj | gate_gbm | +28 to +736 | 19.5% |
| KITTI oracle traj | R1_gbm_clf | +29 to +550 | 14.9% |
| KITTI oracle traj | R1_gbm_reg | +27 to +532 | 14.9% |
| KITTI oracle traj | R1_mlp_clf | +16 to +433 | 13.8% |
| KITTI oracle brake | R1_gbm_clf | +89 to +1099 | 8.8% |
| KITTI mono brake | R1_gbm_reg | +12 to +896 | 3.9% |
| KITTI mono brake | R1_mlp_reg | +8 to +689 | 3.1% |
| nuPlan PDM-Closed scalar_J | gate_ridge | +0.16 to +352 | 30.5% |

## 2. Leave-one-test-unit-out influence at 20%

Each test unit is dropped in turn, with models and thresholds unchanged.

* Of the 36 official win rows, **2 lose more than half their nDG** when a single unit is removed, and **9 have a
  leave-one-out range wider than 0.2**.
* The largest single-unit swings are on nuPlan: PDM-Closed safety R1_gbm_reg goes from +0.072 to −0.922, IDM
  scalar_J R1_gbm_clf from +0.059 to +0.999, and IDM scalar_J R1_mlp_clf from 0.938 to 0.318, all driven by one log.
* On core cells the worst case is a trivial baseline: ego speed on KITTI mono traj falls from +0.554 to −0.260 when
  sequence 0008 is removed.

**The two PDM-Closed cells where the gates reach 0.99.**

| cell | signal | nDG full | nDG min–max over leave-one-out | gain full | gain without log `2021.05.12.23.36.44_veh-35_00152_00504` |
|---|---|---|---|---|---|
| PDM-Closed safety | gate_ridge | 0.992 | 0.992–1.000 | 162.9 | **12.0** |
| PDM-Closed safety | gate_gbm | 0.992 | 0.992–1.000 | 162.9 | **12.0** |
| PDM-Closed scalar_J | gate_ridge | 0.988 | 0.978–0.990 | 171.3 | **12.9** |
| PDM-Closed scalar_J | gate_gbm | 0.989 | 0.986–0.990 | 171.6 | **13.0** |

**This is the important line of the table.** The gates' nDG is unmoved by removing that log, because the oracle
prize collapses with it: the ratio is stable while 93% of the decision value disappears. A normalised score cannot
show this; the raw gain can. The same log is the most influential unit for every R1 router in these cells too.

## 3. Trivial baselines

Ego speed alone, number of cheap detections alone, largest cheap detection area alone, and cheap-side risk alone.
The risk baseline is the column the official `criticality_cheap` signal already uses, so it reproduces that row by
construction; it is reported for completeness.

**No trivial baseline beats random on raw gain at 20%: 0 of 38 rows.** Across all quotas only three rows do
(ego speed at 30% and 50%, cheap-side risk at 30%).

**But on nDG alone, ego speed outranks every learned allocator on all four KITTI cells:**

| cell | best trivial | its nDG | best learned | its nDG | trivial ≥ learned |
|---|---|---|---|---|---|
| KITTI oracle traj | ego speed | 0.784 | gate_gbm | 0.575 | yes |
| KITTI oracle brake | ego speed | 0.563 | R1_gbm_clf | 0.349 | yes |
| KITTI mono traj | ego speed | 0.554 | R1_mlp_clf | 0.337 | yes |
| KITTI mono brake | ego speed | 0.402 | R1_gbm_clf | 0.220 | yes |
| nuPlan PDM-Closed safety | detection count | 0.285 | gate_ridge | 0.992 | no |
| nuPlan PDM-Closed scalar_J | detection count | 0.287 | gate_gbm | 0.989 | no |
| nuPlan IDM scalar_J | detection count | −0.068 | R1_mlp_clf | 0.938 | no |
| nuScenes (6 cells) | ego speed or area | −0.023 to 0.169 | learned | 0.046 to 0.274 | no |

* **PDM-Closed and IDM are where the learned allocators are clearly ahead of the trivial ones**, and both rest on
  one log (§2).
* On KITTI the ranking favours a single feature, but neither the trivial baseline nor the learned allocator beats
  random once the prize denominator is removed, so this is a comparison between two statistically indistinguishable
  rows.

## 4. Harm as a share of all inputs

| cell | inputs | affected | harmed | harmed / affected | harmed / all inputs |
|---|---|---|---|---|---|
| KITTI mono brake | 3,536 | 1,508 | 741 | 49.1% | 21.0% |
| KITTI mono traj | 3,536 | 478 | 192 | 40.2% | 5.4% |
| KITTI oracle brake | 3,536 | 888 | 407 | 45.8% | 11.5% |
| KITTI oracle traj | 3,536 | 250 | 43 | 17.2% | 1.2% |
| nuPlan IDM safety | 384 | 4 | 1 | 25.0% | 0.3% |
| nuPlan IDM scalar_J | 384 | 10 | 1 | 10.0% | 0.3% |
| nuPlan PDM-Closed safety | 384 | 27 | 9 | 33.3% | 2.3% |
| nuPlan PDM-Closed scalar_J | 384 | 48 | 20 | 41.7% | 5.2% |
| nuScenes mono brake | 955 | 120 | 57 | 47.5% | 6.0% |
| nuScenes mono plan_ade | 752 | 422 | 225 | 53.3% | 29.9% |
| nuScenes mono plan_fde | 752 | 178 | 88 | 49.4% | 11.7% |
| nuScenes oracle brake | 955 | 62 | 29 | 46.8% | 3.0% |
| nuScenes oracle plan_ade | 752 | 308 | 154 | 50.0% | 20.5% |
| nuScenes oracle plan_fde | 752 | 120 | 51 | 42.5% | 6.8% |

Mean over cells: 39.4% of affected inputs are harmed, which is 8.9% of all inputs. Both numbers belong in the
paper: the first says how often escalation backfires when it changes anything, the second says how often it
backfires at all. On the whole split the two readings are 40.9% and 8.7%.

## 5. What this changes for the paper

1. **Report the raw-gain interval next to nDG.** Eight of the twelve official 20% wins survive it; the three
   PDM-Closed gate wins and the IDM router win do not.
2. **Drop the 25% prize filter, or report both.** It never removes a win; it adds up to 19 of them, precisely in
   the cells whose prize lives in one unit.
3. **State the single-log dependence of the nuPlan claims.** The gates keep nDG 0.99 without their dominant log
   while losing 93% of the gain.
4. **Report both harm shares**, and give the trivial baselines their own row: on KITTI a single feature ranks
   better than every learned allocator, even though neither clears random once the denominator is gone.
