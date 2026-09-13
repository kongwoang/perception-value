# Phase 0G, Track A — a controlled independent learned planner

*Pre-registered 2026-09-12 16:45 with two amendments, both recorded before any allocation
number was inspected (RESEARCH_LOG.md). Branch `exp/phase0g-planner-conditionality`.*

## 1. What this track was for

Phase 0F left the decisive comparison unbalanced. Planners A and B are hand-written, and
Planner C is PKL's own released planner scored by displacement from **its own**
ground-truth-conditioned output — so PKL, a divergence between predicted and
ground-truth-conditioned heatmaps, and J_C, the displacement of those same heatmaps' argmax,
share model, weights, forward pass, inputs *and* functional form. Phase 0F §6 said most of
PKL's resulting η = 0.872 was internal consistency. Track A was built to fill the missing
quadrant: a **learned** planner **independent** of PKL, with dataset, BEV representation,
fidelity pair, perception outputs, geometry variant and scene set all held fixed.

## 2. What was frozen before results

Split `configs/phase0g_scene_split.json`, generated from a rule rather than a seed: test = the
85 Phase-0F scenes verbatim; the remaining 765 ordered by (location, name), every 8th within
each location to validation. **train 667 / val 98 / test 85 scenes**, asserted pairwise
disjoint; later capped to 200 training scenes by measured render throughput (§4).

Architecture: four 3×3 stride-2 conv blocks at 32/64/128/256 with GroupNorm and ReLU, global
average pool, MLP 256→256→32, output 16×2 waypoints. Input: the same 5-channel BEV raster the
released PKL planner consumes, built by calling **their own** `samp2ego`, `samp2mapname`,
`get_local_map`, `get_other_objs`, `raster_render`. No PKL weights, layers, distillation or
scores were used in training. Three seeds. Two variants: **D-GT** (clean rasters) and **D-Aug**
(object dropout 0.10, translation jitter σ = 0.5 m, size jitter σ = 0.10, heading jitter σ = 5°,
all stipulated in advance).

Primary cost is error against the **real** future ego trajectory,
`J_D^ADE(mode) = mean_t ‖π_D(raster_mode)_t − y_true_t‖`, deliberately *not* the planner's own
GT-conditioned output — that construction is exactly what made Planner C circular.

Viability gate: validation ADE at least 10% below a constant-velocity predictor, checked before
any cheap/full evaluation. Falsifier **D-F4**: if Planner D cannot beat that baseline, its
allocation result is not interpreted.

## 3. Three errors caught before they could reach a result

**A mirrored coordinate frame.** `tests/test_ego_traj.py` asserts our world→ego transform equals
PKL's own `objects2frame` on random poses. It failed immediately: the first implementation
transposed the rotation twice, sending "forward" to negative x. Training converges perfectly
well on mirrored targets, so without that test the whole phase would have been built on
reversed trajectories.

**A withheld input.** The pre-registered architecture saw only the BEV raster and plateaued at
validation ADE ≈ 5.0 m against 1.467 m for constant velocity, while training MSE fell 7×. Data
was verified first, as the pre-registration requires, and was sound: monotone-forward fraction
1.000, final-waypoint lateral offsets symmetric about zero (mean +0.16 m), and **correlation
0.961 between the 4 s waypoint and 4 s × measured ego speed**. That correlation is also the
diagnosis — ego speed nearly determines the target, and the raster does not contain it; the ego
is a static footprint. Ego velocity (2 numbers) was concatenated to the pooled BEV feature under
rule A7. It cannot contaminate the study because **it is identical under CHEAP and FULL**: it
shifts both costs together and leaves their difference driven only by the raster.

**A future-pose leak.** `ego_velocity` used a central difference spanning t0 − 0.5 s to
t0 + 0.5 s, so it read the pose half a second into the future. That inflated the baseline *and*,
once velocity became an input, fed the network part of the answer. Corrected to a backward
difference, with two tests that fail on any future dependence. The corrected baseline is
**2.002 m, not 1.467 m — the leak had inflated it by 27%**. The correction averages 0.15 m/s but
reaches 17 m/s in the tail, concentrated on exactly the accelerating and braking frames where a
downstream decision is most at stake.

## 4. Amendments forced by measurement, recorded before training

**21% of samples were dropped**: a nuScenes scene is ~20 s and the horizon is 4 s, so the last
eight samples of every scene have no real future. Clamping makes their target a stationary ego.
Planner D's test set is therefore **2,655 of the 3,376 frames**, and Planner C is re-scored on
exactly that subset wherever the two are compared. **Training capped at 200 of 667 scenes**
(6,337 samples) because rendering costs 0.41 s per raster; validation (3,113) and test are
complete.

## 5. Result: D-F4 fires on all six runs

| variant | val ADE (3 seeds) | constant velocity | improvement | pass |
|---|---|---|---|---|
| D-GT | 2.004 ± 0.050 m | 2.002 m | **−0.1%** | **0/3** |
| D-Aug | 1.984 ± 0.015 m | 2.002 m | **+0.9%** | **0/3** |

Restricted to the two map locations the test split actually covers, improvements run from −4.9%
to +0.7%; all fail. Seed spread is 0.015–0.050 m, so this is not seed noise. The gate asked for
10%; the entire BEV scene — road, lanes, dividers, objects — contributes about **1%** beyond the
ego's own velocity.

This is not an implementation failure. It reproduces an established result: on nuScenes
open-loop planning, ego status dominates and perception contributes very little (*Is Ego Status
All You Need for Open-Loop End-to-End Autonomous Driving?*, CVPR 2024; *Rethinking the Open-Loop
Evaluation of End-to-End Autonomous Driving*, 2023). Reproducing it independently raises
confidence in the measurement rather than lowering it.

**Consequently, per D-F4, no Planner D η is used as evidence about planner conditionality.**

## 6. What Planner D does establish

The decision value of perception compute under this planner is not merely small — it is
**absent, and symmetric**:

| model | ADE(GT) | ADE(CHEAP) | ADE(FULL) | mean ΔJ_D | ΔJ_D ≠ 0 | harmful among affected |
|---|---|---|---|---|---|---|
| D-GT ×3 | 2.13–2.21 m | — | — | −0.0006…−0.0002 m | 0.633 | 0.493–0.540 |
| D-Aug ×3 | 2.07–2.18 m | — | — | −0.0001…+0.0004 m | 0.633 | 0.500–0.513 |

Quadrupling detector compute moves this planner's trajectory error by ~0.0003 m, and the sign
is a coin flip. So the finding is about the **setting**, not the model: *nuScenes open-loop
waypoint prediction is a downstream task on which perception compute has no decision value to
allocate.* Any learned planner on this data will behave the same way, which is a warning worth
stating — an adaptive-perception method evaluated on this target would measure nothing.

## 7. The result that does survive: Planner C, de-circularised

The Planner D failure prompted the cheap experiment that should have been run first — apply the
same test to Planner C. PKL's released planner, scored against the real trajectory on the same
2,655 frames (`scripts/74_plannerC_vs_truth.py`, reusing the cached rasters):

| input | ADE vs real trajectory |
|---|---|
| GT raster | 3.450 m |
| CHEAP raster | 3.776 m |
| FULL raster | 3.721 m |
| constant velocity | **2.402 m** (PKL's planner is **43.7% worse**) — viability FAIL |

PKL's own planner is a worse *point* predictor of the ego trajectory than assuming constant
velocity. That is partly expected — it predicts a distribution and has no ego-state input — but
it means Planner C's "decisions" are weak ones.

Scoring the same planner by a target **external** to it changes PKL's allocation performance by
more than half:

| signal | Planner C, self-referenced (Phase 0F) | Planner C, real trajectory (Phase 0G) |
|---|---|---|
| random | +0.102 | +0.067 |
| exact ΔE | +0.317 | +0.254 |
| best single ΔE variant | +0.420 | +0.254 |
| **PKL gain** | **+0.872** [0.82, 0.91] | **+0.429** [0.21, 0.57] |
| **TIP gain** | **+0.826** | **+0.439** [0.21, 0.59] |
| Spearman, PKL vs ΔJ | +0.704 | **+0.201** |
| oracle prize @20% quota | 15.11% (vs 6.76% all-FULL) | 5.18% (vs 1.46% all-FULL) |

So **Phase 0F §6's 0.872 was, as suspected, largely the shared functional form.** The honest
number for PKL on its own planner with an external target is **0.429** — well above random
(0.067) and above perception metrics (0.254), and far below the 0.872 the circular construction
produced. Phase 0F's published numbers stand as published; this is the correction, measured.

The ordering reversal that the planner-conditionality claim rests on survives de-circularisation:

| | Planner A (longitudinal) | Planner C (real trajectory) |
|---|---|---|
| best single ΔE variant | **+0.562** | +0.254 |
| PKL gain | +0.109 | **+0.429** |

A risk-weighted perception error beats PKL 5× on one downstream system and loses to it 1.7× on
another, on the same frames and the same perception transition.

## 8. Sign-varying value, now on four decision systems

| downstream system | frames with ΔJ ≠ 0 | made **worse** by the expensive detector |
|---|---|---|
| Planner A, longitudinal (hand-written) | 295 / 3,376 | **48.5%** |
| Planner B, KITTI (hand-written, rollout) | 1,361 / 8,008 | 37.0% |
| Planner C, real trajectory (PKL's own) | 1,251 / 2,655 | **49.6%** |
| Planner D (learned, independent) | 1,680 / 2,655 | 49.3–54.0% |

This is the project's most robust finding and nothing in Phase 0G weakens it.

## 9. What Track A cannot conclude

The pre-registered verdict options assumed Planner D would be **valid**. It is not, so the
controlled C-vs-D conditionality test cannot be run at all: comparing rankings against a
decision value that is identically zero is meaningless, and D-F1/D-F2/D-F3 are all unassessable
rather than passed or failed. Recording that as a departure rather than relabelling it:

- **D-F1** (PKL transfers to an independent learned planner): unassessable.
- **D-F2** (planner identity does not change the ranking): unassessable.
- **D-F3** (sign variation disappears): does not fire — 49.3–54.0% harmful — but under D-F4 this
  is reported as a property of the setting, not as support for the claim.
- **D-F4**: **fired**, 0/6 runs.

The conditionality claim therefore rests on Planner A versus the de-circularised Planner C, both
on nuScenes, with one planner hand-written by us and the other a published planner that fails
the same viability test. **Track B is now the load-bearing evidence, not a nice-to-have**: a
closed-loop nuPlan planner brakes for detected objects, so unlike open-loop waypoint regression
it is a downstream system that genuinely depends on perception.

---

> **Superseded numbers.** The 2026-09-13 code review found seven defects, three of which
> changed numbers in this document, and everything downstream of the detection submissions was
> recomputed. The method and reasoning here stand; for every figure see
> [`iclr_corrected_results.md`](iclr_corrected_results.md), which takes precedence.
