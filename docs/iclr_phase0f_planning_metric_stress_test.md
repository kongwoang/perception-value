# Phase 0F, Stage 1 — do published planning-aware perception metrics already solve compute allocation?

*Pre-registered 2026-09-12 08:20 (RESEARCH_LOG.md). This document is the Stage-1 stop
report: it is written before any closed-loop or nuPlan/CARLA work, and it decides whether
that work happens at all.*

---

## 1. The question, stated narrowly

Phase 0E showed that on our data the ranking induced by *perception* improvement (ΔE) is a
poor proxy for the ranking induced by *decision* improvement (ΔJ), so allocating a scarce
compute budget by ΔE leaves most of the achievable decision-cost reduction on the table.
That comparison was against **task-agnostic** perception metrics, which is no longer the
right bar: planning-aware perception evaluation is an established literature.

So the bar for this project is not "does perception accuracy differ from planning quality"
— that is prior knowledge — but:

> When choosing **which inputs** receive additional neural perception compute, does an
> existing planning-aware perception score already rank inputs by marginal downstream
> decision benefit?

The distinction under test is between **relevance of an error** and the **marginal value of
correcting it with one particular computation**. PKL and TIP are built to answer the first.
Nothing in their design commits them to the second, but they are the closest published
answer and were designed by people thinking carefully about the same gap, so they get the
first and strongest shot at it.

## 2. Pre-registered decision rule

Fixed before any number was computed:

| outcome | rule |
|---|---|
| **NO-GO / STOP** | a published planning-aware metric reaches η ≥ 0.8 at a 20% quota, consistently across downstream tasks |
| **WEAK GO** | it lands in roughly 0.6–0.8 |
| **STRONG EMPIRICAL GAP** | it beats standard ΔE metrics but stays < 0.6 under held-out scenes and moderate fidelity |

η is the project's standing figure of merit: the share of the decision-value oracle's
achievable cost reduction that a signal captures at a given compute quota. η = 1 is the
oracle that ranks frames by true ΔJ; η = 0 is running everything cheap.

**Honest prior expectation, recorded before running**: PKL and TIP beat standard ΔE and
still fall short of the oracle. But Phase 0E already had a rich perception-only oracle
reaching 0.70–0.83 in the flagship KITTI cell, so a PKL/TIP score above 0.8 on nuScenes was
entirely plausible. Had that happened, the project would have stopped here.

## 3. What was run

### 3.1 Prior work, used as published

Both metrics come from the authors' own repositories at pinned commits, unmodified, with
the authors' pretrained planner. Full detail — repository URLs, commits, dependency
versions, the dead-weights problem and its resolution, and both deviations forced by this
board — is in [`third_party/PROVENANCE.md`](../third_party/PROVENANCE.md). In summary:

- **PKL** — Philion, Kar & Fidler, *Learning to Evaluate Perception Models Using
  Planner-Centric Metrics*, CVPR 2020. `nv-tlabs/planning-centric-metrics` @
  `f6865f2`, via `calculate_pkl`.
- **TIP** — *Transcendental Idealism of Planner*, ICML 2023. `qcraftai/tip` @ `e52fd48`,
  via `calculate_tip`.
- Scoring definitions are untouched. The one substantive deviation is that
  `create_splits_scenes` is patched so the "val" split is exactly our 85 scenes; without
  it `load_gt` builds ground truth for all ~28k trainval samples and exhausts the board's
  15 GB of unified memory. Box construction inside `load_gt` runs unmodified.
- PKL's own Google Drive weights no longer resolve; the checkpoint is taken from TIP, which
  vendors it and states it is adapted from PKL. Verified compatible: it loads into PKL's
  `compile_model(cin=5, cout=16, with_skip=True)` with `strict=True`, 149 keys, zero
  missing, zero unexpected.

### 3.2 Getting our detections into their interface

Both metrics consume a **nuScenes 3D detection submission**. Our perception is 2D monocular
YOLOv8s on CAM_FRONT, so `src/rap/nusc_submission.py` converts detections into that format
in two variants:

- **`mono`** — a fully monocular lift: ground-plane back-projection for translation, class
  size priors, yaw assumed equal to ego heading.
- **`oracle`** — detections matching ground truth at IoU ≥ 0.5 inherit that object's exact
  3D translation, size and rotation; unmatched detections keep the monocular lift.

`oracle` is the **primary** variant and `mono` the robustness check, and this is deliberate
and in the metrics' favour: the monocular lift injects depth and yaw error that has nothing
to do with the 320-vs-640 fidelity decision under study, and that noise would depress PKL
and TIP for a reason unrelated to the hypothesis. Under `oracle`, everything PKL and TIP see
about a *detected* object is exactly right; what still differs between the cheap and the
expensive mode is which objects were detected at all, plus their class and 2D extent. The
decision table is built from the same geometry variant, so the metrics and the planner see
the same boxes.

### 3.3 The allocation signal

PKL and TIP are divergences from the planner's ground-truth output: **higher is worse**.
TIP's `calculate_tip` states it converts its scores "to comply with PKL's definition", so
the direction is shared. The per-frame allocation signal is therefore

```
G_metric(frame) = score_cheap(frame) − score_full(frame)
```

positive when the expensive mode helps that frame — the same sign as the project's own ΔE
and ΔJ gains. No other transformation is applied.

### 3.4 Data and comparison set

- nuScenes trainval01: **85 scenes, 3,376 CAM_FRONT keyframes**; fidelity pair
  `ns_cheap_320 → ns_full_640`, YOLOv8s.
- Downstream tasks: **longitudinal** braking (KEEP / DECELERATE / HARD_BRAKE) and
  **lateral** corridor avoidance (KEEP / LEFT_AVOID / RIGHT_AVOID / BRAKE).
- Quotas 10 / 20 / 30 / 50%, pooled top-quota selection.
- Signals compared at identical quota: random (16 seeds), visual uncertainty, downstream
  criticality, exact perception gain ΔE, best single ΔE variant of the eight, the
  multi-metric ΔE oracle (a GBM on perception-change features, fit leave-one-scene-out —
  the strongest perception-only baseline available), **PKL gain**, **TIP gain**, and the
  decision oracle ΔJ at η = 1 by construction.
- η is reported with a **95% bootstrap interval resampling scenes**, not frames: frames
  inside a scene are strongly correlated, and a single 595-frame slice moved η by 0.4
  between adjacent quotas, so a point estimate alone cannot support a 0.6/0.8 verdict band.
- Ranking diagnostics against ΔJ: Spearman ρ, Kendall τ, pairwise inversion rate, and
  top-10% / top-20% frame-set overlap.

