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

### 3.5 Wiring validation — run before believing any low η

A low η is only a finding if the metrics are connected the way their papers define them.
Two checks, on the first slice of 15 scenes / 595 frames, before looking at any η:

| check | Spearman ρ | expected |
|---|---|---|
| `pkl_cheap` vs cheap-mode false negatives | **+0.288** | more missed objects → worse score |
| `pkl_full` vs full-mode false negatives | **+0.231** | same, independently |
| `pkl_cheap` vs number of ground-truth objects | **+0.284** | more objects → more to get wrong |
| **`G_PKL` vs ΔE (exact perception gain)** | **+0.065** | — |
| `G_PKL` vs downstream criticality | +0.159 | — |

The first three confirm both the direction ("higher is worse") and the frame-level join: the
PKL **level** tracks that frame's error count. The fourth is the first substantive result
rather than a check — the **difference** between the cheap and the expensive mode carries
almost none of that structure. This is the shape the rest of the analysis has to test at
scale: PKL is informative about *how much a frame's errors matter*, and close to
uninformative about *how much this particular extra computation reduces them*.

### 3.6 How much decision signal exists at all

On the same slice, ΔJ is non-zero on **32 / 595 frames (5.4%)** longitudinally, with 17
action changes, and on **0 / 595** laterally — which is why the lateral η is undefined on a
single slice: the oracle's achievable gain is exactly zero, so η's denominator vanishes.
This sparsity is the standing weakness carried over from Phase 0E, it is a property of the
scenario mix rather than of the metrics, and it applies identically to every signal
compared. It is stated here so the full-split numbers are read with it in mind: η on
nuScenes rests on a small minority of frames.

### 3.7 η's denominator — is there anything to allocate?

η is a ratio, and a ratio of nothing is meaningless, so the absolute stakes are recorded
before any η is interpreted. If the 320→640 fidelity step barely moved the planner, a low
η for PKL or TIP would say nothing about those metrics — no signal can rank noise.

| | | all cheap (320) | all full (640), **100%** compute | decision oracle, **20%** compute | ΔJ ≠ 0 |
|---|---|---|---|---|---|
| nuScenes | longitudinal | J = 893.8 | 804.8 (**−9.96%**) | 670.1 (**−25.03%**) | 295 / 3376 (8.7%), +152 / −143 |
| nuScenes | lateral | J = 154.2 | 150.1 (−2.70%) | 140.7 (−8.79%) | 42 / 3376 (1.2%), +25 / −17 |
| KITTI | longitudinal | J = 19372.1 | 9653.9 (−50.17%) | 7362.2 (**−62.00%**) | 3026 / 8008 (37.8%), +1794 / −1232 |
| KITTI | lateral | J = 2170.4 | 2437.2 (**+12.29%**) | 2032.6 (−6.35%) | 1010 / 8008 (12.6%), +400 / −610 |

Three things follow, and they cut in different directions.

**The denominator is real where it matters.** On nuScenes longitudinal the oracle at a 20%
quota cuts 25.0% of the all-cheap cost, while paying for the expensive mode on *every* frame
cuts only 10.0%. Allocation at one fifth of the compute beats uniform full fidelity by 2.5×.
There is a large, genuinely available prize, so a low η there is a statement about the
signal.

**A near-zero mean gain is not evidence of a near-zero effect.** Of the 295 nuScenes frames
where the fidelity step changes the decision cost, 152 are helped and **143 are hurt** —
running the expensive detector makes the decision *worse* on nearly half of the frames it
affects. Frame-level effects of both signs largely cancel in any average. This is why the
per-slice PKL summaries read as "640 is not better on average" (slice 1: PKL 52.44 cheap vs
52.53 full) while the oracle simultaneously finds a 25% cost reduction: the value is
frame-specific and sign-varying, so a metric's mean says nothing about whether it can *rank*
frames, which is the only thing an allocator needs.

**The lateral task is too thin to carry a verdict.** On nuScenes, ΔJ is non-zero on 1.2% of
frames and the oracle prize is 8.79%; on KITTI, uniform full fidelity is 12.3% *worse* than
uniform cheap and the oracle prize is 6.35%. The lateral column is therefore reported for
completeness and is **not** used to support the Stage-1 verdict, which rests on the
longitudinal task. Stating this now, before the numbers, prevents the weaker task from being
recruited to whichever side it happens to favour.

### 3.8 Giving the published metrics their best shot

Two choices are deliberately in PKL's and TIP's favour, because the pre-registration
requires prior work not be strawmanned:

1. The **`oracle` geometry variant is primary** (§3.2), so monocular depth and yaw error —
   irrelevant to the fidelity decision under test — does not depress their scores.
2. PKL and TIP levels differ roughly fourfold between scene groups (14 vs 52 on two slices),
   and so do their cheap-minus-full differences. A pooled top-quota ranking on the raw gain
   is therefore partly a ranking of *scenes* rather than of frames. Every signal is
   additionally evaluated in a **within-scene standardised** form (`[scene-z]`), and the
   published metrics are credited with whichever form does better.

### 3.9 The bar PKL and TIP have to clear, and the baseline that actually threatens the paper

Phase 0E's signals on exactly the cell PKL and TIP are evaluated on, η at a 20% quota:

| signal | nuScenes ns320→640, oracle geom, longitudinal | KITTI 320→640 mono, longitudinal |
|---|---|---|
| random (16 seeds) | +0.089 | +0.170 |
| visual uncertainty | −0.061 | +0.024 |
| downstream criticality | +0.163 | +0.021 |
| exact perception gain ΔE | +0.147 | +0.155 |
| best single ΔE variant | **+0.562** | +0.274 |
| multi-metric ΔE oracle (GBM, leave-one-scene-out) | +0.434 | **+0.700** |
| best trivial heuristic | +0.242 (ego speed) | **+0.787** (ego speed) |
| decision oracle ΔJ | 1.000 | 1.000 |

Two things this table settles before the PKL/TIP numbers arrive.

**A very low η for PKL is a liability, not a victory.** The pre-registered NO-GO threshold is
0.8 and nothing here exceeds 0.562, so there is room for a published planning-aware metric to
beat every task-agnostic signal and still fall short. But if PKL lands near random, the
natural reading is not "planning-aware metrics do not solve allocation" — it is "these authors
broke PKL when they converted 2D monocular detections into a 3D submission". That reading has
to be closed off by evidence, not by assertion, which is what §3.5 (the level tracks error
counts), the `oracle` geometry variant (§3.2) and the within-scene normalisation (§3.8) are
for. They belong in the main argument, not an appendix.

**The baseline that actually threatens this paper is ego speed.** On the flagship KITTI cell a
single scalar read off the CAN bus, involving no perception at all, reaches **η = 0.787** —
essentially the kill threshold. The mechanism is real rather than an artefact: the braking
requirement goes as v²/2·gap, so both the cost and the room to reduce it grow with speed.

The honest consequence is that this project cannot claim the allocation problem is unsolved.
What it can claim is that **no signal transfers**: ego speed nearly solves KITTI (0.787) and
fails on nuScenes (0.242), while the best ΔE variant does the opposite (0.274 → 0.562). A
practitioner reading either result alone would adopt the wrong allocator for the other
dataset. That is a weaker headline than "nothing works", and it is the one the data supports.

### 3.10 What survives, stated plainly

For the record, and because the pre-registration requires the negative side be written down:

*Removed by prior work* — first task-aware perception; introducing decision-aware perception;
introducing value of computation; "accuracy does not imply planning quality"; adaptive
perception compute. *Removed by our own results* — the recoverability factorisation (Phase
0B), the criticality contribution (Phase 0C), the lateral task as evidence (§3.6), and the
"unsolved problem" framing (ego speed, §3.9).

*Surviving, strongest first*

1. **The frame-level value of extra perception compute is sign-varying.** Of the frames whose
   decision cost the 320→640 step changes at all, close to half get *worse*: 143 / 295
   (48.5%) on nuScenes under Planner A, 1232 / 3026 (40.7%) on KITTI under Planner A, and
   504 / 1361 (37.0%) on KITTI under the structurally unrelated Planner B. The adaptive-
   compute literature assumes more compute is better and trades accuracy against latency;
   nothing in it reports that the *sign* is not guaranteed per frame. PKL and TIP do not
   either, because they score a *detector* aggregated over a dataset rather than asking a
   per-frame ranking question.
2. **Its quantified consequence:** spending the expensive detector on 20% of frames beats
   spending it on 100% — 25.0% vs 9.96% cost reduction on nuScenes (Planner A), 28.1% vs
   15.4% on KITTI (Planner B). Not a paradox: uniform full fidelity also pays for the frames
   where it hurts.
3. **No allocation signal transfers across datasets** (§3.9).
4. **Whether published planning-aware scores rank frames by marginal value of a specific
   computation** — the question Stage 1 exists to answer, and one the literature has not
   asked, because it is a per-frame ranking question rather than a detector-evaluation one.

*Still at risk*: both planners are hand-written and open-loop, so errors do not compound.
Planner B removes the single-controller objection but not the hand-written one. The strongest
available fix is to use PKL's own published learned planner as a third downstream decision
maker — maximally favourable to PKL, since ΔJ would then be defined by the very planner PKL
was designed around. If PKL's per-frame score still fails to rank frames by its own planner's
cost change, the redundancy objection closes completely. That uses their planner as an
evaluator and builds no method, so it stays inside the pre-registration.

