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

---

## 4. A sign convention that had to be derived, not read

PKL and TIP are both described as divergences of a planner's output from its
ground-truth-conditioned output, and TIP's code carries the comment *"to conply with PKL's
definition"*. Taking that at face value — both higher-is-worse — was wrong, and the wiring
check of §3.5 is what caught it: PKL's per-frame gain correlated **+0.188** with the exact
perception gain ΔE while TIP's correlated **−0.178**. Two metrics measuring the same kind of
thing on the same frames cannot disagree in sign about whether the expensive detector helped.

Reading `get_tip` settles it. With `p` the ground-truth-conditioned distribution, `q` the
predicted one, `A* = argmax p` and `Â = argmin (p − q)`:

```
TIP_t = [ q(A*) − q(Â) ] − [ p(A*) − p(Â) ]
```

Because `A*` maximises `p`, the second bracket is ≥ 0. Because `Â` minimises `p − q`,
`q(Â) − p(Â) ≥ q(A*) − p(A*)`, so the first bracket is ≤ the second. Hence **TIP ≤ 0, and
more negative is worse**, reaching 0 only when the predicted distribution induces the same
preference gap as ground truth. This is the *opposite* direction to PKL. The `"comply with
PKL's definition"` comment sits on a `torch.from_numpy` call and concerns the tensor type.

So the allocation signals are

```
G_PKL = pkl_cheap − pkl_full          (PKL: higher is worse)
G_TIP = tip_full  − tip_cheap          (TIP: higher is better)
```

both positive when the expensive mode helps that frame. After the correction TIP's gain
correlates **+0.178** with ΔE, matching PKL's **+0.188** — the two published metrics now agree
about direction, which is the check that the convention is finally right. The gains are
recomputed from the stored per-mode scores in the analysis step, so fixing a convention never
requires re-running a metric.

## 5. Stage-1 result

nuScenes trainval01, 85 scenes, 3,376 CAM_FRONT keyframes, `ns_cheap_320 → ns_full_640`,
YOLOv8s, `oracle` geometry, pooled top-quota selection. η with a 95% interval from a
scene-level bootstrap (400 draws, degenerate draws dropped and counted). **Longitudinal task**
— the lateral task is reported in the artefacts but is too thin to carry a verdict (§3.6).

| signal | η@10 | **η@20** | 95% CI | cost cut @20% | η@30 | η@50 | ρ vs ΔJ |
|---|---|---|---|---|---|---|---|
| random (16 seeds) | +0.039 | +0.078 | [−0.10, +0.22] | 2.0% | +0.126 | +0.239 | — |
| visual uncertainty | −0.083 | −0.061 | [−0.31, +0.13] | −1.5% | −0.042 | +0.088 | −0.034 |
| downstream criticality | +0.070 | +0.163 | [−0.10, +0.42] | 4.1% | +0.115 | +0.346 | +0.033 |
| exact perception gain ΔE | +0.069 | +0.147 | [−0.00, +0.26] | 3.7% | +0.163 | +0.437 | +0.046 |
| **best single ΔE variant** (E6 risk-weighted) | +0.369 | **+0.562** | **[+0.28, +0.75]** | **14.1%** | +0.532 | +0.442 | +0.064 |
| multi-metric ΔE oracle (GBM, LOSO) | +0.374 | +0.434 | [+0.22, +0.62] | 10.9% | +0.535 | +0.580 | +0.080 |
| **PKL gain** (CVPR 2020) | +0.195 | **+0.109** | [−0.20, +0.34] | 2.7% | +0.181 | +0.250 | −0.007 |
| **TIP gain** (ICML 2023) | +0.139 | **+0.070** | [−0.24, +0.33] | 1.8% | +0.092 | +0.144 | −0.031 |
| decision oracle ΔJ | +1.000 | +1.000 | — | 25.0% | +1.000 | +1.000 | +1.000 |

With the within-scene normalisation of §3.8, which helps the published metrics most:

| signal [scene-z] | η@10 | η@20 | 95% CI | η@30 | η@50 |
|---|---|---|---|---|---|
| PKL gain | +0.154 | **+0.224** | [−0.10, +0.42] | +0.295 | +0.344 |
| TIP gain | +0.128 | +0.155 | [−0.07, +0.40] | +0.269 | +0.229 |
| exact ΔE | +0.222 | +0.373 | [+0.21, +0.54] | +0.429 | +0.425 |
| best single ΔE variant | +0.482 | +0.539 | [+0.27, +0.74] | +0.547 | +0.609 |
| multi-metric ΔE oracle | +0.370 | +0.441 | [+0.26, +0.61] | +0.491 | +0.483 |

### Hard Kill Test 1 — **PASSED, proceed**

The pre-registered rule: stop if η_PKL ≥ 0.8 or η_TIP ≥ 0.8 at a 20% quota, consistently
across tasks. The maximum η@20 reached by any published planning-aware metric in any form,
including the normalisation that favours them, is **0.224** (PKL, scene-z, longitudinal).
That is not close to 0.8, and not in the 0.6–0.8 WEAK GO band either.

### The outcome the pre-registration did not anticipate

The bands were written expecting PKL and TIP to *beat* task-agnostic perception metrics and
still fall short of the oracle — "STRONG EMPIRICAL GAP if planning-aware scores beat standard
perception metrics but stay < 0.6". They stay well below 0.6, but they **do not beat the
standard metrics**: a plain risk-weighted perception error reaches 0.562 [0.28, 0.75] while
PKL reaches 0.109 [−0.20, +0.34] and TIP 0.070 [−0.24, +0.33], both intervals containing
random's 0.078. At a 20% quota, neither published planning-aware metric is distinguishable
from choosing frames at random, and each is beaten by an ordinary perception metric.

That wording is recorded rather than quietly relabelled, because it changes what the paper
claims. The finding is not "planning-aware metrics get partway there". It is that **being
planning-aware about which errors matter does not by itself say which frames are worth more
computation** — and on this evidence it helps less than weighting perception error by risk.

Neither metric is at fault on its own terms: both were designed to score a *detector* over a
dataset, and both do track error counts frame by frame (§3.5). The gap is between *relevance
of an error* and *marginal value of one specific extra computation*, and it is not closed by
making an evaluation metric planning-aware.

---

## 6. Planner C — PKL's own planner as the downstream decision maker

§5 rested on decision costs from planners we wrote. The obvious objection is that a metric
built around a different planner cannot be expected to predict them. Planner C removes the
objection by making the downstream decision maker *PKL's own published planner* at its
released weights (`scripts/66_planner_c_pkl_planner.py`).

That planner emits, for each of 16 future timesteps from 0.25 s to 4.0 s, a heatmap of where
the ego will be on a 0.3 m BEV grid — the 16 channels are timesteps, not trajectory templates
— so its decision is the path it intends. The cost of a mode is

```
J_C(mode) = mean over timesteps of || argmax path given that mode's detections
                                    − argmax path given ground-truth boxes ||   [metres]
```

No hand-written cost function, no threshold, units of metres. On the 3,376 frames the signal
is far denser than either hand-written planner's: **|ΔJ_C| > 0 on 1,523 / 3,376 frames (45%)**,
against 295 / 3,376 (8.7%) for the longitudinal task. The oracle prize is 15.11% of the
all-cheap cost at a 20% quota, against 6.76% for uniform full fidelity at 100%.

### The result, and it reverses the ordering

| signal | η@20 | 95% CI | ρ vs ΔJ_C | inversion rate |
|---|---|---|---|---|
| random | +0.102 | [+0.00, +0.17] | — | — |
| visual uncertainty | +0.025 | [−0.12, +0.17] | −0.023 | 0.511 |
| downstream criticality | +0.227 | [+0.09, +0.34] | +0.056 | 0.478 |
| exact perception gain ΔE | +0.317 | [+0.18, +0.42] | +0.133 | 0.427 |
| best single ΔE variant (E6) | +0.420 | [+0.26, +0.56] | +0.175 | 0.413 |
| multi-metric ΔE oracle | +0.341 | [+0.18, +0.48] | +0.107 | 0.453 |
| **PKL gain** | **+0.872** | **[+0.823, +0.911]** | **+0.704** | **0.166** |
| **TIP gain** | **+0.826** | [+0.772, +0.867] | +0.637 | 0.200 |
| decision oracle ΔJ_C | +1.000 | — | +1.000 | 0.000 |

Side by side with §5, the ordering inverts completely:

| signal | η@20, Planner A (longitudinal) | η@20, Planner C (PKL's own) |
|---|---|---|
| best single ΔE variant | **+0.562** [+0.28, +0.75] | +0.420 [+0.26, +0.56] |
| PKL gain | +0.109 [−0.20, +0.34] | **+0.872** [+0.82, +0.91] |
| TIP gain | +0.070 [−0.24, +0.33] | **+0.826** [+0.77, +0.87] |

### How much of this is circular — stated before any interpretation

A large part of it. PKL is a divergence between the predicted trajectory heatmap and the
ground-truth-conditioned one; J_C is the displacement of the **argmax of those same
heatmaps**. They share the model, the weights, the forward pass, the inputs, *and* the
functional form (both compare prediction against ground truth). A frame whose heatmap has
moved a lot will tend to score high on both almost mechanically. η = 0.872 is therefore in
large part an **internal-consistency** result, not evidence that PKL ranks frames by the
marginal value of computation for an independent downstream task. When §3.10 called this test
"maximally favourable to PKL" it understated the point: the test is close to tautological, and
saying so is part of reporting it.

ρ = +0.704 rather than ≈1 shows the two are not the same functional — the argmax can stay put
while the distribution shifts, and vice versa — but the shared construction is enough that
this cell cannot carry a claim about PKL solving allocation in general.

What it *does* establish, and what no weaker check could: **PKL and TIP are wired correctly
here.** Neither could reach 0.87 and 0.83 on 3,376 frames if converting our 2D monocular
detections into a 3D submission had damaged them. That closes the "these authors broke PKL"
reading of §5 far more firmly than the ρ = +0.29 level check of §3.5.

### Hard Kill Test 1, re-evaluated honestly

The pre-registered rule stops the project if η_PKL ≥ 0.8 or η_TIP ≥ 0.8 at a 20% quota
**consistently across downstream tasks**. Observed: 0.109 (longitudinal), 0.107 (lateral),
0.872 (Planner C) for PKL. That is as far from consistent as the data could be, so the rule
does not fire — but that is a technicality about the wording, and it is not the reason to
continue. The substantive reasons are that the one cell above 0.8 is the near-circular one,
and that the cells using independent planners sit at random.

### What the claim has to become

Not "planning-aware perception metrics fail at compute allocation" — §6 refutes that as
stated. Not "they solve it" either — §5 refutes that. What the evidence supports is:

> **Which signal allocates compute best is determined by which planner defines the downstream
> cost, and no signal transfers.** PKL is near-optimal for the planner it was built around
> (η = 0.87) and indistinguishable from random for an independent one (η = 0.11). A plain
> risk-weighted perception error is the reverse (0.56 vs 0.42). Allocation value is a property
> of the perception–planner pair, not of the perception system alone.

This is a weaker claim than the project set out with and a more defensible one, and it stacks
with the two findings that survive unchanged: the frame-level value of extra compute is
**sign-varying** — on Planner C, 643 of the 1,523 affected frames (42%) are made *worse* by the
expensive detector, matching 48.5% under Planner A — and **selective allocation at a 20% quota
beats uniform full fidelity at 100%** in every cell measured, including this one (15.11% vs
6.76%).


---

> **Superseded numbers.** The 2026-09-13 code review found seven defects, three of which
> changed numbers in this document, and everything downstream of the detection submissions was
> recomputed. The method and reasoning here stand; for every figure see
> [`iclr_corrected_results.md`](iclr_corrected_results.md), which takes precedence.
