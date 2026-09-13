# Corrected results — the numbers to write from

*Written 2026-09-13, after the code review of the same date found seven defects (three of which
changed published numbers) and everything downstream of the detection submissions was recomputed.
**This file supersedes every number in `iclr_phase0f_planning_metric_stress_test.md` and
`iclr_phase0g_planner_d.md`.** Those documents keep their method and reasoning; where a number
there disagrees with one here, the number here is correct. Pre-review artefacts are preserved
under `data/cache/stale_pre_review/` and `results/final/stale_pre_review/`.*

Setting throughout: nuScenes trainval01, 85 scenes, 3,376 CAM_FRONT keyframes, YOLOv8s
`320 → 640` at a fixed operating point, Jetson AGX Xavier. η = share of the decision-value
oracle's achievable cost reduction captured at a compute quota; 95% intervals from a 400-draw
scene-level bootstrap; tie-breaks randomised and averaged over 8 seeds.

---

## 1. What the review changed, and what it did not

| defect | effect on results |
|---|---|
| ranking ties broken by row order | η of **count-valued** signals only (ΔE, E1); worst at the 20% quota, where 52% of the selected set was decided by row order |
| the same defect in the cross-target statistics | **`overlap@30` was an artefact** (see §4) |
| monocular lift wrote the near-face range as the box centre | every submission; fully for `mono`, in the false-positive boxes for `oracle` |
| pixel→metre inversion off half a cell | absolute ADE in the truth-referenced Planner C cost |
| deployable and diagnostic signals not separated | presentation of every signal table (see §5) |

**Two errors found after this file was first written, by the user reading the artefacts against
the text.** (i) The mono η run read the *oracle* truth-referenced cost file, because
`--planner_c_truth` had a fixed default instead of being derived from `--variant`, so it paired
mono signals with an oracle target. The giveaway was that `plannerC_ade_truth` came out identical
in the oracle and mono runs — which I saw, and explained away by noting that the *signals* do not
depend on the variant, without checking the *cost* columns. The flag is now derived from
`--variant` and prints the file it uses; the mono `plannerC_ade_truth` row is being recomputed and
must not be used until it is. The cross-target mono result used the correct file and is unaffected.
(ii) The affected-frame count for the truth-referenced target in §2 was quoted from the *pre-review*
run (1,251) while everything around it was post-review (1,226). Numbers here now come from
`allocation_stakes.csv` in the run directories rather than from prose.

**Verified unaffected.** The oracle prize is tie-invariant — 670.0625 across five tie-break seeds
— because frames added from the ΔJ = 0 tie group contribute nothing. So every oracle-prize
figure, every affected/harmed count, and the sign-varying finding never depended on the defect.

A methodological point worth keeping: an independent hand-computed pipeline reproduced the
pre-review numbers **exactly**, because it shared the same stable-argsort convention. Agreement
between two implementations cannot detect an arbitrary convention they both use; only reading the
code found it.

---

## 2. Sign-varying value of perception compute — the leading claim

Of the frames whose decision cost the 320→640 step changes at all, close to half are made
**worse** by the expensive detector:

| downstream system | geometry | frames with ΔJ ≠ 0 | made worse |
|---|---|---|---|
| braking controller | oracle | 295 / 3,376 | **48.5%** |
| braking controller | mono | 486 / 3,376 | **45.5%** |
| rollout planner (KITTI) | mono | 1,361 / 8,008 | 37.0% |
| PKL's own planner, real-trajectory cost | oracle | 1,226 / 2,655 | **51.4%** (630 of 1,226) |
| PKL's own planner, real-trajectory cost | mono | *pending re-run* | **51.6%** |
| learned waypoint planner (6 models) | oracle | 62.5% of frames | 47.6–53.6% |

Stable across two datasets, two geometry variants, and four decision systems that share no
decision logic. The corrections moved it slightly **up** (0.496 → 0.514 for Planner C).

## 3. Selective allocation beats uniform full fidelity

| system | all-CHEAP → all-FULL at 100% compute | oracle at a **20%** quota |
|---|---|---|
| braking, oracle geometry | −9.96% | **−25.03%** |
| braking, mono geometry | −12.99% | **−27.97%** |
| rollout planner, KITTI | −15.37% | **−28.05%** |
| PKL's planner, real trajectory | −1.46% | −5.18% |

Not a paradox: uniform full fidelity also pays for the frames where it hurts.

## 4. Do two downstream systems agree about which frames deserve compute?

The central claim in score-free form — no metric, no PKL, no perception feature, only the two
decision values. Every quantity's baseline is recomputed inside the same bootstrap draw so a
**paired** difference can be reported; comparing two separately-computed intervals tests a much
weaker claim.

**Oracle geometry**, braking controller vs PKL's planner on real-trajectory cost, 2,655 frames:

| quantity | value | chance / random | paired difference |
|---|---|---|---|
| Spearman, all frames | −0.023 | 0 | [−0.073, +0.031] |
| Spearman, frames where both respond | −0.146 | 0 | [−0.357, +0.091] |
| overlap@10 | 0.100 | 0.100 | −0.000 [−0.021, +0.047] |
| overlap@20 | 0.206 | 0.200 | +0.006 [−0.024, +0.034] |
| overlap@30 | 0.298 | 0.300 | −0.002 [−0.026, +0.028] |
| cross-η brake→plan @20 | +0.010 | +0.044 | −0.034 [−0.112, +0.052] |
| cross-η plan→brake @20 | +0.037 | +0.111 | −0.074 [−0.245, +0.105] |

**Every paired difference contains zero.** Under mono geometry the same holds (largest is
overlap@10, +0.052 [−0.006, +0.110]).

**The one pre-review result that excluded zero was an artefact.** `overlap@30` was 0.438 against
a chance level of 0.300, paired +0.139 [+0.044, +0.238]. The braking controller responds on only
219 of 2,655 frames, so 73% of its top-30% set was tied at ΔJ = 0; both systems broke that tie by
row order and therefore selected the *same early frames*. With random tie-breaks it sits at
chance. Removing it makes the picture uniform rather than weakening it: the two systems are
indistinguishable from unrelated at every budget, by every measure.

`responsive_frac` is now reported so this cannot recur silently: the braking controller genuinely
distinguishes only 46.6% / 23.4% / 15.6% of its top-10/20/30% sets under oracle geometry.

## 5. Allocation signals — deployable and diagnostic kept apart

A signal is **deployable** only if it can be computed *before* deciding to escalate. Note that
`G_PKL = pkl_cheap − pkl_full` requires the expensive output, so **PKL and TIP as used here are
diagnostics, not candidate allocators**; so is every ΔE variant. And `crit_sum` comes from
**ground-truth** geometry — a different quantity from the cheap-side `feat_crit_sum` in
`features.py`.

**η@20, nuScenes, braking controller (longitudinal)**

| signal | class | oracle geometry | mono geometry |
|---|---|---|---|
| random, 16 seeds | deployable | +0.078 [−0.10, +0.22] | +0.103 [−0.03, +0.27] |
| cheap-detection uncertainty | **deployable** | −0.061 [−0.31, +0.13] | **+0.198 [+0.01, +0.35]** |
| criticality (GT geometry) | diagnostic | +0.163 [−0.11, +0.42] | +0.177 [+0.06, +0.34] |
| exact perception gain ΔE | diagnostic | +0.134 [+0.02, +0.24] | +0.030 [−0.08, +0.14] |
| **best single ΔE variant (E6 risk-weighted)** | diagnostic | **+0.562 [+0.28, +0.75]** | +0.165 [−0.01, +0.35] |
| multi-metric ΔE oracle (GBM, LOSO) | diagnostic | +0.434 [+0.22, +0.62] | +0.151 [+0.04, +0.28] |
| **PKL gain** (CVPR 2020) | diagnostic | **+0.220 [−0.04, +0.42]** | +0.087 [−0.02, +0.19] |
| **TIP gain** (ICML 2023) | diagnostic | **+0.202 [−0.04, +0.43]** | +0.068 [−0.05, +0.17] |
| decision oracle ΔJ | label | 1.000 | 1.000 |

**η@20, PKL's own planner scored against the real future trajectory**

| signal | oracle geometry | mono geometry |
|---|---|---|
| random | +0.036 [−0.07, +0.13] | +0.036 [−0.07, +0.13] |
| exact ΔE = best ΔE variant (E1) | +0.222 [+0.05, +0.35] | +0.222 [+0.05, +0.35] |
| multi-metric ΔE oracle | −0.031 [−0.18, +0.09] | −0.031 [−0.18, +0.09] |
| **PKL gain** | **+0.395 [+0.14, +0.57]** | +0.260 [+0.07, +0.41] |
| **TIP gain** | **+0.390 [+0.15, +0.57]** | +0.221 [+0.04, +0.38] |

### The ordering reverses with the downstream system

| signal | braking controller | PKL's own planner |
|---|---|---|
| best single ΔE variant | **+0.562** | +0.222 |
| PKL gain | +0.220 | **+0.395** |
| TIP gain | +0.202 | **+0.390** |

A risk-weighted perception error beats PKL 2.6× on one downstream system and loses to it 1.8× on
the other, on the same frames and the same perception transition. **Pre-review these ratios were
5.2× and 1.7×**: correcting the box centres roughly doubled PKL (0.109 → 0.220) and tripled TIP
(0.070 → 0.202), so most of PKL's apparent failure on the braking controller had been our own
box-placement error. The reversal survives; its magnitude halved.

Two further points that survive unchanged. PKL's η on **its own** planner falls from **0.872 to
0.395** when the cost is referenced to the real trajectory instead of to the planner's own
ground-truth-conditioned output — so more than half of the headline 0.872 was the shared
functional form, now measured rather than suspected. And **under mono geometry every signal
collapses**: the best is cheap-detection uncertainty at 0.198, the only deployable signal whose
interval excludes zero, while every diagnostic falls to near random.

## 6. Phase 0F headline table, corrected (η@20, longitudinal)

| cell | ΔE exact | best ΔE variant | multi-metric oracle |
|---|---|---|---|
| nuScenes 320→640, oracle | 0.147 → **0.134** | 0.562 → **0.562** | 0.434 → **0.434** |
| nuScenes 320→640, mono | 0.047 → **0.030** | 0.165 → **0.165** | 0.151 → **0.151** |
| KITTI 320→640, mono | 0.155 → **0.161** | 0.274 → **0.274** | 0.700 → **0.700** |
| KITTI 512→640, mono | 0.213 → **0.133** | 0.221 → **0.195** | 0.167 → **0.167** |

Largest change over all 16 rows: 0.083. `best ΔE` and the multi-metric oracle are unmoved because
both are continuous (tie group of exactly 1); only ΔE-exact and E1 move, as predicted.

## 7. Claims that must not be made

- **Not** "planning-aware metrics fail at compute allocation" — PKL reaches 0.395 on its own
  planner and 0.220 on the braking controller, the latter's interval overlapping random but its
  point estimate twice what we first reported.
- **Not** "we know what to allocate by" — under realistic monocular geometry nothing beats
  cheap-detection uncertainty at 0.198, and no diagnostic separates from random.
- **Not** anything resting on Planner D. It failed its pre-registered viability gate on all six
  runs (D-F4): the whole BEV scene contributes ~1% beyond the ego's own velocity, reproducing the
  known result that nuScenes open-loop planning is ego-status dominated. Its decision value is
  absent and symmetric — mean |ΔJ_D| ≈ 0.0003 m with a coin-flip sign — which belongs in
  limitations as evidence about the *setting*, not as evidence for the claim.
- **Not** any closed-loop claim. Everything here is open-loop and single-step; errors never
  compound.

## 8. What the evidence does support

> The marginal downstream value of additional perception compute is **sign-varying at the frame
> level** — close to half the affected frames are made worse — and **which frames are valuable is
> a property of the perception–planner pair, not of the perception system**: two downstream
> systems' oracle rankings of the same frames are indistinguishable from unrelated at every
> budget tested, and the ranking of allocation signals reverses between them.

External validation is the open gap. Track B infrastructure is in place and step B2 passed:
PDM-Closed and IDMPlanner both run untouched on nuPlan mini, 4/4 scenarios each, on the same
scenario tokens, with official metrics. The perception intervention and counterfactual branching
are not yet built.
