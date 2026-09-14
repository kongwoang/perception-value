# Phase 0D — Is the Problem Real Enough to Pursue?

**Verdict: STRONG GO.** Knowing exactly where extra perception compute improves the
detector tells you almost nothing about where it improves the downstream decision. That
holds on two datasets, two downstream tasks, two fidelity gaps, and every artefact control
the pre-registration named. The two direct answers that matter are J and the one-sentence
answer at the end.

| | |
|---|---|
| Datasets | KITTI tracking (21 sequences, 8,008 frames) · nuScenes mini (10 scenes, 404 keyframes) |
| Perception | YOLOv8s FP16 TensorRT; 320→640 and 512→640 on KITTI, 320→640 on nuScenes |
| Tasks | longitudinal braking (KEEP/DECELERATE/HARD_BRAKE) · lateral avoidance (KEEP/LEFT/RIGHT/BRAKE) |
| Runs | `20260911_223528_stage1`, `20260911_224615_stage2`, `20260911_233520_nusc_decision`, `20260912_002239_final_matrix` |

## Claim being tested

Pre-registered before any Phase 0D experiment was run:

> Marginal improvement in perception quality is not a reliable proxy for marginal
> improvement in downstream decision quality. ΔJ is not reducible to ΔE, uncertainty,
> criticality, or trivial operating-state variables.

Three properties had to hold separately: **distinct**, **general**, **non-trivial**.
Six falsifiers were named in advance. None fired.

## Existing Phase 0C evidence

Reproduced exactly: action-change rate 27.3% (17.0% beneficial, 10.3% harmful),
67.0% of frames with improved detection keep the same action, corr(ΔE, ΔJ) = +0.041,
η of the perception-gain oracle at a 20% quota = 0.155 against 1.000 for the
decision-value oracle.

## Speed confound

Phase 0C found ego speed alone reached η = 0.787, which threatened to make the whole
problem trivial. Two controls settle it:

* **Partial Spearman(ΔE, ΔJ | ego speed) = +0.042**, essentially unchanged from the raw
  +0.041. Speed does not explain the decoupling.
* **Under a speed-stratified budget** — the quota spent separately inside each speed bin,
  so a policy cannot satisfy it by picking fast frames — ego speed collapses from 0.787 to
  **0.285**, while the perception-gain oracle stays at 0.213.

Within individual speed bins the mismatch persists and is strongest at low-to-moderate
speed:

| ego speed | n | action differs | corr(ΔE,ΔJ) | η_E@20 | η random |
|---|---|---|---|---|---|
| 0–5 m/s | 3,342 | 0.190 | −0.071 | 0.062 | −0.031 |
| 5–10 m/s | 2,422 | 0.289 | −0.034 | 0.001 | 0.000 |
| 10–15 m/s | 1,893 | 0.360 | +0.294 | 0.225 | 0.210 |
| 15–20 m/s | 351 | 0.484 | +0.230 | 0.430 | 0.262 |

## Empty-detection confound

14.6% of KITTI frames have no CHEAP detection at all and carry 57.9% of positive decision
gain. Removing them **strengthens** the result rather than weakening it:

| subset | n | corr(ΔE,ΔJ) | η_E@20 | oracle gap |
|---|---|---|---|---|
| all frames | 8,008 | +0.041 | 0.155 | 0.845 |
| non-empty CHEAP | 6,839 | **−0.006** | **0.107** | 0.893 |
| ≥2 CHEAP candidates | 5,618 | **−0.033** | **0.061** | 0.939 |
| non-empty and moving >5 m/s | 3,765 | +0.043 | 0.128 | 0.872 |

Falsifier 3 does not fire: this is not a cheap-model-collapse artefact.

## Range-estimation confound

Three geometry sources, the middle one isolating *which objects were detected* from *how
well they were ranged*:

| geometry | action differs | harmful | corr(ΔE,ΔJ) | η_E@20 |
|---|---|---|---|---|
| deployed monocular | 0.273 | 0.103 | +0.041 | 0.155 |
| oracle range for matched detections | 0.172 | 0.049 | +0.141 | 0.198 |
| calibrated-noise GT range (σ = 0.35) | 0.346 | — | +0.056 | 0.141 |

Monocular error inflates the action churn — roughly 40% of it — but the mismatch survives
under both controlled-geometry variants. Falsifier 4 does not fire.

## Moderate fidelity

| pair | action differs | corr(ΔE,ΔJ) | η_E@20 |
|---|---|---|---|
| 320→640 (aggressive) | 0.273 | +0.041 | 0.155 |
| 384→640 | 0.226 | +0.032 | 0.160 |
| **512→640 (moderate)** | **0.191** | **+0.039** | **0.213** |
| 512→640 non-empty | 0.198 | +0.028 | 0.161 |

The effect is not an artefact of a severely degraded cheap model. Falsifier 3 does not fire.

## Longitudinal task

Already summarised above. The headline: a perfect oracle on perception gain captures
15.5% of the achievable decision-cost reduction at a 20% quota — statistically
indistinguishable from random (16.6%).

## Lateral / trajectory task

A second, independent decision: choose a corridor (KEEP / LEFT_AVOID / RIGHT_AVOID /
BRAKE) from perceived obstacle geometry, scored against the true scene by whether the
chosen corridor was actually clear, plus lateral deviation and progress. Parameters were
chosen by measuring the clearance distribution, not guessed: a 3.2 m swept corridor with a
2.5 s headway leaves the lane blocked on 22% of frames with a shift available on 86% of
those. A 2 m corridor made ground truth choose KEEP on 97% of frames and carried no
information.

The result is sharper than the longitudinal one, and uncomfortable:

* **FULL perception is more accurate yet more costly.** Correct action 83.8% vs 81.9%,
  but total cost 2,437 vs 2,170. It cuts missed manoeuvres (0.078 → 0.043) and raises
  phantom ones (0.071 → 0.087): extra detections block clear corridors.
* This holds in **all ten** planner and cost variants tested, including the most
  safety-weighted one. It is not a cost-weighting artefact.
* **η_E@20 = −0.257.** Ranking frames by perception gain is *worse than useless* here:
  it actively selects frames where escalation hurts.
* 88.1% of frames with improved detection keep the same lateral action.

## Cross-task conditionality

The same perception, two objectives, and they want different frames:

| configuration | Spearman(ΔJ_long, ΔJ_lat) | top-10% overlap | top-20% overlap |
|---|---|---|---|
| KITTI 320→640 | −0.150 | **0.032** | 0.282 |
| KITTI 512→640 | +0.009 | 0.131 | 0.388 |
| KITTI oracle range | −0.186 | 0.031 | 0.503 |
| nuScenes mini | +0.026 | 0.150 | 0.630 |

Pre-registered threshold was "top-20% overlap well below 80%"; observed 0.28–0.63.
Cross-applying a ranking is catastrophic: ranking by lateral value and paying the
longitudinal cost gives η = 0.087; the reverse gives η = −1.544.

**Negative control:** when both tasks are given the *same* cost function, top-20% overlap
is exactly 1.000, confirming the machinery does not manufacture task dependence.

Falsifier 2 does not fire. This is the strongest novelty signal in Phase 0D: the optimal
compute allocation is a property of *what the robot is trying to do*, not only of the scene.

## Second dataset

nuScenes mini, CAM_FRONT keyframes, engines rebuilt at 16:9 to preserve the same 4×
pixel ratio as the KITTI pair.

| configuration | task | action differs | improved, same action | corr(ΔE,ΔJ) | η_E@20 | oracle gap |
|---|---|---|---|---|---|---|
| nuScenes mini | longitudinal | 0.111 | 0.855 | −0.046 | **0.058** | 0.942 |
| nuScenes mini | lateral | 0.030 | 0.978 | +0.020 | **−0.181** | 1.181 |
| nuScenes oracle range | longitudinal | 0.067 | 0.899 | +0.003 | 0.271 | 0.729 |
| nuScenes oracle range | lateral | 0.027 | 0.969 | −0.032 | −0.277 | 1.277 |

The claim replicates, and more strongly than on KITTI. Falsifier 5 does not fire.

**A bug worth recording.** nuScenes `sample_annotation.translation` is the centre of the
3D box, not the centre of the bottom face as in KITTI. Treating it as the bottom shifted
every projected 2D box up by h/2 (~30 px at 30 m) and left **0.5%** of detections matching
a ground-truth box. After the fix, 63.4% — comparable to KITTI. Every nuScenes number
above is post-fix.

## Optional second detector

Not run. Per the execution order it is gated behind the dataset test, and the dataset test
only completed at mini scale.

## Trivial-heuristic ceiling

| heuristic | KITTI pooled | KITTI speed-stratified | nuScenes pooled |
|---|---|---|---|
| ego speed | 0.787 | 0.285 | 0.210 |
| empty-frame indicator | 0.560 | 0.116 | 0.297 |
| few cheap detections | 0.639 | 0.130 | 0.255 |
| low confidence | 0.631 | 0.128 | — |
| **speed × empty** | **0.732** | **0.563** | — |
| criticality | 0.021 | 0.128 | 0.076 |
| uncertainty | 0.024 | 0.034 | 0.080 |
| perception-gain oracle | 0.155 | 0.213 | 0.058 |

The pre-registered falsifier was a 1–2 variable heuristic exceeding **90%** of oracle value
across configurations. The best observed is 0.732 on KITTI pooled, falling to 0.563 under
speed stratification and to 0.297 on nuScenes. On the lateral task every heuristic is
negative. Falsifier 6 does not fire — but this is the weakest of the six margins, and
`speed × empty` deserves to be a named baseline in any future paper.

## Oracle allocation gap

The table the plan asked for, with per-sequence robustness added:

| configuration | task | η_E@20 | oracle gap | per-seq median | sequences with η_E > 0.5 |
|---|---|---|---|---|---|
| KITTI 320→640 | longitudinal | 0.155 | 0.845 | 0.159 | 3 / 18 |
| KITTI 320→640 | lateral | −0.257 | 1.257 | −0.308 | 2 / 15 |
| KITTI 512→640 | longitudinal | 0.213 | 0.787 | −0.016 | 4 / 17 |
| KITTI 512→640 | lateral | −0.130 | 1.130 | −0.077 | 0 / 14 |
| KITTI oracle range | longitudinal | 0.198 | 0.802 | 0.405 | 6 / 18 |
| KITTI oracle range | lateral | −0.243 | 1.243 | −0.202 | 2 / 15 |
| KITTI non-empty | longitudinal | 0.107 | 0.893 | 0.130 | 1 / 18 |
| KITTI non-empty | lateral | −0.071 | 1.071 | −0.308 | 1 / 15 |
| nuScenes mini | longitudinal | 0.058 | 0.942 | 0.000 | 2 / 7 |
| nuScenes mini | lateral | −0.181 | 1.181 | 0.000 | 2 / 5 |
| nuScenes oracle range | longitudinal | 0.271 | 0.729 | 0.000 | 1 / 5 |
| nuScenes oracle range | lateral | −0.277 | 1.277 | 0.000 | 0 / 2 |

**Every one of the twelve rows is at or below η_E = 0.271**, against a pre-registered
threshold of 0.5.

## Negative controls

| control | expectation | observed |
|---|---|---|
| perception-independent fixed action | ΔJ ≡ 0 | exactly 0 for all three actions |
| planner-insensitive controller | concentration shrinks | action-change rate 0.273 → 0.055 |
| shuffled FULL improvements | corr(ΔE,ΔJ) → 0 | +0.067, *above* the real +0.041 |
| task-independent cost | ranking overlap → 1 | exactly 1.000 |

## Counterexamples

All seven case types exist and are common. Shares of all KITTI frames:

| type | share | example |
|---|---|---|
| A — large ΔE, zero decision effect | **0.208** | seq 0019 f123: ΔE = 4, ΔJ = 0 |
| B — ΔE ≤ 0 but decision improves | 0.042 | seq 0019 f777: ΔE = 0, ΔJ = +33.3 |
| C — perception improves, decision worsens | **0.117** | seq 0019 f83: ΔE = 2, ΔJ = −77.9 |
| D — valuable longitudinally, not laterally | 0.155 | seq 0019 f196: ΔJ_long = +78.1, ΔJ_lat = 0 |
| E — valuable laterally, not longitudinally | 0.019 | seq 0004 f14: ΔJ_lat = +1.0, ΔJ_long = 0 |
| F — fast ego and decision improves | 0.131 | — |
| G — **fast ego but escalation wasted** | **0.198** | — |

Type G exceeds type F: even the strongest trivial heuristic is wrong more often than right.

## Threats to validity

1. **The second dataset is mini-scale.** 10 scenes, 404 keyframes. The plan explicitly says
   not to conclude from mini. The direction and magnitude replicate clearly, but the
   per-scene estimates are noisy (2 of 7 scenes show η_E > 0.5) and the top-20% task
   overlap of 0.63 is the least comfortable number in the report. A 31.6 GB
   `v1.0-trainval01` blob (~85 scenes) is downloading; this section should be re-run on it
   before any submission, and the verdict revisited if it disagrees.
2. **Per-sequence variance is high.** In 3–6 of 18 KITTI sequences the perception-gain
   oracle does capture more than half the decision value. The pooled claim is strong; a
   per-scene claim would not be.
3. **Better geometry weakens the effect.** With oracle range the KITTI per-sequence median
   η_E rises from 0.159 to 0.405. A stack with LiDAR or stereo would sit somewhere between
   the monocular and oracle columns, and the gap would be smaller than reported here.
4. **Open loop, one frame at a time.** No closed-loop simulation, so the compounding cost
   of a bad action is absent, as is any recovery from one.
5. **Two hand-built planners, not a real stack.** Both are deterministic rule-based
   controllers. The cost functions were swept (10 variants) but they are still proxies.
6. **One detector family.** YOLOv8s at two resolutions throughout.
7. **The lateral result is partly a false-positive story.** FULL hurting the lateral task
   is driven by extra detections blocking corridors; a detector with better precision at
   the same recall might not show it.

## Literature-position implications

The defensible contribution is a *problem and measurement* contribution, not a method:
perception-level metrics — mAP, recall, and the risk-weighted variants from Phase 0 — do
not rank frames by downstream decision value, and an oracle on them captures under a
quarter of the achievable decision-cost reduction. Two further findings sharpen it: the
optimal allocation is **task-conditional** (top-10% overlap 0.03–0.15 between two tasks on
the same frames), and more perception is **not monotonically better** at the decision level
(10.3% of longitudinal frames get worse; the whole lateral task gets worse).

This also retires the Phase 0 result on its own terms. Criticality-weighted risk, the
contribution of Phase 0, has η ≈ 0.02–0.08 for decision cost — near zero. Phase 0 measured
a better perception metric, not a better allocation objective.

## Final answers

**A. Is ΔJ genuinely different from ΔE?** Yes. corr = +0.041 pooled, and a shuffle control
returns +0.067 — the measured association lies *inside* the null. The perception-gain
oracle captures 0.155 of achievable decision value on KITTI, 0.058 on nuScenes.

**B. Does this remain true when ego speed is controlled?** Yes. Partial Spearman given
speed is +0.042, unchanged. Under a speed-stratified budget the speed heuristic drops
0.787 → 0.285 while η_E stays at 0.213.

**C. Without empty CHEAP frames?** Yes, and more strongly: corr −0.006 and η_E 0.107 on
non-empty frames, corr −0.033 and η_E 0.061 with ≥2 candidates.

**D. With reliable geometry?** Yes. η_E = 0.198 with oracle range, 0.141 with calibrated
noise. The effect shrinks but does not disappear; the per-sequence median does rise to
0.405, which is the honest caveat.

**E. Does it survive a moderate compute gap?** Yes. 512→640: η_E = 0.213 with a 19.1%
action-change rate.

**F. More than one downstream task?** Yes, and the second is starker — η_E = −0.257,
i.e. perception-gain ranking is worse than random for lateral decisions.

**G. Does optimal allocation depend on which task the robot is doing?** Yes, strongly.
Top-10% overlap between the two tasks' optimal allocations is 0.032–0.150; cross-applying
a ranking gives η = 0.087 and −1.544. The shared-cost control returns overlap 1.000.

**H. Does it replicate on a second dataset?** Yes at mini scale: η_E = 0.058 longitudinal
and −0.181 lateral, both stronger than KITTI. The 85-scene confirmation is pending.

**I. Can a trivial heuristic solve most of it?** No. The best 2-variable heuristic
(speed × empty) reaches 0.732 pooled on KITTI, 0.563 under speed stratification, and no
heuristic exceeds 0.30 on nuScenes. All are negative on the lateral task. But this is the
narrowest margin in the report.

**J. Would I invest the remaining CVPR 2027 time in this project?** **Yes** — as a
problem-formulation, benchmark and measurement paper, not as a method paper. The evidence
that perception metrics mis-rank compute allocation is strong, controlled, replicated and
mechanistically explained. The evidence that anyone can *exploit* that gap with a learned
gate is not: Phase 0C showed the best deployable predictor is essentially ego speed plus an
empty-frame flag. Write the paper that establishes the problem and the benchmark; let the
method follow.

## STRONG GO

All seven pre-registered criteria are met, one of them (per-scene robustness) only in the
median rather than universally, and one (second dataset) so far only at mini scale.

> **If I already knew exactly where FULL perception improves detector accuracy, would I
> still need to know something about the downstream decision to allocate compute optimally?**
>
> **Yes.** A perfect perception-gain oracle captures at most 27% — and typically under 20% —
> of the achievable downstream decision-cost reduction, across two datasets, two tasks and
> two fidelity gaps; on the lateral task it is worse than random. And the answer depends on
> which task you are doing: the two tasks agree on only 3–15% of their top-ranked frames.

## What to do next, in order

1. **Finish the nuScenes blob run** (~85 scenes, downloading). This is the only criterion
   resting on insufficient data.
2. **Add a closed-loop or multi-frame cost** so the compounding of bad actions is visible.
3. **One second detector family** to separate a YOLOv8 property from a general one.
4. **Then** write the benchmark paper. Ship the two planners, the cost sweeps, the oracles
   and the artefact controls as the benchmark — the controls are half the contribution.
5. Do **not** start method development. Phase 0C already showed the exploitable headroom
   above trivial heuristics is thin, and nothing in Phase 0D changed that.
