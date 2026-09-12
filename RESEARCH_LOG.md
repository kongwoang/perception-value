# Research Log

Append-only. Newest entries at the bottom. Do not rewrite past entries except to fix
typos — being able to see what was believed *at the time* is the point.

Entry template:

```
## YYYY-MM-DD HH:MM — title
**Objective**
**Changes made**
**Run IDs**
**Observations**
**Problems encountered**
**Current interpretation**
**Next step**
```

---

## 2026-09-11 12:30 — Repository bootstrap, dataset and environment audit

**Objective**
Stand up the Phase-0 repository and establish what this board and this data can
actually support, before writing any experiment code.

**Changes made**
- Found `~/research/risk-aware-perception/` existing but completely empty and not a
  git repository. Nothing to preserve; `git init` and built from scratch.
- Audited the board: Jetson AGX Xavier, L4T R35.6.5, MAXN, GPU at 1377 MHz. Conda env
  `edge` already had torch 2.1.0a0 (CUDA 11.4, working), torchvision 0.16, cv2 4.5.4,
  TensorRT 8.5.2.2. Added scikit-learn 1.3.2 and ultralytics 8.3.40 (`--no-deps`, so
  the existing Jetson torch build is untouched).
- Chose **KITTI tracking** as the dataset. It is the only openly downloadable driving
  set that supplies all three things Phase 0 needs at once: sequences (so splits can
  be by scene and motion features are causal), GT 3D boxes in ego coordinates (so
  criticality is geometric rather than invented), and track IDs (so range rate and TTC
  are measurable). 21 labelled training sequences, ~8 k frames.
- Started the 15.8 GB image archive downloading at ~3.5 MB/s; labels, calibration and
  oxts (2.2 MB / 88 KB / 8 MB) landed immediately and unblocked all geometry work.

**Observations**
- Label census over all 21 sequences: Car 27300, DontCare 18039, Pedestrian 11470,
  Van 3301, Cyclist 1938, Truck 1189, Misc 793, Person 676, Tram 595.
- The rectified-camera -> velodyne -> IMU chain reproduces sane ego-frame geometry: a
  car at camera (x=+9.6, z=21.8) lands at ego (x=+23 forward, y=-10 left), i.e. 10 m
  to the right. Sign conventions verified by unit test rather than by eye.

**Problems encountered**
- The first background download died when its parent session ended, at 226 MB of
  15.8 GB. Replaced with a `setsid` resume loop that re-issues `curl -C -` until the
  content length matches.
- `curl` 7.68 on this image does not have `--retry-all-errors`.

**Current interpretation**
Data and environment are adequate. The open question is whether the *compute* side
can be made to behave — see the next entry.

**Next step**
Profile the two fidelities on the board before committing to a CHEAP/FULL pair.

## 2026-09-11 12:50 — PyTorch eager cannot measure this experiment; moved to TensorRT

**Objective**
Confirm that FULL is meaningfully more expensive than CHEAP on this board. This is
GO criterion 1, and the whole latency/power story depends on it.

**Changes made**
- Wrote `scripts/01_profile_jetson.py` (median/p95 latency, peak GPU memory, INA3221
  rail power sampled at 20 Hz against a measured idle baseline).
- Wrote `scripts/00_build_engines.py` and `src/rap/trt.py`; added a `backend` switch to
  `TwoFidelityDetector` so the same pre/post-processing serves both backends.

**Run IDs**
`20260911_125017_profile`

**Observations**
- In PyTorch eager the median end-to-end latency was **flat**: 25.5 / 24.9 / 24.9 /
  25.5 / 26.9 ms for 320 / 384 / 512 / 640 / 960. GPU rail power over idle was *not*
  flat: 1.36 / 1.67 / 2.11 / 2.56 / 6.07 W. Compute was scaling; wall clock was not.
- Direct diagnosis: a fixed ~19 ms floor for yolov8s and yolov8n alike, unchanged from
  31 kpx to 492 kpx, and only breaking at 1311 kpx. CUDA-event timing agreed with wall
  clock, so this is real GPU-timeline idle, not measurement error — ~225 modules at
  ~85 us of launch overhead each on Xavier's weak CPU.
- After FP16 TensorRT: cheap_320 fell from 19.3 ms to **6.8 ms** end to end.
- The TRT engine reproduces eager FP32 to <0.004 confidence and <0.1 px on boxes, with
  one borderline detection near the 0.10 threshold dropped. Validated on a real-aspect
  image, not on noise.

**Problems encountered**
- `trtexec` invoked via `subprocess.run` from the export script reported "no
  CUDA-capable device is detected" while the identical command worked from a shell.
  Not chased; engine building moved to a standalone shell loop.
- `trt.nptype()` in TensorRT 8.5 dereferences `np.bool`, removed in numpy 1.24. Mapped
  TRT dtypes to torch dtypes directly instead.
- Engine builds take ~11-12 minutes each on this board.

**Current interpretation**
A latency study in PyTorch eager on Xavier would have measured Python, not perception.
Everything downstream — profiling, detection, and the risk-per-millisecond numbers —
runs on the TensorRT engines, which is also the honest deployment path for this board.

**Next step**
Detection over all 21 sequences once the archive finishes extracting.

## 2026-09-11 13:20 — Pipeline validated on a synthetic negative control

**Objective**
Exercise every analysis stage before the images arrive, and find out how the analyses
behave when there is provably no criticality signal to find.

**Changes made**
- `scripts/99_synthetic_pipeline_check.py`: real GT labels, simulated detections whose
  recall depends only on apparent box height. Size correlates with distance, but there
  is no mechanism by which *criticality* governs whether CHEAP fails.
- Added the specificity analysis (same arms against `Value_visual`), the standard-metric
  scoring of the budget selections, and a `feat_n_det` negative control for the pair test.
- Added `H_visual_plus_stakes` (all visual features plus the single scalar
  `feat_crit_sum`) to separate "stakes" from criticality *structure*.

**Observations**
- On the synthetic null, criticality arms still predict `Value_task` at rho ~ 0.31 and
  F -> G gains +0.157 per-sequence Spearman (p = 0.012). This is **not** signal: it is
  scale. `Value_task = sum_i crit_i * err_i`, so a frame's total criticality sets the
  magnitude of its value, and criticality features recover that magnitude while
  complexity features cannot.
- The uncertainty-matched pair test on the same data returns null, as it should:
  rho(d_crit, d_value) = -0.027 (p = 0.17), sign agreement 0.47. Matching on the
  *extensive* uncertainty features (candidate counts, entropy sums) implicitly matches
  scene scale, which is exactly what removes the confound.
- Budget selections scored on the standard detection metric were near-identical across
  all learned policies (0.31-0.33 at 20 %) while differing by 0.2+ on risk — the
  signature of task-specific routing.

**Problems encountered**
- The feature registry was populated as a side effect of computing features, so the
  leakage guard passed vacuously on a table loaded from disk. Now primed at import.
- Matching in 46 raw dimensions admitted 1 pair. Matching now runs in an 8-component
  PCA subspace fitted on the matching variables only.
- `HistGradientBoosting`'s OpenMP threads were contending: leave-one-sequence-out over
  8 folds took 38.6 s. Fanning out over folds with single-threaded workers gives
  **1.5 s** for bit-identical predictions (26x).
- No parquet engine on this board; tables are pandas pickles.

**Current interpretation**
The raw predictability delta from adding criticality is partly a scale artefact and
must not be reported as the headline. The matched-pair test is the decisive evidence,
and it now has a demonstrated negative control.

**Next step**
Run detection on the real frames and produce the real numbers.

## 2026-09-11 14:30 — Real data: the hypothesis is wrong in its original form

**Objective**
Run the full pipeline on all 21 KITTI tracking sequences and decide GO/NO-GO.

**Changes made**
- `scripts/02b_mode_selection.py`: pick the CHEAP/FULL pair on measured evidence rather
  than convention. Ran all five modes over 6 sequences.
- `scripts/08_mechanism.py`: per-object table of which mode detected what, so the
  *reason* for any result is visible rather than inferred.

**Run IDs**
`20260911_140226_profile`, `20260911_140836_modesel`, `20260911_142608_detect`,
`20260911_143838_mechanism`, `20260911_150044_analysis`

**Observations**
- Chose cheap_320 / full_640: 1.58x end-to-end latency, 2.5x GPU inference, and the
  largest value heterogeneity of any pair (36 % zero-value frames, top 20 % of frames
  carrying 86 % of the gain).
- The mechanism analysis overturned the framing. Over 46 469 objects,
  `corr(criticality, recovered by FULL) = -0.188` and `corr(distance, recovered) = +0.289`.
  Extra resolution buys **distant** objects; criticality lives on **near** ones. Inside
  10 m, FULL adds 4 points of recall; at 45-60 m it adds 54.
- So "spend compute where the stakes are high" is false as stated. Value lives in the
  overlap band (~10-45 m, in-corridor, closing) where both gradients are non-trivial.

**Problems encountered**
- The first profile run was contaminated: a TensorRT engine build was still running and
  stealing the GPU (p95 spiked to 63 ms, rail power swung 0.7-8 W). Added a contention
  guard that aborts rather than publishing such a measurement, and a warm-up round that
  is discarded (round 0 differs from rounds 1-3 by up to 50 % on clock ramp).
- The rail-power panel was uninformative because the average includes the ~6 ms Python
  post-process where the GPU idles. Replaced with energy per frame.

**Current interpretation**
GO looks likely, but the paper's story is the *tension* between the two gradients, not
the naive stakes hypothesis.

**Next step**
Sensitivity sweep over every alternative definition of risk.

## 2026-09-11 15:45 — Cache bug found while the sweep was too slow to finish

**Objective**
The sensitivity sweep was running at 13 minutes per configuration — 6 hours for 25.

**Changes made**
- Profiled `build_sequence` instead of guessing. 52 s of every 73 s was
  `numpy.lib.format.read_array`: `DetCache` held a lazy `NpzFile`, and indexing it
  re-inflates the entire array on **every** access — 49 773 array reads for 1 059 frames.
  Materialising the arrays once on construction made table building **8.6x** faster
  (69 -> 8.1 ms/frame) and target-only builds **18x** faster.
- Verified the change is behaviour-preserving before trusting it: rebuilt the full
  8 008-frame table and compared against the stored one, max abs difference 0.0 across
  all 82 numeric columns.

**Run IDs**
`20260911_155416_sensitivity`

**Observations**
- Sweep dropped from ~6 h to 35 min, 25 configurations.
- **23 of 23** non-control configurations show criticality beating uncertainty at a 20 %
  quota; median deta +0.118, minimum +0.063; F->G significant in 22 of 23.
- The two negative controls behave exactly as they must: `uniform` (criticality = 1)
  gives +0.002, `proximity` (criticality = distance only) gives -0.018. The gain appears
  only where criticality encodes ego-path geometry or TTC — information that apparent
  object size, and therefore visual uncertainty, cannot carry.

**Problems encountered**
- Earlier in the session I had assumed feature extraction was the bottleneck and added a
  feature cache to the sweep on that assumption. The cache is still useful, but it was
  not the problem; profiling first would have saved an hour.

**Current interpretation**
GO. The sensitivity result is mechanistic rather than merely robust, which is a stronger
claim than the one Phase 0 set out to test.

**Next step**
Phase 1 as proposed in `docs/cvpr_phase0_findings.md`: factorise the gate into a
failure-probability head and a criticality head and predict their product, rather than
regressing `Value_task` end to end.

---

## 2026-09-11 17:05 — Phase 0B opens: is failure the same thing as recoverability?

**Objective**
Phase 0 established that criticality carries information uncertainty does not, and
falsified the naive "high stakes → more compute" mechanism. It left a specific,
testable successor hypothesis, which Phase 0B exists to kill or confirm:

    Value(frame) ≈ sum_i  c(i) · P(cheap fails on i) · P(full recovers i | cheap fails)

The claim is that modelling **recoverability** separately from **failure** generalizes
better than a monolithic frame-level regressor — and, more sharply, that it beats the
obvious cheap baseline `uncertainty × criticality`. If that baseline already matches the
three-factor decomposition, there is no method contribution here and the paper is a
problem-formulation and benchmark paper instead.

Phase 0 evidence that motivates it: extra compute recovers *distant* objects
(corr(distance, recovered) = +0.289) while criticality concentrates *near*
(corr(criticality, recovered) = −0.188). Value therefore lives in an overlap band that
neither factor alone identifies. Whether a gate can *predict* that band from cheap
output is exactly what "recoverability" means, and it is unmeasured.

**Execution order (deliberately not parallel)**
The kill test runs first and entirely on existing KITTI data: object-level fail/recover
labels, then `uncertainty × criticality` versus `fail × recover × criticality` at matched
compute. Only if recoverability adds measurable value do moderate fidelity pairs, a
second detector family and a second dataset follow.

**Open design question to resolve before any modelling**
A deployed gate scores frames from CHEAP output, so a per-object score has to be anchored
on something CHEAP produces. Ground-truth objects that CHEAP misses *entirely* — no
candidate box at any score — are invisible to such an anchor. The fraction of recovered
risk that sits on those objects is an upper bound on what an object-anchored factorized
model can ever capture, and it is the first thing to measure.

**Next step**
Measure that coverage, then build the object-level table.

**Do not claim** (carried forward from Phase 0)
Uncertainty is real signal, not useless. Critical frames do *not* simply deserve more
compute. Per-frame criticality signal is weak (matched-pair rho ~ 0.1); the effect is an
aggregate-allocation effect.

---

## 2026-09-11 17:40 — Phase 0C opens: is decision value a distinct target from perception value?

**Objective**
Phase 0B killed the three-factor decomposition (`p_recover` inert, median dEta −0.002,
p = 0.73) and found the object-anchored formulation structurally blind to ~33% of
recoverable risk. Generic criticality-aware scheduling also has close prior work. So the
formulation itself is being replaced rather than patched.

The new question is not "is this frame hard" or "is this frame critical" but:

    if I spend more perception compute on this frame, does the DOWNSTREAM DECISION
    actually improve?

    V_dec(t) = J(pi(z_cheap), s*) - J(pi(z_full), s*)

**Hypothesis under test**
`V_dec` is not fully explained by visual uncertainty, scene complexity, task criticality,
or even by *oracle* perception gain. In short: "better perception" != "better decision".

**Why this might be false, and the honest prior**
Phase 0 showed extra resolution mainly recovers *distant* objects (mean 33 m) while a
braking controller is governed by *near* ones, where CHEAP already reaches 0.80 recall.
The plausible null is therefore that CHEAP and FULL almost never produce different
actions, which is an explicit NO-GO condition. I expect a low action-change rate and the
kill test is designed to surface that immediately rather than after a modelling effort.

**Anti-circularity constraint**
The decision cost must NOT be `sum criticality x detection error` — that is the Phase-0
metric renamed. The controller emits an ACTION; the action is scored against GT scene
geometry through an asymmetric cost (under-braking risk vs unnecessary braking, progress,
jerk). GT geometry appears only in the evaluator, never in the deployed gate.

**Kill test (before anything else, per the plan's execution order)**
One deterministic longitudinal braking controller, identical code for both modes, run on
the existing KITTI CHEAP/FULL cache. Report: % frames where perception differs, % where
the ACTION differs, % where detection improved but the action did not, corr(U, dJ),
corr(C, dJ), corr(dE, dJ), and eta for uncertainty / criticality / dE-oracle / dJ-oracle.

**Stated NO-GO conditions**
Stop and report NO-GO if dJ is a monotone function of criticality, or of dE, or if
cheap/full almost never change the decision, or if the planner is so insensitive that
extra perception rarely matters. The single decisive comparison is dE-oracle versus
dJ-oracle at matched compute: if knowing exactly where perception improves already solves
allocation, this direction closes.

**Note on Phase 0B deliverable**
`docs/cvpr_phase0b_findings.md` was never written: Phase 0B stopped at its kill test by
design, and its interim report is `docs/reports/phase0b_killtest.html`. The generalization
tests in the Phase 0B plan (second dataset, second detector, moderate fidelity pairs)
remain unrun.

## 2026-09-11 18:30 — Phase 0C result: the problem is real, the mechanism is not

**Run IDs**
`20260911_181039_decision`, `20260911_181639_decision_budget`

**Observations**
- 27.3 % of frames get a different braking action from CHEAP vs FULL; 17.0 % beneficial,
  **10.3 % harmful**. 67 % of frames where detection improved keep the same action.
- `corr(perception gain, decision gain) = +0.041`, and a shuffle control gives +0.067 —
  the association is *inside the null*. An oracle on perception gain captures 15.5 % of
  achievable decision-cost reduction at a 20 % quota, indistinguishable from random.
- Criticality — the Phase-0 contribution — is worthless at the decision level
  (eta = 0.009, below random), in all ten planner and cost variants.
- The proposed mechanism is **false**: rho(distance to action threshold, |dJ|) = +0.006,
  non-monotonic. Decision margin as a feature is at chance (AUC 0.495).
- Ablation: removing ego speed collapses the learned model from eta 0.831 to **−0.011**.
  Ego speed alone reaches 0.787; "cheap detected nothing" alone reaches 0.560.
- 14.6 % of frames have no CHEAP detection at all and carry **57.9 %** of all positive
  decision gain. The dominant decision failure is blindness, not degraded localisation.

**Problems encountered**
- First diagnostic looked too strong, so I checked whether `dJ` was noise before believing
  it: lag-1 autocorrelation +0.286 (not white), fixed-action control gives V_dec exactly 0,
  and the shuffle control behaved correctly. It is signal.
- Monocular range error inflates the effect. Recomputing with GT range for matched
  detections drops action changes 27.3 % -> 17.2 % and harmful changes 10.3 % -> 4.9 %;
  `corr(dJ_mono, dJ_oracle_range) = +0.688`. About 40 % of the action churn is range noise.
- The `confidence (low first)` baseline scored suspiciously high (0.631) because
  `feat_conf_mean` is 0 on empty frames, so it ranks blind frames first. Ran it down rather
  than reporting the number: excluding empty frames it drops to 0.121.

**Current interpretation**
GO on the problem, NO-GO on the method. "Perception metrics do not predict decision value"
is a clean, quantified, well-controlled result. But the exploitable signal is ego speed and
an empty-frame indicator, neither of which is a perception contribution, so there is no
method paper here yet.

**Next step**
Either a problem/benchmark paper (needs a second downstream task, a second dataset and a
closed-loop variant), or investigate what predicts *complete* cheap-perception failure —
the 14.6 % blind frames are where the remaining headroom lives.

---

## 2026-09-11 18:45 — Phase 0D pre-registration: validating the PROBLEM, not a method

**Core claim to test**

    Marginal improvement in perception quality is not a reliable proxy for marginal
    improvement in downstream decision quality.

    dE_t = E(cheap_t) - E(full_t)
    dJ_t = J(pi(cheap_t), s*_t) - J(pi(full_t), s*_t)

    Hypothesis: dJ is not reducible to dE, uncertainty, criticality, or trivial
    operating-state variables.

**Three properties, tested separately. All three must hold.**
- DISTINCT: dJ differs meaningfully from perception-level quantities.
- GENERAL: it happens across tasks, planners and datasets.
- NON-TRIVIAL: it is not explained almost entirely by ego speed, empty detections, or one
  obvious heuristic.

**Pre-registered falsifiers (recorded before running anything)**
1. dE-oracle nearly matches dJ-oracle -> better perception *is* better decisions.
2. Longitudinal and lateral optimal rankings nearly identical -> not task-conditional.
3. Effect disappears on non-empty frames or a moderate fidelity pair -> cheap-collapse artefact.
4. Effect disappears with oracle/controlled geometry -> monocular range artefact.
5. Effect absent on a second dataset -> does not generalize.
6. Speed + empty indicator explains ~all oracle value everywhere -> trivial.

**Threshold committed in advance**
STRONG GO needs eta_E@20 <= 0.5 in the main configurations, survival of the speed /
non-empty / geometry controls, survival at a moderate fidelity pair, replication on a
second dataset, top-20% overlap between task rankings well below 80%, and no 1-2 variable
heuristic capturing almost all oracle value.

**Prior expectation, stated now so it cannot be rationalised later**
Phase 0C already showed ego speed alone reaches eta 0.787 of the learned 0.831, and that
14.6% empty-CHEAP frames carry 57.9% of positive dJ. My honest prior is that Stage 1 will
downgrade the problem substantially: I expect the mismatch to survive as a *statement*
(dE really is uninformative about dJ) but the *allocation opportunity* to shrink a lot once
speed and empty frames are controlled. The verdict will follow the numbers.

**Execution order**
Stage 1 on existing KITTI only: reproduction, speed-controlled, non-empty, oracle-range,
moderate fidelity pair. Interim report before any lateral planner is written. No method
development anywhere in Phase 0D.

## 2026-09-12 00:30 — Phase 0D verdict: STRONG GO on the problem

**Run IDs**
`20260911_223528_stage1`, `20260911_224615_stage2`, `20260911_233520_nusc_decision`,
`20260912_002239_final_matrix`

**Observations**
- Every one of 12 configuration x task rows has eta_E@20 <= 0.271 against a
  pre-registered threshold of 0.5. Most are below 0.22; the lateral task is negative.
- Removing the obvious artefacts makes the mismatch *stronger*, not weaker: non-empty
  frames eta_E 0.107, >=2 candidates 0.061.
- Partial Spearman(dE, dJ | ego speed) = +0.042, unchanged from raw. Under a
  speed-stratified budget the speed heuristic falls 0.787 -> 0.285.
- Second task (lateral avoidance): FULL is more accurate (83.8% vs 81.9%) yet more
  costly (2437 vs 2170) in all 10 planner/cost variants, because extra detections block
  clear corridors. eta_E = -0.257: perception-gain ranking is worse than random there.
- Task conditionality is the strongest novelty signal: top-10% overlap between the two
  tasks' optimal allocations is 0.03-0.15; the shared-cost control returns exactly 1.000.
- nuScenes mini replicates and more strongly (eta_E 0.058 / -0.181). The KITTI ego-speed
  shortcut does NOT replicate (0.210 vs 0.787).

**Problems encountered**
- nuScenes `sample_annotation.translation` is the box CENTRE, not the bottom-face centre
  as in KITTI. Treating it as the bottom shifted projected 2D boxes up by h/2 and left
  0.5% of detections matching ground truth; after the fix, 63.4%. Caught because the
  oracle-range variant returned numbers identical to mono, which is impossible if
  matching ever succeeds.
- The first lateral planner was degenerate (GT chose KEEP on 97% of frames). Parameters
  were then chosen by measuring the clearance distribution rather than guessing.

**Current interpretation**
STRONG GO for a problem/benchmark paper. The evidence that perception metrics mis-rank
compute allocation is strong, controlled, replicated and mechanistically explained. The
evidence that the gap is *exploitable* is not - Phase 0C already showed the best
deployable predictor is ego speed plus an empty-frame flag.

**Next step**
Finish the nuScenes trainval01 blob (~85 scenes, downloading) and re-run Stage 4 on it;
that is the only criterion currently resting on insufficient data.

---

## 2026-09-12 00:40 — Phase 0E pre-registration: final experimental validation

**Purpose**
Decide whether *Decision-Conditional Value of Perception Compute* survives the strongest
reasonable attacks. No paper, no LaTeX, no method, no scheduler. A negative result is
preferable to a weak paper.

**Claim under attack (unchanged from 0D, restated)**

    dE_t      = E(z_c,t) - E(z_f,t)
    dJ_t^(q)  = J_q(pi_q(z_c,t), s*) - J_q(pi_q(z_f,t), s*)

    Marginal perception improvement is not a reliable proxy for marginal downstream
    decision improvement, and the value of extra compute is TASK-CONDITIONAL:
    V = V(scene, downstream objective) rather than a scalar property of the frame.

**Validation dimensions; no claim may rest on one of them**
dataset, detector family, fidelity gap, downstream task, perception metric, geometry
quality, temporal evaluation.

**Eight falsifiers, recorded before the new results exist**
- F1 a rich multi-metric perception oracle reaches eta@20 > 0.7 consistently.
- F2 a second detector family eliminates the gap.
- F3 full-scale nuScenes eliminates it.
- F4 a moderate fidelity gap eliminates it.
- F5 temporal evaluation eliminates it.
- F6 longitudinal and lateral rankings converge (top-20% overlap > 0.8 consistently).
- F7 one trivial heuristic reaches eta > 0.9 across configurations.
- F8 better geometry makes dE nearly sufficient (eta_E approaching 1.0).

**What I expect to be the weakest points, stated now**
(1) The trivial-heuristic margin. On KITTI `speed x empty` already reached 0.732 pooled;
it fell to 0.563 under speed stratification and 0.297 on nuScenes, but this is the
narrowest of the 0D margins and Stage 11 may narrow it further.
(2) The multi-metric perception oracle (Stage 3) is a genuinely new attack that 0D never
ran. If a rich description of *how* perception changed predicts dJ well, the claim is
substantially weaker, and I consider this the single most likely way Phase 0E ends in a
downgrade.
(3) Better geometry already raised the KITTI per-sequence median eta_E from 0.159 to
0.405; a detector with better precision might narrow the lateral result too.

**Note on missing prior deliverables**
`docs/cvpr_phase0b_findings.md` was never written - Phase 0B stopped at its kill test by
design and its interim report is `docs/reports/phase0b_killtest.html`. `EXPERIMENTS.md`
has never existed in this repository; per-run provenance lives in `results/raw/*/config.json`
plus `environment.json`, and the narrative lives here.

**Execution order (not parallel)**
0D reproduction -> ~85-scene nuScenes -> perception-metric robustness and the multi-metric
oracle -> Detector-B on KITTI -> Detector-B fidelity pairs -> Detector-B on nuScenes ->
temporal replay -> geometry/task/fidelity controls -> Jetson profiling -> final matrix and
statistics -> findings report. Stop and report if a falsifier fires.

## 2026-09-12 07:40 — Phase 0E verdict: WEAK GO

**Run IDs**
`20260912_011500_nusc_tv`, `20260912_012320_percep_metrics`, `20260912_014735_temporal_kitti`,
`20260912_071140_core_matrix`, `20260912_073405_finalize`

**Observations**
- 16 configuration x task rows. eta_perception_oracle@20 never exceeds **0.213**; negative
  in 6 of 16. 60 of 64 sequence-level Wilcoxon tests significant after Holm.
- F2 does not fire and is the strongest result: RT-DETR-l (set prediction, no NMS, 32.1M
  params) shows eta_E 0.046 / 0.088 with corr(dE,dJ) ~ 0.005, stronger than YOLOv8s.
- F3, F4, F5, F8 do not fire. Moderate 512->640 (1.18x GPU compute) still gives 0.213;
  temporal replay gives 0.162 against 0.155 per-frame.
- F1 does not fire on its stated condition but is the closest call: the multi-metric
  perception oracle reaches 0.700 on KITTI/YOLOv8/320->640 longitudinal and 0.832 with
  oracle range, while staying <= 0.434 in the other fourteen rows.
- F6 fires in 2 of 16: nuScenes oracle-range top-20 task overlap 0.803 vs a 0.80 threshold.
- F7 does not fire: best trivial heuristic 0.864 < 0.90, and the winning heuristic changes
  in every row (six different ones win somewhere).
- On the lateral task FULL is more accurate but more costly in all 10 planner/cost
  variants; 11.7% of KITTI frames get a *worse* decision from better perception.

**Problems encountered**
- The core-matrix run was killed by a session teardown and left an empty run directory;
  rerun cleanly.
- An `until ! pgrep -f "01_profile_jetson"` loop deadlocked because the wrapper's own
  command line contains the pattern - the same self-match class as the earlier
  `pkill -f firefox`.
- The profiler only globbed `*.png`, so nuScenes (`.jpg`) profiling failed until fixed.
- Two counts in the first draft of the report were wrong (negative rows, multi-metric
  row count); caught by the verification pass and corrected before commit.

**Current interpretation**
WEAK GO, not STRONG GO. The phenomenon is weakest exactly where the setup is most
artificial (KITTI + YOLOv8 + aggressive gap + oracle geometry) and strongest in the more
realistic configurations. That is reassuring scientifically, but it means a paper must lead
with RT-DETR and the moderate gaps rather than the flagship KITTI configuration.

**Next step**
No further experiments. If the paper is written, it is a problem-formulation and benchmark
contribution; the open weakness is that both planners are rule-based and open-loop.

---

## 2026-09-12 08:20 — Phase 0F pre-registration: literature-aware validation

**Why this phase exists**
Phase 0E returned WEAK GO against *task-agnostic* perception metrics. That is no longer the
right bar. Several ideas this project treated as its own are prior knowledge:

- selective perception / active vision (Reece & Shafer 1995, Ulysses-2)
- Value of Computation / rational metareasoning
- PKL, planner-centric perception evaluation (CVPR 2020)
- planning-aware prediction/detection evaluation (Ivanovic & Pavone, IV 2022)
- TIP, expected-utility planner-aware decomposition (ICML 2023)
- task-aware risk estimation (Antonante et al., RSS 2023, PERSEVERE)
- adaptive perception compute (DNN-SAM, Self-Cueing, CA-MOT, EneAD, VLA pruning)

**Claims this project must NOT make**: first to make perception task-aware; introducing
decision-aware perception; introducing value of computation; "perception accuracy does not
imply planning quality". All prior.

**The narrow claim actually under test**

    Planning-aware *evaluation* and adaptive *inference* have been studied separately.
    When choosing WHICH INPUTS receive additional neural perception compute, even exact
    perception improvement -- and possibly existing planning-aware scores -- may fail to
    rank inputs by marginal downstream decision benefit.

    relevance of an error  !=  marginal value of correcting it with a particular computation

**The kill test, stated before any result**
Hard Kill Test 1: if eta_PKL >= 0.8 or eta_TIP >= 0.8 at a 20% quota consistently across
downstream tasks, the thesis is substantially redundant. STOP and report, do not proceed to
closed loop.

Stage-1 verdict bands, fixed now: NO-GO if a planning-aware metric consistently reaches
>= 0.8 of the decision oracle; WEAK GO at ~0.6-0.8; STRONG EMPIRICAL GAP if planning-aware
scores beat standard perception metrics but stay < 0.6 under held-out scenes and moderate
fidelity.

**Honest prior expectation**
PKL and TIP are built to score how much a perception error matters to a planner. That is a
*relevance* question, and my Phase 0C-0E results say relevance and marginal-value-of-a-
specific-computation come apart. So I expect them to beat standard dE and still fall short
of the decision oracle. But they were designed by people who thought carefully about
exactly this, and Phase 0E already showed a rich perception-only oracle reaching 0.70-0.83
in the flagship KITTI cell, so a PKL/TIP score landing above 0.8 on nuScenes is entirely
plausible. If it does, the project stops.

**Execution priority**
P0 PKL on existing nuScenes detections; P0 TIP on the same; P0 the diagnostic table against
existing decision targets; P0 ranking/regret analysis. P1 independent Planner B, good
geometry, operating-point control. P2 closed loop. No method development at any point.

**Rule for prior code**: use official implementations, never reimplement from memory; record
repository URL, commit, dependency versions, pretrained weights, split, and any modification;
do not alter their scoring definitions; document every sign convention explicitly.

---

## 2026-09-12 16:45 — Phase 0G pre-registration: planner-conditionality falsification

Written before any Planner D training and before any Phase 0G allocation number exists.
Branch `exp/phase0g-planner-conditionality`. Paper title: *The Decision Value of Perception
Compute*. This is **not** a method phase: no allocator is built.

### The claim under test

> The marginal downstream value of additional perception compute is conditional on the
> downstream decision maker, and an allocation score that works for one planner need not
> preserve the valuable-input ranking for another.

with `V_i^q = J_q(pi_q(z_i^cheap), s_i) − J_q(pi_q(z_i^full), s_i)`. The question is whether
`q` changes the **ordering** of `V_i^q`, not its scale.

Phase 0F left PKL at η@20 = 0.109 for Planner A and 0.872 for Planner C, but Planner C is
near-circular with PKL (PKL is a divergence of the planner heatmaps; J_C is displacement of
those same heatmaps' argmax). Phase 0G fills the missing quadrant — a learned planner that is
*independent* of PKL — and adds external planners we did not write.

### Frozen now: split

`configs/phase0g_scene_split.json`, committed before training. Test = the 85 Phase-0F scenes
verbatim; the remaining 765 ordered by (location, name) with every 8th within each location to
validation. No RNG. **train 667 scenes / 26,828 keyframes · val 98 / 3,945 · test 85 / 3,376**,
verified pairwise disjoint by assertion. The 85 test scenes are never used for training, early
stopping, architecture selection, augmentation choice, normalisation fitting or
hyperparameters.

**Distribution shift recorded in advance**: the test scenes are only boston-seaport (26) and
singapore-onenorth (59) — the trainval01 blob contains no hollandvillage or queenstown test
scenes — while training is 58% boston-seaport and 16% onenorth. Planner D viability will
therefore be reported both on all validation scenes **and** restricted to validation scenes in
those two locations. This is a property of which blob is on disk, not a choice.

### Frozen now: Planner D

Input: the same 5-channel BEV raster the released PKL planner consumes, built by calling
`planning_centric_metrics.planning_kl`'s own `samp2ego`, `samp2mapname`, `get_local_map`,
`get_other_objs`, `raster_render` — so C-vs-D is a planner comparison, not a representation
comparison. No PKL weights, layers, distillation, or PKL/TIP scores are used in training.

Architecture, frozen: four conv blocks (3×3, stride 2, GroupNorm, ReLU) at 32/64/128/256
channels, adaptive global average pool, MLP 256→256→32, output 16×2 waypoints. No heatmap, no
classification over cells, no attention or recurrence. Three training seeds.

Target: the **real** future ego trajectory at 0.25…4.00 s in 16 steps, expressed in the current
ego frame using PKL's own `objects2frame`, from official ego poses with interpolation over
timestamps. Loss: MSE over the 16×2 coordinates. Never trained against Planner C output.

Two variants, both frozen: **D-GT** (GT rasters only) and **D-Aug** (same rasters with fixed
corruption — object dropout 0.10, translation jitter σ = 0.50 m, size jitter σ = 0.10
multiplicative, heading jitter σ = 5°). These corruption values are stipulated, not fitted to
the Phase-0F test detections, and are not tuned on η. D-Aug exists only to test whether D-GT
results are an artefact of train/test shift.

Viability gate, checked **before** any cheap/full evaluation: validation ADE at least 10% below
a constant-velocity / constant-heading predictor; FDE also reported. Architecture may change
only while looking at validation ADE/FDE, with every change committed. Once cheap/full
evaluation begins, everything is frozen.

Primary Planner-D cost is error against the **real** future trajectory,
`J_D^ADE(mode) = mean_t || pi_D(raster_mode)_t − y_true_t ||`, with
`V^D = J_D^ADE(cheap) − J_D^ADE(full)`. Secondary: FDE, and self-consistency against
`pi_D(raster_GT)` — secondary because it is planner-internal, the same weakness that makes
Planner C near-circular.

### Frozen now: quotas, statistics, falsifiers

Quotas 10/20/30/50%, pooled top-quota selection. At least 400 scene-level bootstrap draws,
never frame-level; degenerate draws dropped and counted. Absolute cost reductions reported
alongside η. Holm correction on the pre-declared family of paired tests.

Planner D falsifiers: **D-F1** PKL or TIP reaches η@20 ≥ 0.80 on *both* D-GT and D-Aug under
real-trajectory ADE with a non-degenerate CI → the "PKL is tied to its own planner" argument is
substantially weakened. **D-F2** C-vs-D top-20 oracle overlap ≥ 0.80 *and* cross-planner η@20 ≥
0.80 in both directions, for both D variants → controlled evidence for planner conditionality
fails. **D-F3** fewer than 5% of affected frames have V_D < 0 for both variants → Planner D is
not evidence for sign-varying value. **D-F4** Planner D fails the constant-velocity baseline →
its allocation result is not interpreted.

External falsifiers: **B-F1** both external planners have P(V<0 | V≠0) < 0.05 → sign variation
does not replicate externally. **B-F2** for both planners oracle@20 gains < 10% extra over
all-FULL relative to the oracle's achievable improvement → selective allocation is weak
externally. **B-F3** PDM-vs-IDM top-20 overlap ≥ 0.80 and both cross-η@20 ≥ 0.80 → planner
identity does not alter the valuable-state ranking externally. **B-F4** one published
planning/task-aware score reaches η@20 ≥ 0.80 for both external planners → benchmark novelty
downgraded.

A single number will not end the phase: if PKL reaches ~0.85 on Planner D, the decision also
requires that it holds for both D variants under real-trajectory ADE, that C-D oracle overlap
is high, that cross-planner η is high, and that the transfer survives an external target.

### Deviation declared in advance: the Track B perception intervention

Track B wants PDM-Closed and IDM driven by a real YOLOv8s 320→640 fidelity pair. Measured
today: nuPlan mini db + maps is 9.5 GB and downloads in ~26 min at 6.03 MB/s, but the mini
**camera** blobs are 9 shards of 45–54 GB (~450 GB) against 138 GB of free disk, and shards are
split by blob rather than by log, so an arbitrary shard may not contain complete 15 s scenario
windows at all. PDM-Closed and IDM consume tracked objects from the db, not images.

Declared plan, in this order:

1. **Track A first** — it is in-domain, uses the real detector, has 3,376 frames, and changes
   exactly one variable.
2. **Track B on all of nuPlan mini with a *measured-transfer* intervention.** Rather than
   inventing a corruption model, fit `P(detect | range, image-space size, class, truncation)`
   and the false-positive distribution separately for YOLOv8s at 320 and at 640 from the ~11k
   KITTI + nuScenes frames already computed, and apply that measured function to nuPlan tracked
   objects. This is a measurement of the real detector transported to a new domain, not a
   stipulated corruption. Stated limitation: it assumes the miss profile transfers from
   nuScenes CAM_FRONT to nuPlan's front camera across different intrinsics, resolution and city.
3. **Then one camera shard** (~50 GB, ~2.3 h) as a real-detector spot check on whatever subset
   it covers, if step 2's coverage holds up.

Steps 2 and 3 have *opposite* weaknesses — coverage versus in-domain realism. Agreement between
them makes the deviation immaterial; disagreement is itself a reportable finding. Neither is
presented as satisfying B4 as written.

### Honest prior expectation

Planner C's 0.872 is mostly internal consistency, so I expect PKL to fall well below it on
Planner D. But Planner D shares PKL's *input representation* and its training data domain, and
both planners are ultimately predicting where the ego goes — so a high PKL η on Planner D is
entirely possible, and would mean the planner-conditionality claim is about objectives rather
than architectures, or is wrong. If D-F1 and D-F2 both fire, the paper is reframed or stopped.
