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

### 2026-09-12 17:20 — Phase 0G amendment, before any training run

Two decisions forced by measurement, recorded before Planner D sees any data.

**1. Frames whose 4 s horizon runs past the end of the scene are excluded.** Measured on two
validation scenes: **21% of samples**. A nuScenes scene is ~20 s (40 keyframes at 0.5 s), so the
last eight samples of every scene have less than 4 s of future left. `interp_poses` clamps such
queries to the final pose, which makes the target a *stationary* ego — training on it would
teach Planner D to predict stopping, and evaluating on it would compare planners against a
fabricated trajectory. These frames are dropped from training, from validation and from the
**primary** Planner D evaluation, and their count is reported.

Consequence for comparability, accepted deliberately: the Planner D test set becomes ~2,670 of
the 3,376 frames. Planner A and Planner C are therefore **re-evaluated restricted to exactly
that subset** wherever they are compared with Planner D, so the comparison is like-for-like
rather than across different frame sets. The full-3,376 Phase 0F numbers remain as published.

**2. Training scenes capped at 200 of the 667, by measured throughput.** Rendering costs
0.41 s per raster (startup excluded), so the full plan would be ~5 h of rendering before a
single training step: 26,828 train + 3,945 val + 3,376×3 test rasters. The pre-registration
permits a cap "decided by measured throughput before training"; this is that decision. The 200
scenes are taken deterministically as every third scene of the 667 in (location, name) order, so
location balance is preserved and the choice involves no RNG and no result. That is ~8,000
training samples for a 0.46 M-parameter network, and validation and test remain complete.

Neither decision is informed by any allocation number, η, or PKL/TIP result; no model has been
trained at this point.

### 2026-09-12 20:35 — Phase 0G: D-F4 fired, architecture changed under the A7 rule

The pre-registered Planner D architecture **failed its viability gate decisively**, and the
change made in response is recorded here before any allocation number is looked at.

Observed on D-GT seed 0, validation only:

| epoch | train MSE | val ADE | val FDE |
|---|---|---|---|
| 0 | 33.79 | 6.363 m | 12.124 m |
| 4 | 13.79 | 5.222 m | 10.036 m |
| 8 | 7.88 | 5.076 m | 9.763 m |
| 13 | 4.83 | 5.043 m | 9.702 m |
| — constant-velocity baseline — | | **1.467 m** | 3.828 m |

Training MSE fell by 7× while validation ADE plateaued around 5.0 m, three and a half times
*worse* than constant velocity. The network was fitting what it could see, and what it could
see was insufficient.

**Debugging came first, as A7 requires, and found no implementation fault.** The extracted
data was verified independently: monotone-forward fraction 1.000, final-waypoint lateral offset
symmetric about zero (mean +0.16 m), zero clamped frames retained, and the correlation between
the 4 s waypoint and 4 s × measured ego speed is **0.961**. The coordinate transform is
asserted equal to PKL's own `objects2frame` in `tests/test_ego_traj.py`.

That last number is also the diagnosis: **ego speed explains the target almost completely, and
the 5-channel BEV raster does not contain it.** The ego appears as a static footprint; there is
no velocity channel. The pre-registered Planner D was asked to regress a trajectory whose
dominant factor was withheld, while its baseline was handed exactly that factor.

**Change, under A7:** the ego's own velocity (2 numbers, the cached `ego_v`) is concatenated to
the pooled BEV feature before the MLP head. The convolutional encoder and the raster are
untouched, so the *perception* representation still matches PKL's exactly; what is added is
proprioception, which Planner A already uses and which every real planner reads off the CAN bus.

Why this cannot contaminate the study: **ego velocity is identical under CHEAP and FULL.** It
shifts `J_D(cheap)` and `J_D(full)` together and leaves `V_D = J_D(cheap) − J_D(full)` driven
only by the difference between the two rasters. It also makes the viability gate meaningful
instead of rigged: Planner D now has exactly the information the constant-velocity baseline has,
so beating that baseline by 10% means the *scene* contributes at least that much beyond
kinematics — which is the thing "did it learn a planning function" should test.

No allocation metric, η, PKL/TIP score or test-split quantity was inspected in making this
change; only validation ADE/FDE, as A7 permits. All six training runs restart from scratch with
the new architecture, which is frozen from here.

### 2026-09-13 03:05 — code review: five defects, and what has to be recomputed

Full audit at the request of the user, before any further runs. Five real defects; three change
published numbers. Fixes are committed, reruns deferred.

**D1 — an arbitrary tie-break decided the majority of a selection.** `select_pooled` broke ties
by row order. Count-valued signals have enormous tie groups: at a 20% quota over 3,376 nuScenes
frames, only **326** frames are strictly above ΔE's cut value while **366** share it, so **349 of
the 675 selected frames (52%)** were chosen by row order, not by the signal. So η@20 for `dE` and
`E1_fn_only` was mostly not a property of the signal — this touches ΔE 0.147 (oracle
longitudinal), 0.047 (mono), 0.317 (Planner C) and the 0.254 attributed to `best_dE_metric
(E1_fn_only)` under Planner C vs truth. At a 10% quota only 4% of the selection is tied, so those
numbers are much less affected. Continuous signals (E6, uncertainty, PKL, TIP) had a tie group of
exactly 1 and are unaffected. Fixed: ties broken by a seeded random key, averaged over 8 seeds,
with the tie fraction reported next to every η.

*Methodological note worth keeping:* an independent hand-computed pipeline reproduced these same
numbers exactly, because it shared the same stable-argsort convention. Agreement between two
implementations cannot detect an arbitrary convention they both use.

**D2 — the same defect, worse, in the score-free cross-target measurement.** The braking
controller responds on only 219 of 2,655 frames, so its top-q sets are mostly tied at ΔJ = 0:
18% of the set at a 10% quota, **59% at 20%**, **73% at 30%**. Both systems broke those ties by
row order and therefore selected the *same early frames*, manufacturing overlap out of nothing.
That is the likeliest source of **overlap@30 = 0.438 vs chance 0.300, paired +0.139 [+0.044,
+0.238]** — the only apparently significant result in that table. **overlap@10 = 0.105 vs chance
0.100 is not contaminated** and the "indistinguishable from unrelated at tight budgets"
conclusion stands on it. Fixed: random tie-breaks with a different seed per system, and the
responsive fraction of each top-q set is now reported so a statistic computed on mostly-tied sets
cannot be read as a property of the systems.

**D3 — the monocular lift wrote the near-face range as the box centre.** `geo["z"]` is
`range_ground(y2)`, the distance to the box's ground contact, i.e. the object's near face;
nuScenes `translation` is the centre. Every lifted box therefore sat ~L/2 too close — **2.3 m for
a car, 5.6 m for a bus**, so the error scaled with class. The same `geo["z"]` is *correct* where
the braking controller consumes it as a gap to the nearest point, which is why it survived four
phases. Consequence: **every submission on disk is stale** — completely for the `mono` variant,
and in its false-positive boxes for the `oracle` variant, since matched detections there inherit
ground-truth boxes and only unmatched ones are lifted.

**D4 — pixel→metre inversion off by half a cell.** `get_grid` returns `bx = lower + dx/2` and the
forward mapping is `pts = round((poly − lower)/dx)`, so the inverse is `poly = pts·dx + bx − dx/2`.
Using `bx` left a +0.15 m bias on both axes. It cancels exactly in `66`, which only differences
two paths on the same grid, but not in `74`, where paths are compared with the real trajectory and
ADE is a norm.

**D5 — reporting: deployable and diagnostic signals were never separated.** `G_PKL = pkl_cheap −
pkl_full` needs the expensive output, so PKL and TIP as used here are **diagnostics, not candidate
allocators**; the same is true of every ΔE variant and the multi-metric oracle. And `crit_sum` in
the decision table comes from **ground-truth** geometry — a different quantity from the cheap-side
`feat_crit_sum` in `features.py`, which is correctly labelled `cheap_det`. Presenting them
side-by-side as comparable heuristics overstated what is deployable. Under mono the best
*deployable* signal is cheap-detection uncertainty at 0.198 [+0.01, +0.35], the only one whose
interval excludes zero.

**Lesser notes.** The leakage registry is populated as a side effect of computing features rather
than statically, so `assert_no_leakage` fails safe but can only catch *unregistered* columns, not
a column whose source label is wrong. `features.py:66` is dead (registration already asserts
legality). `75` does not report dropped bootstrap draws, while `62` does.

**Rebuild chain, ~11 h of compute, deferred.** Submissions (~15 min) → PKL/TIP oracle + mono, 24
chunks (~5 h) → Planner C oracle + mono, 12 chunks (~2.6 h) → Planner D test rasters oracle +
mono, 12 chunks (~2.4 h) → the η, transfer and cross-target analyses (~1 h). The stale mono test
rasters built from the old submissions were deleted rather than kept.

## 2026-09-13 23:55 — Benchmark pre-registration: frozen splits and one protocol

### Why

§6 described a benchmark that did not exist yet: no track had a split, and every baseline number
came from a different run with its own tie handling, bootstrap count and frame pool. Everything in
this entry is committed before any baseline is re-run on these splits.

### Frozen now: splits

`configs/benchmark_splits.json`, generated by `scripts/90_benchmark_splits.py`. One rule, no RNG:
sort a track's units by (stratum, name); the unit at position i goes to train if i mod 10 ∈ 0–4,
val if 5–6, test if 7–9.

| track | unit | stratum | train / val / test units | frames or states (train / val / test) |
|---|---|---|---|---|
| nuScenes | scene (the 85 with cheap+full detections) | location | 45 / 16 / 24 | 1,785 / 636 / 955 (trajectory truth: 1,404 / 499 / 752) |
| KITTI | tracking sequence | — | 11 / 4 / 6 | 3,320 / 1,152 / 3,536 |
| nuPlan | **log** | map | 19 / 6 / 9 (33 / 11 / 16 scenarios) | 792 / 264 / 384 |

* The Planner D split (`configs/phase0g_scene_split.json`) is not reused: it held these 85 scenes
  out as its test set.
* **nuPlan splits by log, not scenario token.** 3 of the 34 same-log scenario pairs overlap in time
  by 5.2–9.1 s (`configs/benchmark_nuplan_scenarios.csv`), so a token split would put
  near-identical states on both sides. For the same reason the nuPlan bootstrap unit is the log.
* Two consequences of the rule, accepted rather than tuned away: KITTI test holds 44% of frames
  because sequences differ tenfold in length; nuPlan test is entirely Las Vegas, because the three
  smaller maps sort into train positions. Las Vegas is 45 of the 60 scenarios.

### Frozen now: the benchmark table — `scripts/92_benchmark_table.py`, one run, one CSV

**Cells.** nuScenes × {oracle, mono} × {brake, plan_ade, plan_fde}; KITTI × {oracle, mono} ×
{brake, traj}; nuPlan × {PDM-Closed, IDM} × {safety, scalar_J}. `traj` is Planner B, the rollout
planner — the only trajectory system with both geometries on the sequence track. `plan_*` is PKL's
planner against the real future trajectory; it fails the viability test and carries that flag.
nuPlan costs are Track B's external cost, not the official nuPlan metrics.

Inputs: nuScenes `*_phase0g_eta_fde_{oracle,mono}/joined_frames.pkl`; KITTI
`20260913_133004_core_matrix_postreview` and `20260912_111225_planner_b_static_fixed` (KITTI
geometry, untouched by review defects D3, D4 and D7); nuPlan `phase0g_external_{pdm_closed,idm}_raw.csv`
plus the signals below.

**Signals, and where each cannot exist.**

| signal | deployable | nuScenes | KITTI | nuPlan |
|---|---|---|---|---|
| random | yes | ✓ | ✓ | ✓ |
| uncertainty | yes | `unc_sum` | `unc_sum` | miss-model proxy Σ(1−p_cheap), **privileged**: the simulator's own detection probability |
| criticality, cheap side | yes | `feat_crit_sum` | `feat_crit_sum` | composite criticality over CHEAP tracks |
| criticality, GT | no | `crit_sum` | `crit_sum` | composite criticality over all tracks in the camera |
| exact ΔE + 8 variants | no | ✓ | ✓ | E1 and E6 only; E2–E5 and exact ΔE N/A — the miss model makes no false positives, class or localisation errors |
| PKL, TIP | no | ✓ | N/A (no map rasters) | N/A |
| gate-ridge, gate-GBM | yes | `features.py` | `features.py` | feature list below |
| decision oracle | anchor | ✓ | ✓ | ✓ |

**Learned baselines** are fit on train ∪ val and scored on test once. Neither selects anything on
val: RidgeCV chooses alpha by internal CV on the training rows, and the GBM uses the fixed
configuration in `predict.make_model`. Val is reserved for methods that tune.

**Statistics.** η at q ∈ {10, 20, 30, 50}% of the test frames. The gain of a signal is its expectation
over uniformly random tie-breaks, computed exactly (the limit of averaging over seeds); random is
k/n·ΣV. Bootstrap over test units (scene / sequence / log), nboot = 1000, seed 0, with the paired
difference to random taken in the same draws; draws whose oracle prize falls below 0.25 of the
full-sample prize are dropped and counted. `tie_frac` and `responsive_frac` are reported per row.

**Descriptives per cell**, on test and on all units: affected share, harmed share, destroyed-benefit
ratio D = Σ max(−V, 0) / Σ max(V, 0), all-FULL reduction, oracle@q reduction. Self-agreement share
S(M) = 1 − η_truth(M) / η_self(M) for M ∈ {PKL, TIP}, where self is Planner C path deviation, on
identical frames, against ADE and FDE truth, both geometries, at every quota.

### Frozen now: nuPlan cheap-side features — `scripts/91_nuplan_cheap_features.py`

Rebuilt offline: the intervention is a deterministic hash, so the CHEAP and FULL track sets of each
Track B state are regenerated exactly and checked against the stored `n_tracks_cheap/full`.
Gate features use only the tracks the CHEAP branch kept at the decision iteration and the one
before, the ego state, and traffic-light status: ego speed and acceleration; kept tracks in total
and in the camera cone, per class; static objects in the cone; nearest range in the cone; nearest
gap, count within 20 m and 40 m, and minimum TTC in a ±2 m corridor; tracks in the 30–40 m band;
sum and max of composite criticality over kept tracks; red lights; change in kept tracks in the cone
since the previous iteration. Excluded from the gate: miss-model probabilities (privileged) and
anything computed from FULL or reference tracks.

### Frozen now: allocation under measured cost — `scripts/93_budget_allocation.py`

* **Cost model: cascade.** CHEAP runs on every frame, an allocator adds its own per-frame
  overhead, escalated frames additionally run FULL. Costs: end-to-end median latency and GPU-rail
  energy per frame from `docs/full_project_report.md` (profile runs `20260912_021305` KITTI,
  `20260912_073228` nuScenes). nuPlan uses the KITTI YOLOv8s 320/640 numbers, since the miss model
  was fitted on those outcomes. Overheads of cheap-side features and gate inference are measured on
  this board. Diagnostics need FULL on every frame, so they cost at least CHEAP + FULL per frame and
  are reported as infeasible below that budget rather than given an η.
* **Budgets:** mean ms (and mJ) per frame, set at the levels where a zero-overhead allocator
  escalates 10/20/30/50% of frames.
* **Two fidelity levels are a relabelling.** With per-mode costs, a ms budget is a linear transform
  of a frame quota; on those tracks the only differences from the count table are overhead and
  feasibility, and they are reported as that, not as a new result.
* **Multi-fidelity, where cost can change the decision:** KITTI, mono, YOLOv8s cascade 320 → {384,
  512, 640}, brake and traj. Latency and energy disagree about the intermediate levels (384 costs
  0.78 of 640 in ms but 0.58 in mJ). Per frame, one level is chosen under a ms budget or a mJ budget.
  Oracle: Lagrangian relaxation, exact up to one fractional frame, with the bound reported.
  Deployable: one ridge or GBM gate per level, trained on train ∪ val. Random: random frames sent
  to 640. Reported: η, the level mix, and cross-cost (mJ spent by the ms-optimal plan and vice versa).

### Honest prior

The gates will score below their leave-one-scene-out development numbers (braking/mono GBM 0.465):
they now train on about half the units. Test intervals over 6 KITTI sequences and 9 nuPlan logs will
be wide, and some cells may not separate anything from random.

### 2026-09-14 00:40 — amendment, after the overhead measurement and before reading any budget result

The measured single-frame GBM inference is **16.1 ms** — 83% of a FULL pass on nuScenes (19.35 ms) —
with the CPU rail 7.4 W over idle; ridge is 0.40 ms and the cheap-side features 3.6 ms. The GBM number
is almost certainly scikit-learn's per-call cost (HistGradientBoosting dispatches an OpenMP pool on
every `predict`), not what a depth-3, 200-tree model costs to evaluate. The primary result keeps the
measured default, as registered. Declared now, before any budget row has been read: a sensitivity run
repeats `93_budget_allocation.py` unchanged under `OMP_NUM_THREADS=1` (`--suffix _1thread`), and both
are reported side by side.

## 2026-09-14 01:20 — Benchmark v0 results (pre-registered 2026-09-13 23:55)

Report: `docs/iclr_benchmark.md`; every table at every quota: `docs/benchmark_tables.md`, generated
from `results/final/benchmark_*.csv` (runs `20260913_233410_benchmark_table`,
`20260913_233938_benchmark_budget`). Checks that ran before any number was read: exact tie expectation
against 20,000 random orders; nuPlan track sets regenerated offline match Track B on 1440/1440 states
for both planners; the multi-fidelity greedy against brute force on 150 instances.

* **Harm on the test split.** Replicates wherever enough frames are affected. D ranges from ~0 (KITTI,
  oracle geometry: Planner B 0.004, braking 0.09 over all units) to 0.91 (PKL's planner, ADE, oracle,
  test). A single "22–88%" range overstated how uniform it is. nuPlan test has 11–42 affected states
  per cell; PDM-Closed safety has none harmed.
* **Baselines, test split.** No existing score is deployable; the existing scores that beat random are
  diagnostics (ΔE E6 on nuScenes oracle braking and on nuPlan; exact ΔE on KITTI oracle braking). Across
  all quotas 21 deployable rows beat random: 20 learned gates (KITTI, nuPlan) and cheap-side criticality
  on nuScenes braking/mono at 30%. Cheap-detection uncertainty beats random nowhere, and is worse than
  random on PKL's planner under mono and on KITTI Planner B. **The nuScenes braking/mono gate
  (LOSO +0.360 over random) does not survive the frozen split: +0.21 [−0.15, +0.49].**
* **Self-agreement.** All scenes: PKL S = 0.56 (ADE) / 0.67 (FDE) under oracle geometry, 0.48 / 0.61
  under mono; TIP similar. Test split: S ≈ 0.75–1.04, because truth-referenced η is near zero on those
  24 scenes.
* **Measured cost.** Feature extraction 3.6–3.9 ms (≈20% of FULL), ridge 0.39 ms, GBM 16.1 ms per
  frame. No learned gate escalates anything at a 20% budget; GBM escalates nothing up to 50%. Under
  measured cost four rows beat random, all at 50% (ridge on nuPlan PDM-Closed; cheap-side criticality
  on nuScenes braking/mono in ms and in mJ). With 320/384/512/640 on KITTI mono, the best 640-only plan
  reaches 0.74–0.78 of the multi-level oracle, and for braking the energy-optimal plan overruns the
  latency budget by 14% while the latency-optimal plan leaves 12% of the energy budget unspent.
* **Sensitivity run (declared 00:40), `OMP_NUM_THREADS=1`.** GBM single-frame 12.9 ms, CPU rail 1.5 W
  over idle (7.4 W in the primary run). Latency conclusions unchanged. Energy overheads ~5× lower:
  ridge becomes feasible under mJ budgets, and rows beating random under measured cost go from four to
  five (ridge on nuPlan PDM-Closed at 50% gains its mJ rows; cheap-side criticality on nuScenes
  braking/mono loses its mJ row). A descriptive timing note (`95_gate_inference_batch_timing.py`, feeds
  no allocation) shows GBM inference is per-call overhead — 16.05 ms single vs 0.021 ms per frame in a
  1,000-frame batch — so the binding cost is feature extraction (3.6–3.9 ms, ≈20% of FULL).

## 2026-09-14 — Task 1 pre-registration: does sign-varying value survive per-mode operating points?

### The concern

Harm could be an artefact of one shared threshold (`op_conf = 0.25` for CHEAP and FULL): a
higher-resolution pass puts more low-confidence boxes above the same cut, so FULL may "hurt" only
through extra false positives. The test: give each fidelity its own operating point, chosen without
test units, and see whether harm and the harm-to-benefit ratio remain.

### Step 0 (read-only, done before this entry)

Every detection cache stores boxes down to conf 0.10 (the engines ran at `Mode.conf = 0.10`):

| cache | min conf | share of boxes < 0.25 |
|---|---|---|
| nuScenes ns_cheap_320 / ns_full_640 | 0.1002 / 0.1002 | 48.5% / 40.9% |
| KITTI Y8 cheap_320 / cheap_384 / cheap_512 / full_640 | 0.1002 | 46.4% / 45.2% / 42.9% / 39.8% |
| KITTI RT-DETR rt_mid_480 / rt_full_640 | 0.1002 | 76.6% / 74.8% |

Re-thresholding a cache at t ≥ 0.10 is identical to running the engine at t: greedy NMS only lets a
box be suppressed by a higher-scoring one, and `max_det` truncates a confidence-sorted list, so the
boxes above t are the same either way. Thresholds below 0.10 are not available: the 0.05 point of
the S2 grid is dropped. Re-running the engines at conf ≥ 0.01 for the eight modes would take about
35–45 min (logged runs: nuScenes 208 s; KITTI Y8 708 s + 502 s; RT-DETR 842 s + 870 s). It is not
done unless a scheme selects 0.10 for some mode, in which case that choice is reported as limited
by the boundary.

### Step 1 — wiring

`RiskConfig.thr(role)` returns `op_conf` for CHEAP and `op_conf_full` (falling back to `op_conf`)
for FULL. It replaces the shared threshold in `decision.build`, `decision.add_perception_gain`,
`65_planner_b_decision.build_b`, `percep_metrics.frame_losses` and the recall bins of
`50_percep_metrics.primitives`, `nusc_submission.build_submission` and `60_build_submissions`, and
`objects.py`. With `op_conf_full = None` every path is the current code. The box-centre fix and all
other conventions are untouched.

Evaluation path — the threshold is applied where detections are filtered, then the unchanged
pipeline runs:

* **Per-mode outcomes, composed into cells.** Nothing couples the two modes: the braking and lateral
  controllers and Planner B carry the previous action within a mode, perception losses are per
  mode, submissions are per box. Each mode is therefore run at each threshold it needs, and a cell
  is V = J_CHEAP(t_c) − J_FULL(t_f).
* **q_plan.** Submissions are built once per mode and geometry at 0.10. The submission at t is its
  subset with score ≥ t: the monocular lift and the oracle-geometry GT match are per box and do not
  depend on the threshold. Map and ego raster channels do not depend on detections, so only the
  object channel is redrawn, from the filtered boxes through the same `load_prediction →
  add_center_dist → filter_eval_boxes → get_other_objs` path and PKL's own corner rasterisation.
  Planner C and ADE as in `74_plannerC_vs_truth.py`.

**Checks that must pass before any scheme is scored** (|Δ| ≤ 1e-9 unless stated). A failure stops
the run and is reported instead of results.

1. S0 (0.25 / 0.25) reproduces `20260913_133004_core_matrix_postreview` (J, Jlat, dE, every dE_E*,
   n_cheap, n_full), `20260912_111225_planner_b_static_fixed` (JB) and `planC_vs_truth[_mono].csv`
   (JC_ade) on every frame.
2. A direct run at (0.15, 0.45) and at (0.45, 0.15) equals the composition from per-mode runs.
3. The 0.10 submissions filtered at 0.25 equal the current submission files, box for box.
4. Rasters rebuilt at 0.25 equal the cached CHEAP and FULL test rasters bit for bit.

### Step 2 — threshold schemes

All selection uses train ∪ val units of `configs/benchmark_splits.json` only.

| scheme | CHEAP | FULL |
|---|---|---|
| S0 current | 0.25 | 0.25 |
| S1 count-matched | 0.25 | `tables.match_detection_counts` over train ∪ val caches |
| S2 F1-optimal | argmax pooled F1 | argmax pooled F1 |
| S3 precision-matched | 0.25 | lowest t on a 0.01 grid (0.10–0.70) with pooled precision ≥ CHEAP precision at 0.25; if none, 0.70 flagged |
| S4 downstream-tuned | argmin mean J of that mode | argmin mean J of that mode |

Details:
* Precision = matched detections / detections; recall = matched measurable GT / measurable GT.
  Pooled over train ∪ val frames, with the matching `RiskConfig` uses: greedy by confidence,
  class-agnostic, IoU 0.5, `min_gt_height`.
* S2 grid: 0.10–0.70 in steps of 0.05, each mode separately.
* S4 is chosen per mode, downstream system and geometry, on the same grid; exact ties go to the
  value closest to 0.25.
* Sweep: (t_c, t_f) ∈ {0.15, 0.25, 0.35, 0.45, 0.55}², all units, harm rate and ρ only.

**Cells (14).**
* nuScenes Y8 320→640 × {oracle, mono} × {q_brake, q_plan (ADE)}.
* KITTI mono × {q_traj (Planner B), q_brake} × {Y8 320→640, 384→640, 512→640, RT-DETR 480→640}.
* KITTI oracle Y8 320→640 × {q_traj, q_brake}.

**Reported per scheme × cell × split (all units, test):**
* chosen thresholds;
* detections per frame, precision and recall per mode;
* affected count;
* harm rate = P(V<0 | V≠0);
* ρ = Σ max(−V,0) / Σ max(V,0);
* all-FULL and oracle@20 loss reduction, with the exact tie expectation of `92_benchmark_table.py`;
* unit bootstrap (scene / sequence), 1000 draws, seed 0, 95% CI for harm rate and ρ.

**nuScenes cells, additionally.** For each gain g ∈ {exact FN (dE), FN+FP (E2), combined (E5_combined),
E_risk (E6)}:
* sign disagreement: share of frames with g ≠ 0 and V ≠ 0 whose signs differ;
* Spearman correlation of g and V over all frames;
* harmed | gain > 0 = P(V<0 | g>0), also given conditional on V ≠ 0.

Brake-vs-planner: the number of plan frames with V_brake · V_plan < 0, with the number where both
are nonzero. These definitions are written out here because the source of the paper's
sign-agreement table is not in this repository; they should be checked against it.

### Reading, fixed now

Applied to all-unit statistics, separately under S1, S2 and S3.

* **Survives:** under each of S1–S3, harm rate ≥ 20% and ρ ≥ 0.20 in at least 3 of the 4 nuScenes cells
  **and** in all 6 KITTI moderate-gap cells (mono × {q_traj, q_brake} × {384→640, 512→640, RT-DETR
  480→640}). This is the literal reading of "in the KITTI moderate-gap cells". The number of KITTI
  cells passing is reported, so a "most cells" reading can also be applied.
* **Collapses:** under S1–S3, harm rate < 10% or ρ < 0.10 in more than half of the 14 cells.
* **Otherwise:** mixed, reported per scheme and cell.

S4 and the sweep are reported but do not enter the reading. Test-split statistics are reported next
to every all-unit number. The outcome is reported whatever it is.

### Outputs

`results/final/calibration_cells.csv`, `calibration_sweep.csv`, `calibration_thresholds.csv`;
per-mode outcomes under `results/raw/*_calibration_*`; `docs/iclr_calibration.md`.

### Cost and prior

Estimated cost:
* precision/recall curves: about 10 min;
* per-mode pipeline runs at about 15 thresholds for 7 pair × geometry combinations: about 1–1.5 h,
  with 4 CPU worker processes;
* q_plan: about 45 min, as one GPU job;
* statistics: about 15 min.

Prior: KITTI with oracle geometry already has ρ ≈ 0 for Planner B (0.004 over all units), so a
"collapse" count will include cells that had nothing to collapse from; the moderate-gap and
nuScenes cells are where the question is live.

## 2026-09-14 — Task 2 pre-registration: lightweight routers on the benchmark and budget tracks

### Question

Do cheaper, differently-shaped routers than the 65-feature gate — one reading the raw detection
list, one reading raw pixels — beat random on the frozen test split, and can they be afforded under
measured cost?

### Common protocol

Unchanged from the benchmark (RESEARCH_LOG 2026-09-13 23:55):
* cells and splits of `92_benchmark_table.py`; fit on train ∪ val, scored on test once;
* η at 10 / 20 / 30 / 50% with the exact tie expectation;
* unit bootstrap, 1000 draws, seed 0, paired against random.

Additional rules:
* No router hyperparameter is changed after any test score has been seen.
* No weights are downloaded. `~/.cache/torch/hub/checkpoints` is empty, so R2 trains from scratch.

### R1 — detection-list router (ORIC-style)

* **Input:** every cached CHEAP detection (conf ≥ 0.10, the engine floor), top 25 by confidence.
  Per detection: conf; xyxy normalised by image width and height; one-hot coarse class (vehicle,
  person, cyclist); normalised box area. That is 9 × 25 = 225 dims, zero-padded, built with
  vectorised numpy.
* **nuPlan adaptation.** There is no confidence, so R1 takes the CHEAP-kept tracks at the decision
  iteration, regenerated offline with the Track B filter exactly as in 91, top 25 by distance.
  Per track: presence, x / 80, y / 40, length / 10, width / 5, one-hot class (vehicle, pedestrian,
  bicycle, static), area / 50 — 10 × 25 = 250 dims.
* **Models:**
  * MLP: scikit-learn, 2 × 64 ReLU, StandardScaler on inputs, α = 1e-4, 300 iterations, seed 0.
  * GBM: the fixed configuration in `predict.make_model`.
* **Targets:** V (regression, score = prediction) and 1[V>0] (classification, score = P(V>0)).
  This gives four rows: R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf.
* **Cells:** nuScenes, KITTI and nuPlan.

### R2 — raw-pixel router (weak-skipping CNN)

* **Input:** the CHEAP camera image (nuScenes CAM_FRONT, KITTI image_02), resized to 128 × 128 and
  normalised. No augmentation: a horizontal flip would change the scene the target describes.
* **Network:** torchvision MobileNetV2. The width multiplier is chosen before training as the value in
  {1.0, 1.1, 1.2, 1.3, 1.4} whose counted FLOPs at 128 × 128 are closest to 0.15 GFLOPs; the exact
  count is reported.
* **Heads:** one network per dataset, with one sigmoid head per cell (nuScenes 6, KITTI 4).
* **Loss:** masked BCE on 1[V>0], with pos_weight = negatives / positives per head from train ∪ val.
  Plan heads only see frames with trajectory truth.
* **Training:** from scratch; AdamW, lr 1e-3, weight decay 1e-4, batch 128, 30 epochs, cosine
  schedule, seed 0; train ∪ val frames; no early stopping and no model selection.
* **Export:** ONNX → TensorRT FP16 with `trtexec`. Test scores come from the TensorRT engine. Sanity
  check: Spearman correlation between engine and PyTorch scores ≥ 0.99 per head, otherwise reported.
* **Cells:** nuScenes and KITTI only.

### Profiling and budgets

Profiling runs on an otherwise idle board, one job at a time. Single-frame median latency for:
* R1 feature construction;
* R1 inference — MLP and GBM on one row, and the GBM amortised over a 1,000-row batch;
* R2 preprocessing — resize and normalise from a decoded frame, with decode timed separately;
* R2 TensorRT inference.

Energy: CPU-rail and GPU-rail power over idle during sustained loops, converted to mJ per frame.

The routers and a **batched-inference GBM gate** (the 65-feature gate charged its feature time plus
the measured amortised batch inference cost) are added to the measured-cost allocation, and the ms
and mJ tables are re-run with `93_budget_allocation.py`'s cost model, so that row is evaluated rather
than computed analytically.

### Reading, fixed now

* A router beats random in a cell at a quota if the paired 95% lower bound is above zero.
* It is affordable at a budget if its measured overhead leaves at least one escalation.

Reported against the 65-feature gates:
* the number of cells where any router beats random at 20% under a frame quota;
* the same count under the measured ms budget at 20%.

### Outputs

`results/final/benchmark_table_routers.csv`, `benchmark_budget_routers.csv`, `docs/iclr_routers.md`.

## 2026-09-14 — Task 3 pre-registration: documentation and feasibility (no downloads)

### 3a

`docs/gate_spec.md`, generated by a script so it cannot drift from the code, will contain:
* every gate feature in `feature_columns` order, with name, group and registered source, taken from
  `rap.features._REGISTRY`;
* the 18 nuPlan gate features from `benchmark_nuplan_signals.json`;
* the exact training procedures: leave-one-scene-out in `84`, train ∪ val → test in `92`, and
  single-frame timing in `93`;
* the model hyperparameters, read from `predict.make_model`.

### 3b

From the local nuPlan mini DBs only:
* for the 34 benchmark logs and the 9 test logs, the CAM_F0 image count for the whole log and inside
  the benchmark scenario windows, with the relative `filename_jpg` paths and camera resolution;
* size estimate: image count × mean JPEG size. No nuPlan image exists locally, so the mean is the
  measured nuScenes CAM_FRONT mean (146.9 KB at 1600 × 900) scaled by pixel count, and it is
  labelled an estimate;
* which distribution archives hold these logs cannot be determined without an index on disk, and is
  reported as unknown rather than guessed;
* comparison against free disk (108 GB);
* detector runtime scaled from the logged nuScenes detection run (3,376 frames, two modes, 208 s).

Reading: "feasible now" if the scenario-window images of the 9 test logs fit in 25% of free disk and
the scaled detector runtime is under 2 h.

Output: `docs/nuplan_real_perception_feasibility.md`.

### 2026-09-14 10:32 — Task 1: check 1 caught a configuration error; nothing had been scored

The first calibration outcome run (`20260914_090507_calibration_outcomes`) passed check 1 for every
braking, lateral and perception column (all 7 pair × geometry combinations, every frame) and check 2
(direct asymmetric runs equal the composition). It **failed check 1 for Planner B on every KITTI
cell**: JB differed on ~2,300 of 8,008 frames, by up to 16.9.

The cause is the runner's configuration, not the pipeline. The benchmark's Planner B is the
`static_obstacles` preset (`PlannerBParams(obstacle_closes=False)`; run
`20260912_111225_planner_b_static_fixed`, the table `92_benchmark_table.py` reads), and
`100_calibration_outcomes.py` had called `PARAMS_B["default"]`.

The fix:
* The preset is now a named constant.
* A `--reuse_run` mode recomputes only the Planner B columns of the existing outcome files, with
  `static_obstacles`, at the same thresholds.
* Every other column is copied unchanged: they already passed check 1.
* Checks 1 and 2 then run again on the new run directory.

No scheme had been scored, and the reading is unchanged. The supervisor had moved on to Task 2 by
design, and its first step crashed on an array one column too narrow in `104_nuplan_track_lists.py`
(a 10-dim track vector allocated as 9); fixed. The supervisor is restarted with Task 1 first.

### 2026-09-14 11:03 — Task 1: q_plan split into two processes after a CUDA out-of-memory

`101_calibration_plan.py` passed check 3 for the first submission, then ran out of CUDA memory at
the first Planner C batch. It had loaded the nuScenes devkit tables and the planner into one process,
and on this board's unified memory that left 1.36 GB when 512 MB was requested. `74` never loaded the
tables, so it had never hit this.

Nothing was scored. The script now runs as two processes with the same logic and the same checks:
* `--stage boxes` (devkit, CPU): check 3, then every filtered box at 0.10 in the ego frame, with its
  score. `get_other_objs` transforms each box on its own, so subsetting by score afterwards equals
  transforming the subset.
* `--stage plan` (GPU, no tables): rasters per threshold, check 4, Planner C with 74's batch size and
  frame order, check 1.

## 2026-09-14 11:29 — Task 1 results: per-mode operating points — **survives**

Report: `docs/iclr_calibration.md`. Every table: `docs/calibration_tables.md`. Runs:
`20260914_103252_calibration_outcomes` (Planner B recomputed from `20260914_090507`),
`20260914_110316_calibration_plan_boxes`, `*_calibration_plan`, `*_calibration_cells`.
All four equivalence checks passed before any scheme was scored.

* **Reading, all units.**
  * S1, S2 and S3 each pass 4/4 nuScenes cells and 6/6 KITTI moderate-gap cells.
  * Cells below the collapse line: 2 (S1) and 3 (S2, S3) of 14. All are KITTI Y8 320→640 cells already
    below ρ 0.10 at S0.
  * Verdict: survives.
* **The false-positive explanation.** Under S1, FULL emits no more boxes than CHEAP and is more precise:
  0.764 vs 0.654 on nuScenes, 0.793 vs 0.701 on KITTI 384→640. Harm remains:
  * nuScenes mono braking: 38.3% [33, 44], ρ 0.40 [0.28, 0.59];
  * KITTI 384→640 braking: 38.7%, ρ 0.49.
  * Harm on nuScenes braking falls 5–7 points against S0.
  * The planner cells do not move: harm ~51%, ρ 0.67–0.89.
* **Sweep over 25 threshold pairs:**
  * every pair passes in all four nuScenes cells;
  * 11–25 pairs pass in the moderate-gap KITTI cells;
  * none passes in the KITTI oracle cells.
* **Signs.** Perception gains still disagree with V's sign on 29–54% of frames, with |Spearman| ≤ 0.10.
  Braking and the planner disagree in sign on about half the frames where both respond, under every
  scheme.
* **S4.** The CHEAP optimum sits at the 0.10 grid floor in 7 KITTI cells, so those optima are limited by
  the boundary, as the pre-registration requires. Harm under S4 is 28–55% in every nuScenes and
  moderate-gap cell.

### 2026-09-14 11:32 — Task 2: R1 crashed on a file name, no result had been written

`103_routers_r1.py` scored every nuScenes and KITTI cell, then failed to save the first nuPlan score
file, because the geometry label "n/a" put a slash into the file name. It writes its table only at the
end, so no R1 number exists. The fix drops the slash when writing, and `93` restores the label when
reading. The supervisor had already started R2's image cache. A continuation script waits for that
cache to finish, then runs R1, R2 training and the router budget tables, one at a time. Nothing about
the design changed.

### 2026-09-14 12:53 — Task 2: the board rebooted during R2; R2 is now staged per dataset and per step

R2 finished training the nuScenes network: 30 epochs, loss 1.199 → 0.015, weights saved 11:50:46. The
board rebooted, with uptime 2 min at 12:51, taking the session, the supervisor and the job with it.
No R2 result had been written.

The first entry named unified-memory exhaustion by `trtexec` as the likeliest cause. The board's own
telemetry (the dashboard agent's 20-s history) contradicts it. The last sample before the reboot,
11:50:40, shows:
* GPU at 99% for 5 min, 36 W;
* 76 °C;
* memory at 59%, swap at 1.7%;
* no `trtexec` process yet.

The cause is unknown. See 13:20 for the second reboot, which rules out memory and heat more firmly.

Changes kept (harmless, and they reduce peak memory):
* Every R2 stage is its own process, one dataset at a time.
* Train (and score with PyTorch) → exit.
* Export: ONNX on the CPU, then `trtexec` with the workspace capped at 512 MiB.
* The finished nuScenes weights are reused (same procedure, all 30 epochs completed).

After the reboot: `git fsck` clean; both image caches intact.

### 2026-09-14 13:20 — a second reboot, at light load: not memory, not heat

R2 then completed on both datasets:
* nuScenes: weights reused, TensorRT engine built in 120 s;
* KITTI: trained 30 epochs at 99% GPU, 36 W and up to 78.5 °C for ten minutes without incident, engine
  built in 118 s;
* TensorRT and PyTorch scores agree to Spearman ≥ 0.9996.

The router budget run started at 13:10:59. The board rebooted about 40 s later, before its log had a
line. The last telemetry sample (13:11:20):
* CPU 86%, GPU 0%;
* 12 W, 56 °C;
* memory at 22%.

The history also shows a brief sample at 13:15:00 (memory at 6%, just booted), then another gap until
13:16:40, and uptime was 1 min at 13:17. So the board apparently rebooted again while idle during boot.

Neither reboot coincides with memory pressure or heat. The second happened at light load and was
followed by one at idle. That points away from the workload, towards power or hardware; this is not
verifiable without a kernel log. No result was lost: every finished step had written and committed
its output. Only the router budget step remains.

## 2026-09-14 13:40 — Task 2 results: lightweight routers on the benchmark and budget tracks

Report: `docs/iclr_routers.md`. Tables: `docs/routers_tables.md`. Runs: `20260914_113930_routers_r1`,
`20260914_125344_router_r2`, `20260914_132102_benchmark_budget_routers` (`20260914_131102_*` is the run the reboot interrupted).

### Frame quota, test split

| signal | wins | losses | chance wins | where |
|---|---|---|---|---|
| R1 detection-list router (four variants) | 7–11 of 56 | 0–1 | ~1.4 | KITTI 6–8 of 16; nuScenes 0–3 of 24; nuPlan 0 of 16 |
| R2 pixel CNN (0.151 GFLOPs, TensorRT ~ PyTorch Spearman ≥ 0.9996) | 0 of 40 | 4 | ~1.0 | — |
| 65-feature GBM gate | 14 of 56 | 0 | ~1.4 | KITTI 7, nuPlan 7 |

At 20%, a router beats random in 3 cells and a gate in 5. None do on nuScenes.

### Measured cost on the board

| component | cost |
|---|---|
| R1 features | 0.11 ms |
| R1 MLP inference | 0.54 ms (total 0.65 ms, 3.4% of a FULL pass) |
| 65-feature extraction | 3.5 ms |
| batched GBM inference | 0.018 ms/frame |
| R2 resize + upload | 8.2 ms (nuScenes) / 3.3 ms (KITTI) |
| R2 TensorRT | 1.55 ms |

### Under measured cost

* **Pre-registered reading at the 20% ms budget:** routers beat random in 1 cell (KITTI oracle
  Planner B, R1-MLP P(V>0), η 0.41, lower bound +0.11), gates in 0.
* **R1-MLP** still escalates about 16.5% of frames at that budget and has 9 latency-budget wins across
  levels. The single-row GBM gate and the R1 GBM escalate nothing.
* **Batched GBM gate:** 3 wins, all at 30–50%.
* **R2:** no wins.
* **Energy budgets:** no learned allocator beats random. R1-MLP's single-row call drew 8.9 W over idle
  on the CPU rail.

### Operational notes

The fan was set to 100% at 13:21 at the user's request, and will be set to 100% for heavy jobs from now on.

## 2026-09-14 — Task 4 pre-registration: nuPlan sensor archive index (metadata only)

**Goal.** Find how nuPlan distributes the CAM_F0 images the benchmark would need, and how much must be
downloaded to run real perception on the 9 test logs and on all 34 benchmark logs. The per-log and
per-scenario image lists come from Task 3.

**Rules, enforced in `scripts/111_nuplan_archive_index.py`.** Every request goes through one logger that
records method, URL, status, response headers and body bytes in `requests.jsonl`, and refuses any request
that would take the total past 50 MB. Allowed:
* official documentation pages;
* HTTP HEAD;
* Range reads of a ZIP's end-of-central-directory record and its central directory.

Not allowed:
* data archives or members;
* credentials, account creation, accepting terms, or tokens.

A 401/403, or a redirect to a login or terms page, stops that line of inquiry, and the requirement is
reported. Archive names and URLs are taken only from official sources — the local devkit docs, the
nuPlan website or its linked download page — and every URL records its source. Nothing is guessed.

**What the local devkit already says** (`third_party/nuplan_devkit/docs/dataset_setup.md`, README): the
download page is `https://www.nuscenes.org/nuplan#download`, and downloading requires creating an account
and agreeing to the Terms of Use. The devkit lists no archive names and no archive URLs. PROVENANCE records
that the mini camera blobs are nine shards of 45–54 GB split by blob, but not their URLs, so they cannot be
used here.

**Steps.**
1. Fetch the official page and list any archive links it exposes without logging in.
2. HEAD every archive URL it exposes. If Range requests work without login, read only the ZIP central
   directory.
3. Map the 34 and 9 logs to archives with exact member counts and compressed bytes.
4. Compare three options against free disk: whole archives; needed CAM_F0 members via Range, only if the
   server allows it; scenario-window images only.

The measured mean CAM_F0 JPEG size replaces Task 3's 212 KB estimate, if a central directory is reachable.

**Outputs.** `results/final/nuplan_archive_index.csv`, `docs/nuplan_archive_index.md`, and the request log
under `results/raw/*_nuplan_archive_index`.

**Reading.** "Doable now" requires every needed archive URL to come from an official source reachable
without login, and the chosen option's download to fit within free disk with a 25% margin. Anything short
of that is reported as the exact action the user must take.

## 2026-09-14 14:05 — Task 4 result: the nuPlan archive index is behind a login; stopped as registered

Report: `docs/nuplan_archive_index.md`. The official sources give no archive names or URLs that can be
reached without logging in:
* The local devkit docs point to `https://www.nuscenes.org/nuplan#download`. They state that an account and
  acceptance of the Terms of Use are required, and that the archives appear after logging in.
* The official page itself is JavaScript-rendered: 6,574 bytes, no archive links.

No HEAD, Range or central-directory read was possible, because no official URL was reachable. No bundle or
API was mined for links. Total bytes fetched: 6,574.

What stands without the index:
* the exact image counts from Task 3: 3,320 scenario-window and 37,000 whole-log CAM_F0 images for the 9
  test logs; 12,921 and 144,939 for all 34;
* the Task 3 size estimate: ~0.70 GB for the test windows;
* the earlier PROVENANCE record of nine 45–54 GB mini camera shards split by blob (~450 GB), which would
  not fit in ~107 GB free.

User action required: log in, accept the terms, and provide the mini sensor-archive names and links from
the download page. Steps 2–4 can then be finished within the 50 MB metadata cap.
