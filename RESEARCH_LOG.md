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

### 2026-09-14 14:20 — Task 4 update: archive names and sizes from the logged-in download page

Transcribed by the user, without links. The v1.1 Mini Sensors are:
* a "Metadata" archive shown as 0.00 GB;
* nine camera shards of 42.06–50.48 GB, 419.7 GB in total;
* nine lidar shards.

The v1.0 section holds only DBs and maps. The whole-archive option cannot fit in ~107 GB free. The
central directories of all nine shards (~25–35 MB each by estimate) would exceed the 50 MB cap, so the
plan reads the metadata archive first and only the needed shards' directories. Waiting for the link
addresses.

### 2026-09-14 14:30 — Task 4: two official links provided by the user

The user copied two links from the logged-in download page:
* "Mini Sensors Metadata":
  `https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan_mini_sensor.txt`
* "Camera 0":
  `https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_0.zip`

Both sit under `public/` with no signature. Recorded with their source in
`results/raw/nuplan_archive_urls/user_provided.tsv`.

Camera 1–8 URLs are **not** built from the naming pattern. They are used only if an official source,
such as the metadata file, lists them; otherwise the user is asked. Order: HEAD both; fetch the metadata
text if it is small; read the central directory of Camera 0 only if Range works without login and it
fits the 50 MB cap.

## 2026-09-14 14:27 — Task 4 result: archive index, log-to-shard map and download options

Requests, all logged:
* HEAD of both official links (no login; `Accept-Ranges: bytes`);
* GET of the 2,622-byte metadata file;
* EOCD, ZIP64 record and full central directory of `nuplan-v1.1_mini_camera_0.zip` by Range (all 206).

Total fetched **42,019,619 B** of the 52,428,800-B cap. No archive member was fetched.

**Findings.**
* **Camera 0 central directory:** ZIP64; 242,385 entries: 242,320 JPEGs (all deflated), 64 directories and a LICENSE file;
  8 cameras × 30,290 images. Its logs equal metadata File group 0 exactly, so group *i* ↔ Camera *i* is
  verified for shard 0.
* **Measured CAM_F0 JPEG:** mean 211,049 B stored, 212,086 B uncompressed. This replaces the Task 3
  estimate of 212 kB, which moves by < 0.5%.
* **Page units:** the page's "GB" are GiB (Camera 0 HEAD = 48.63 GiB).
* **Shards needed:** the 9 test logs are in Cameras 0, 2, 3, 6; the 34 benchmark logs span all nine.

**Download options (9 test logs / all 34), against 114.6 GB free.**
* (a) whole archives: 202.0 / 450.7 GB. They do not fit together; each fits one at a time.
* (b) needed CAM_F0 members via Range: 37,000 images, 7.76 GB / 144,939 images, 30.6 GB.
* (c) scenario windows via Range: 3,320 images, 0.71 GB / 12,921 images, 2.72 GB.
* Camera 0 portions are exact (every needed file name found in the directory); the other shards are
  estimated at the measured mean.
* (b) and (c) also need the other shards' central directories: ~120 MB / ~320 MB, estimated from
  Camera 0 by size.

**Verdict.** Not doable in full now. Camera 0 (3 test logs; 769 window images, 167 MB) is fetchable by
Range today, but was not fetched, because Task 4 allows no data. The rest needs from the user:
* the official links for Camera 2, 3, 6;
* permission for the ~120 MB of directory reads beyond the cap;
* approval of the download.

Recommended: option (c), 0.83 GB including directories.

Outputs:
* `results/final/nuplan_archive_index.csv`;
* `docs/nuplan_archive_index.md`;
* working files in `results/raw/nuplan_archive_urls/`, including `needed_cam_f0.csv.gz`,
  `benchmark_logs_to_groups.csv` and `report_summary.json`.

## 2026-09-14 14:44 — Task 5 pre-registration: real YOLOv8s perception on the nuPlan external track

Committed before any request of this task and before any image is fetched, detected or scored.

**Goal.** Replace the transported KITTI miss profile (Track B) with real YOLOv8s 320/640 detections on
nuPlan CAM_F0. Re-measure sign-varying decision value for PDM-Closed and IDM on the same 1,440 states
(60 scenarios, 34 logs, 24 states each) that Track B and the benchmark use.

**URL sources.**
* Camera 0 and the metadata file: recorded in Task 4.
* Camera 1: the link the user pasted with this task.
* Camera 2–8: built by the user's own stated rule, quoted: "thay số là ra link nên tôi ko copy nữa"
  (replace the number and you get the link). Each URL's source column says so.
* A URL counts as verified only if HEAD returns 200 with `application/zip` **and** its central
  directory's log set equals metadata File group *i*.
* Any failure → STOP and report; no guessing beyond the user's rule.

### Stage A — data (`scripts/112_nuplan_fetch_cam_f0.py`)

**A2. Directories.**
* Per shard, in order 0–8: HEAD, EOCD (ZIP64), then the full central directory by Range. Name, method,
  sizes, **CRC32**, flags and local offset are stored.
* Task 4's listing lacked CRC32, so Camera 0 is re-read.
* Cap: 400 MB cumulative, counted in this task's own ledger. The nine EOCD-reported directory sizes are
  summed before any directory is read. If the sum exceeds 400 MB, Camera 0 is not re-read and its
  CRCs come from the local headers (flag bit 3 must be clear).

**A3. Members.**
* Only CAM_F0 members inside the Task 3 scenario windows ([t0 − 2 s, t1], `needed_cam_f0.csv.gz`
  `in_window == 1`): 12,921 images of the 34 logs.
* The planned byte total is computed exactly from the directories before the first member request;
  the fetch aborts if it would pass 3.5 GB.
* One Range per member: local header (30 B + name + the central extra length) plus compressed data.
  A longer local extra is fetched with one extra request; any over-read bytes are logged.
* Shards one at a time, 6 connections within a shard, 3 retries per transient error. A non-206 answer
  after retries → STOP. No whole-archive fallback.
* Verification per member:
  * local signature and name equal the directory's;
  * raw inflate to exactly the uncompressed size;
  * CRC32 equals the directory's;
  * `cv2.imdecode` gives 1080×1920×3.
* Every request logged: method, URL, range, status, bytes. Resumable from a manifest.

**A4. Layout.** `~/datasets/nuplan/sensor_blobs_cam_f0/<log>/CAM_F0/<hash>.jpg`.

### Stage B — perception

**B1. Detection** (`113_nuplan_detect.py`, edge env, fan 100%).
* Engines: `yolov8s_ns_cheap_320` (192×320) and `yolov8s_ns_full_640` (384×640). Their 16:9 input
  matches CAM_F0's 1920×1080, as on nuScenes.
* Existing `TwoFidelityDetector` TRT path: conf ≥ 0.10, NMS IoU 0.65, max 100.
* Cache: `rap.cache` format, one npz per (mode, log), frames in timestamp order, plus an index csv
  (log, frame, image token, timestamp, file). No mono geometry arrays.
* Latency: per-mode medians of preprocess, inference and postprocess, CUDA-synchronised over every
  image; JPEG decode reported separately.
* Energy: CPU + GPU rail power over idle, sampled at 20 Hz during a dedicated 300-image pass per mode;
  mJ per frame.

**B2. Time alignment.** For each state and each of its 4 history-buffer iterations (it−3..it), take the
CAM_F0 image nearest the iteration's lidar timestamp. Report the |Δt| distribution and flag > 50 ms.
Flagged images are kept and counted. No motion compensation.

**B3. Matched projection** (`114_nuplan_project_match.py`, nuplan env).

*Projection.*
* 3D boxes are the `lidar_box` rows of that iteration's lidar_pc, keyed by track token.
* Global → ego at the **image's** ego pose, then ego → camera by the inverse extrinsic — the devkit's
  `boxes_lidar_to_img` chain.
* 8 corners, clipped at z_cam = 0.1 m. Normalised coordinates are clamped to 1.25× the undistorted
  image border, then passed through the DB distortion (k1, k2, p1, p2, k3) and intrinsic. The
  distortion must be monotone out to that radius (a check).
* Box = hull of the corners, clipped to the image.
* In camera = centre z_cam > 0 and a clipped box with positive area. No range cap.

*Eligible classes.*
* vehicle, pedestrian, bicycle.
* traffic_cone, barrier, czone_sign and generic_object have no COCO class. They pass through
  unchanged in every branch, as in the transported filter.

*Matching.*
* Per image and per mode, among detections at that mode's threshold: `scipy linear_sum_assignment`
  maximising IoU over class-compatible pairs with IoU ≥ threshold.
* Compatibility (nuPlan category ← YOLO coarse class):
  * vehicle ← vehicle;
  * pedestrian ← person;
  * bicycle, which includes motorcycles and tricycles per the DB ← cyclist or person.

*Branch.*
* An eligible in-camera track is kept iff matched.
* Tracks outside the camera, and static classes, are unchanged.
* Unmatched detections are added as false-positive agents:
  * the bottom-centre pixel is undistorted, cast as a ray, rotated by the extrinsic, and intersected
    with the ego ground plane z = 0 (camera height = extrinsic z, 1.52 m);
  * the FP is dropped if the ray does not hit the ground ≥ 0.5 m ahead, or lands beyond 80 m;
  * centre = ground point + L/2 along the horizontal ray; heading = ego heading; velocity 0;
  * size = the class's median (L, W, H) over tracks of the 25 train ∪ val logs (person → pedestrian,
    cyclist → bicycle);
  * deterministic token per (variant, image, detection).
* FPs are per frame; there is no tracking.

*Variants.*

| variant | CHEAP thr | FULL thr | IoU | FP |
|---|---|---|---|---|
| **primary** | 0.25 | 0.25 | 0.3 | yes |
| nofp | 0.25 | 0.25 | 0.3 | no |
| s1 | 0.25 | `match_detection_counts` on all train ∪ val window images (Task 1 rule) | 0.3 | yes |
| iou50 | 0.25 | 0.25 | 0.5 | yes |

**B4. Checks, before any CHEAP/FULL branch is scored** → `results/final/nuplan_real_perception_checks.json`.
1. Data: counts, CRC and decode failures (must be 0), and the group-to-shard map.
2. Δt distribution and flags.
3. 20 overlays (seed 0, drawn from state images), downscaled to 960×540, in
   `results/final/nuplan_real_perception_overlays/`: projected boxes by class, static classes dashed,
   and each mode's detections at 0.25 with their matches.
4. Match IoU quantiles per mode and IoU threshold.
5. Per-class recall at 320 and 640 (thr 0.25, IoU 0.3 and 0.5), overall and by distance band
   (0, 10, 20, 30, 40, 60, ∞ m), also restricted to projected height ≥ 10 px. Compared with:
   * KITTI observed recall (greedy, class-agnostic, IoU 0.5, height ≥ 10 px — a different matching,
     so a caveat, not a pass/fail);
   * the transported model's predicted recall on the same nuPlan objects.
6. Identity:
   * a branch with every eligible in-camera track marked matched and no FP must equal the reference
     observation object-for-object on all 1,440 states × 4 buffer iterations;
   * its planner outputs must equal the reference exactly on 120 states (states 1 and 13 of each
     scenario), for both planners.
7. Reference re-run: collision, min clearance and both log deviations must equal Track B's stored
   reference values on all 1,440 states, for both planners. If not → STOP before scoring.
8. Ground plane (informational): median bottom height of vehicle boxes within 30 m, in the ego frame.
9. Distortion monotone to the clamp radius.

### Stage C — planners and losses

**C1** (`115_nuplan_real_counterfactual.py`).
* Imports `make_planner`, `score` and `filtered_history` from `82_nuplan_counterfactual.py` unchanged.
* A `RealPerceptionFilter` with the same `apply(detections, ego, scenario, iteration, mode)`
  interface returns each variant's branch.
* Same scenario builder as script 91, asserting Track B's 60 scenarios and 1,440 states. Buffer 4, a
  fresh planner per branch, the same scoring over 0.5–4.0 s against logged tracks.
* Branches: reference, cheap, full, cheap_nofp, full_nofp, full_s1 (cheap_s1 = cheap), cheap_iou50,
  full_iou50.
* Planner calls are de-duplicated: branches whose four buffered observations are identical (tokens,
  types, poses, velocities, sizes) share one result. Checks 6 and 7 establish the determinism this
  relies on.
* Order:
  1. reference for all states and the identity branch on 120 states;
  2. checks 6–7;
  3. all other branches.
* Detached, supervised, checkpointed per scenario. One heavy job at a time.

**C2** (`116_nuplan_real_cells.py`) → `results/final/nuplan_real_perception_cells.csv`.
* Rows: planner × loss × split × variant.
  * Losses: collision, clearance_shortfall, log_deviation, safety, scalar_J, with 83's weights.
  * Splits: all 34 logs; the 9 test logs.
  * Variants: transported (Track B raw), primary, nofp, s1, iou50.
* Columns:
  * states;
  * affected (|V| > 1e-9), V>0 and V<0 counts;
  * harm rate P(V<0 | V≠0);
  * ρ = Σmax(−V,0) / Σmax(V,0);
  * all-FULL reduction (ΣJ_cheap − ΣJ_full) / ΣJ_cheap;
  * oracle@20 reduction (the top 20% positive V over ΣJ_cheap);
  * selective extra share.
* 95% CIs for harm rate, ρ and both reductions: percentile, log-level bootstrap, 1,000 draws, seed 0.
* Descriptive: in-camera eligible tracks, tracks kept per mode, FPs per state.

**C3. Reading, fixed now.** Primary variant, all 1,440 states, evaluated on safety and on scalar_J.
* **Falsifier fires (B-F1)** on an aggregate if the harm rate is < 5% for **both** planners.
* **Consistent with the benchmark** if the harm rate is ≥ 20% **and** ρ ≥ 0.20 for **both** planners on
  **both** aggregates.
* **Intermediate** otherwise.
* Zero affected states counts as harm rate 0.
* The test split, the other losses and the sensitivities are reported under the same rule but do not
  decide the reading.
* The outcome is reported whatever it is.

**Limitations stated in advance.**
* Open-loop per state.
* Per-frame detection without tracking, so FPs flicker across the buffer.
* Lidar tracks serve as ground truth, and occluded or distant tracks are removed in both modes.
* Image–lidar offset ≤ 50 ms, uncompensated.
* Static classes pass through.
* Camera 1–8 URLs follow the user's rule and are verified by HEAD and directory contents.

**Outputs.**
* `results/final/nuplan_real_perception_cells.csv`
* `nuplan_real_perception_checks.json`
* `nuplan_real_perception_{idm,pdm_closed}_raw.csv`
* the overlays
* `docs/iclr_nuplan_real_perception.md`
* request ledgers under `results/raw/`

### 2026-09-14 15:03 — Task 5 amendment, before any branch is built or scored: ground plane of the FP lift

**Found.** A smoke test on 2 states of one val log, before detection on the full set, projected the logged
boxes into fetched images. Alignment is good: vehicles and bollards sit on their image boxes. The
pre-registered ground-plane check, run on DB boxes, disagreed with the registered lift:
* median bottom of vehicle boxes 3–30 m ahead, in the ego frame, over every 20th sweep of the benchmark
  windows: **−0.323 m** on train ∪ val (IQR −0.385 to −0.278, n 2,810);
* −0.357 m on test (reported only, not used).

The nuPlan ego frame's origin is the rear axle, ~0.32 m above the road. The registered plane z = 0
(camera height 1.52 m) would put every false positive about 18% too close; on true boxes the smoke-test
lift landed 4–6 m short.

**Change.**
* The FP lift intersects the plane z = g, where g is that train ∪ val median, recomputed inside 114 by the
  rule above. Camera height above the road ≈ 1.52 − g ≈ 1.85 m.
* No test unit and no decision value is involved.
* New informational check: the lift applied to true projected vehicle boxes, with its xy and range errors.

**Implementation details, design unchanged.**
* FP tokens are keyed by (mode, image, cached detection), not by variant, so identical branches can share
  one planner call.
* The per-object table is csv.gz, because the edge env has no pyarrow.
* The S1 FULL threshold is computed in 113.

**Operational.**
* **Bug 1 in 112:** `m.flags` resolved to the pandas DataFrame attribute, raising a TypeError after
  Camera 0's directory had been saved. Fixed with column access; the saved directory is reused, not
  re-fetched.
* **Bug 2 in 112:** the directory budget check counted the saved Camera 0 directory a second time and
  would have stopped at 401.7 MB. Fixed.
* **Metadata result:** 360,313,925 B of the 400 MB cap. All nine HEADs returned 200, `application/zip`,
  `Accept-Ranges: bytes`, with no login. Every shard's log set equals its metadata File group, so the
  Camera 2–8 URLs built by the user's rule are verified.
* **Fetch plan:** 12,921 members, 2,765,642,770 B (cap 3.5 GB).

### 2026-09-14 15:09 — Task 5 Stage A result: 12,921 CAM_F0 window images fetched by Range, all verified

**Fetch.**
* 12,921 members from the nine shards in 13,727 member requests. Every answer was 206 with the exact
  range, so there was no whole-archive fallback.
* **2,765,645,994 B** of image data, against the 3.5 GB cap and 2,765,642,770 B planned.
  * 806 members had a local extra field longer than the central one and needed one more small request.
  * 96,920 B were over-read in total: local header bytes past a member, where the local extra is shorter.
* All 12,921 inflate to the directory size, match the directory CRC32 and decode to 1080×1920×3.
* Local-header CRCs also match. No other camera, lidar or non-window member was requested.

**Metadata.** 110 requests, 360,313,925 B. Six directory chunk requests to Cameras 2, 3, 4, 6, 7 and 8
got `RemoteDisconnected` on a reused idle connection; each succeeded on retry, and the failed attempts
count 0 B.

**Stored.** Images: `~/datasets/nuplan/sensor_blobs_cam_f0/<log>/CAM_F0/<hash>.jpg`, 2.7 GB.

**Committed** under `results/raw/nuplan_task5/`:
* the shard URLs with sources;
* `dirs.json`, with HEAD, EOCD and per-shard log checks;
* `plan.json` and `plan_members.csv.gz`, with offsets and CRCs of the fetched members;
* `manifest.csv`, the per-member verification;
* `image_index.csv.gz`;
* the request ledgers.

**Kept local.** The full central-directory listings, `cd/`, 63 MB. They can be re-derived from the public
archives with a 360 MB read.

### 2026-09-14 15:29 — Task 5 Stage B: detection, projection and checks, reviewed before any CHEAP/FULL branch is scored

**B1 detection.** 12,921 images, no throttling before or after, fan at 100%.

| mode | dets ≥ 0.10 / ≥ 0.25 | total median | preprocess | inference | postprocess | CPU+GPU mJ/frame |
|---|---|---|---|---|---|---|
| 320 | 139,697 / 82,519 | 14.4 ms | 1.7 ms | 6.4 ms | 6.3 ms | 44.7 |
| 640 | 226,824 / 131,003 | 23.6 ms | 2.5 ms | 14.8 ms | 6.3 ms | 105.8 |

* JPEG decode: 15.6 ms.
* Energy is from the 300-frame dedicated passes.
* S1 FULL threshold on the 25 train ∪ val logs: **0.4766**.

**B2 time alignment.**
* |Δt| over 5,760 buffer iterations: median 25.0 ms, max 40.9 ms. None exceeds 50 ms.
* For every iteration, the DB's nearest CAM_F0 image was among the fetched ones.

**B3/B4 projection.**
* Distortion is monotone to the clamp radius in all 34 logs.
* FP lift plane (train ∪ val): z = −0.324 m, so the camera sits 1.84 m above the road.
* Informational ground check on all state iterations: median vehicle-bottom z −0.370 m.
* Lift applied to 10,303 true vehicle boxes at decision iterations:
  * median |range error| 0.30 m within 15 m, 0.81 m at 15–30 m, 6.3 m at 30–60 m;
  * median xy error 2.55 m, with a long tail (95th percentile 194 m) from slopes and near-horizon rays;
  * lifts beyond 80 m are dropped by the registered rule (4,136 FULL and 431 CHEAP FPs across all
    iterations).

**Matching (primary IoU 0.3).**
* Match IoU median: 0.664 at 320, 0.648 at 640; 5% quantiles 0.38 and 0.36.
* Per state: 18.5 eligible in-camera tracks; CHEAP keeps 5.4 and FULL 7.7; CHEAP adds 0.83 FPs and FULL
  1.68.
* S1 branch: keeps 5.8, adds 0.55 FPs. IoU-0.5 branches: keep 4.4 / 5.9, add 1.75 / 3.10 FPs.

**Recall on 26,611 eligible in-camera tracks at the 1,440 decision iterations** (threshold 0.25):

| class | recall 320 / 640, IoU 0.3 | recall 320 / 640, IoU 0.5 | transported model predicts 320 / 640 | KITTI measured 320 |
|---|---|---|---|---|
| all | 0.293 / 0.415 | 0.236 / 0.320 | 0.204 / 0.540 | 0.475 |
| vehicle | 0.469 / 0.617 | — | 0.297 / 0.592 | — |
| pedestrian | 0.080 / 0.169 | — | 0.091 / 0.481 | — |
| bicycle | 0.054 / 0.196 | — | 0.034 / 0.238 | — |

* **The transported profile exaggerates the fidelity gap on nuPlan.** It predicts a 0.34 recall gap
  against a measured 0.12. By distance, at 40–60 m it predicts 320 0.04 / 640 0.47; measured is 0.22 / 0.35.
* **Coupling:** P(640 | 320) = 0.936, and 1.9% of tracks are hit at 320 but missed at 640. KITTI: 0.969
  and 1.5%. The loss channel is present and slightly stronger.
* **The ≥ 10 px restriction is vacuous.** Every in-camera track projects ≥ 32.7 px, and nuPlan tracks
  reach only ~81 m.
* **Pedestrian recall is low** because many projected pedestrians are occluded (behind barriers, crowds,
  vehicles). The rule removes them from both modes.

**Overlays.** 20 saved; 8 inspected, covering Las Vegas, Boston and Pittsburgh and vehicles 17, 26, 28,
35, 38 and 45. Boxes align with the objects. Projected 3D hulls are somewhat wider than the 2D detections,
and occluded tracks project onto their occluders. No change was made after inspection.

### 2026-09-14 16:26 — Task 5 checks 6–7 passed for both planners; scoring started

Pre-registered checks 6 and 7, before any CHEAP/FULL branch was scored.

| check | IDM | PDM-Closed |
|---|---|---|
| reference re-run vs Track B's stored values, 1,440 states | 0 mismatches | 0 mismatches |
| identity observation equals reference, 1,440 states × 4 buffer iterations | equal | equal |
| identity planner output equals reference, 120 planned states | equal | equal |
| planner calls | 1,560 | 1,560 |
| run time | 18 min | 40 min |

The four checked fields are collision, min clearance and both log deviations.

This establishes three things:
* everything outside the observation filter is unchanged from Track B;
* the planners are deterministic, which the planner-call de-duplication relies on;
* the branch-construction path is transparent.

Part 2 (seven branches per planner, then cells) was launched automatically at 16:25, conditioned on both
checks passing.

**Operational.** `scripts/supervise_task5.sh` was edited (the 117 markdown step appended to part 2) while
part 1 was still running from that file. Bash reads a script by byte offset, so when part 1 ended it
executed the shifted tail of the edited file. That wrote a spurious "TASK5 PART2 DONE" line at 16:25:15
and possibly a fan-to-auto call. Part 2 sets the fan back to 100% when it starts (verified).
No computation was affected. Waits now key on the markdown step, and running supervisor files are no
longer edited.

### 2026-09-14 17:42 — Task 5 interim: IDM branches complete (PDM-Closed still running)

**Why this interim.** The user asked for results. They were computed with the registered 116 statistics
on IDM alone. Nothing can be changed by seeing them, and the PDM-Closed run continues untouched.

**Run.** 1,440 states, every branch computed a trajectory. 7,965 planner calls. Results shared through
identical observations, per branch: cheap 37, full 79, cheap_nofp 596, full_nofp 390, full_s1 240,
cheap_iou50 461, and the rest in `logs/task5_br_idm.log`.

**Primary variant, all 34 logs, IDM** (transported profile in brackets):

| loss | affected states | V+ / V− | harm rate [95% CI] | ρ [95% CI] | all-FULL reduction | oracle@20 reduction |
|---|---|---|---|---|---|---|
| safety | **6** (35) | 5 / 1 | 16.7% [0, 16.7] (31.4%) | 0.013 [0.013, 0.013] (0.36) | 0.5% (3.9%) | 0.5% (6.1%) |
| scalar_J | **25** (96) | 13 / 12 | 48.0% [28.6, 66.7] (40.6%) | 0.062 [0.012, 4.7] (0.37) | 0.2% (2.0%) | 0.2% (3.2%) |
| collision | 1 (19) | 1 / 0 | 0 (26.3%) | 0 (0.36) | 0.5% (4.1%) | 0.5% (6.4%) |

**What this settles.** Under real perception, IDM's decision value is about 6× sparser than under the
transported profile, and its harm mass is small.

**Consequence for the registered reading (C3).** It no longer depends on PDM-Closed:
* "Consistent" needs harm ≥ 20% **and** ρ ≥ 0.20 for **both** planners on **both** aggregates. IDM's
  safety result (16.7%, ρ 0.013) already fails it.
* B-F1 needs harm < 5% for **both** planners on an aggregate. IDM is at or above 16.7% on both
  aggregates, so it cannot fire.
* The primary reading is therefore **intermediate**, whatever PDM-Closed shows.
* Caveat: IDM's safety harm rate rests on 1 of 6 affected states.

**Sensitivities, IDM, all logs.**
* nofp: safety 3 affected (harm 1/3); scalar_J 11 affected (harm 45%, ρ 0.005).
* s1 (FULL threshold 0.477): safety 5 affected, 2 harmful carrying most of the mass (ρ 112); all-FULL
  safety reduction −0.5%.
* iou50: safety 4 affected, 0 harmful; scalar_J 28 affected (harm 46%, ρ 0.06).

## 2026-09-14 21:23 — Task 5 results: real perception on the nuPlan track (pre-registered 14:44)

Part 2 ran 17:41–20:48. All 7 branches computed a trajectory on all 1,440 states for both planners; there
were 7,965 planner calls per planner. Observations are planner-independent, so the counts of shared results
are the same for both planners. Cells and tables were written at 20:48.

**Reading (C3), primary, all 1,440 states: intermediate.**

| aggregate | harm PDM-Closed | harm IDM | ρ PDM-Closed | ρ IDM |
|---|---|---|---|---|
| safety | 44.4% | 16.7% | 0.33 | 0.013 |
| scalar_J | 37.5% | 48.0% | 0.35 | 0.06 |

* B-F1 does not fire on either aggregate.
* Consistency fails on IDM only; PDM-Closed clears both bars on both aggregates.

**Against the transported profile (Track B), safety, all logs.**
* PDM-Closed: affected states 70 → 45, harm 37.1% → 44.4%, ρ 0.30 → 0.33, all-FULL reduction 14.7% → 9.8%,
  oracle@20 21.1% → 14.7%.
* IDM: 35 → 6 affected states; all-FULL reduction 3.9% → 0.5%.
* scalar_J: PDM-Closed 175 → 104 affected (harm 37.5%, ρ 0.35); IDM 96 → 25 (harm 48.0%, ρ 0.06).

**Test split, 9 logs, primary.**
* PDM-Closed safety: 27 affected, harm 33.3%, ρ 0.36, all-FULL 25.2%, oracle@20 39.4%.
* IDM safety: 6 affected, harm 16.7%.
* Reading: intermediate.

**Sensitivities, all logs, reading under the same rule.**

| variant | reading | note |
|---|---|---|
| nofp | intermediate | PDM-Closed safety affected 45 → 6, scalar_J 104 → 28. Without false positives, PDM-Closed's value nearly vanishes. |
| s1 (FULL 0.477) | **consistent** | PDM-Closed safety harm 25.7%, ρ 0.25. The IDM part rests on 5 states with ρ 112. |
| iou50 | intermediate | PDM-Closed safety harm 47.1%, ρ 0.37. IDM safety harm 0 of 4. |

The transported variant, re-evaluated under this rule, reads consistent on all logs and intermediate on test.

**Interpretation, written in `docs/iclr_nuplan_real_perception.md`.**
* The transported profile exaggerated the recall gap: 0.34 predicted against 0.12 measured, and none inside 10 m.
* Under real perception, decision value is sparser and far more planner-dependent. PDM-Closed keeps
  benchmark-level sign variation; IDM is nearly inert.
* For PDM-Closed the value runs mostly through false positives.
* The paper should report this track per planner, with real perception primary and the transported profile as a
  labelled sensitivity. The conclusion is not rewritten to favour the paper.

**Operational.** A background wait of this session was killed when the session was resumed. The detached
supervisor was unaffected.

## 2026-09-14 21:27 — Task 6: figure data export (export only), first run stopped by an assertion

Task 6 exports figure data and registers nothing new. Script: `scripts/118_figure_exports.py`.
* **Source.** Every nuScenes frame is rebuilt through `rap.decision.build`'s functions, in its order.
* **Safeguard.** The rebuilt J_cheap, J_full and actions must equal the registered joined tables (mono and
  oracle, 3,376 frames each) before anything is written.

**Bug (export script only, no registered result affected).** The first run raised an AssertionError before
writing any file. My detection→GT pair re-derivation asserted that a GT counts as matched exactly when it is
assigned a detection. `rap.risk.match` defines matched differently:
* an unassigned GT keeps its best IoU with *any* detection, including one assigned to another GT;
* the pipeline counts a GT as detected when that best IoU is ≥ 0.5 (`per_object_error`, `add_perception_gain`).

**Fix.** The export now follows the existing definition.
* Such a GT is matched; its position and confidence come from its best-overlapping detection.
* It is flagged `det_shared_with_other_gt_cheap/full`, and the number of such rows is recorded in
  `fig_bev_constants.json`.
* False positives remain the detections `rap.risk.match` leaves unassigned.

### 2026-09-14 21:32 — Task 6 result: figure data exported, no registered result changed

**Safeguard.** Rebuilt frames equal the registered joined tables. Mono and oracle, 3,376 frames each:
J_cheap and J_full max |diff| 0.0, and 0 action mismatches.

**A. `results/final/fig_bev_objects.csv.gz`** — 12,773 rows.

| system · geometry | rows | V≠0 frames with rows | not exported: missed by both | not exported: FP in both (IoU ≥ 0.5 / ≥ 0.3) |
|---|---|---|---|---|
| q_brake mono | 3,079 | 481 of 486 | 1,464 | 298 / 335 |
| q_brake oracle | 1,942 | 290 of 295 | 806 | 196 / 223 |
| q_plan ADE oracle | 7,752 | 1,218 of 1,226 | 4,149 | 801 / 885 |

* The 5, 5 and 8 frames without rows hold only objects missed by both modes or FPs in both.
* Rows by type (brake mono / brake oracle / plan):

| type | brake mono | brake oracle | plan |
|---|---|---|---|
| fp_added_by_full | 1,001 | 695 | 2,326 |
| fp_removed_by_full | 349 | 218 | 916 |
| matched_both | 1,087 | 635 | 2,683 |
| miss_lost_by_full | 67 | 44 | 186 |
| miss_recovered_by_full | 575 | 350 | 1,641 |

* 667 rows are matched through a detection assigned to another GT, and flagged as such.
* `position_basis`: near_face (brake), box_centre (plan).
* Constants and definitions: `fig_bev_constants.json`.

**B. `results/final/fig_gallery/`** — 12 CAM_FRONT images copied byte-identical, 12 JSONs and an index.
* q_brake mono, 6 most negative V (−10.11 to −4.39) and 6 most positive (+57.0 to +4.44).
* scene-0032 frame 9 and scene-0048 frame 3 are excluded; every frame is from a different scene.
* Negatives 2–4 tie at −4.440; ties are broken by scene, then frame.
* Per the user's addition, each JSON also lists the GTs missed by both modes and the FPs present in both.
* The images are nuScenes data (CC BY-NC-SA 4.0), redistributed here for research.

**C. `results/final/fig_budget_curves.csv`**
* 856 per-cell rows (unit ms, every allocator and budget level, η with 95% CI and escalated fraction).
* 180 rows of medians over cells per track × signal × budget level.

## 2026-09-14 21:48 — Task 7 pre-registration: the nuPlan allocation track on real-perception decision values

Committed before any feature is rebuilt or any signal is scored. **No existing result file is modified**;
every output is new.

**Goal.** Make the benchmark's nuPlan cells consistent with the Task 5 real-perception track, primary variant.

**Labels.**
* V = J(CHEAP) − J(FULL), from `nuplan_real_perception_{pdm_closed,idm}_raw.csv`: columns `*_cheap` and
  `*_full` (0.25/0.25, IoU 0.3, FPs), costs from 83's `costs`.
* Cells: {PDM-Closed, IDM} × {safety, scalar_J}.
* Frozen log split from `configs/benchmark_splits.json`: 25 train ∪ val logs for fitting, 9 test logs scored once.

**Pre-escalation inputs, CHEAP branch only** (`scripts/119_nuplan_real_features.py`, pynuplan env).
* **Branch source.** The CHEAP-branch observation exactly as the planner received it: 115's
  `RealPerceptionFilter` on the logged tracks, i.e. kept tracks plus the real 320 false-positive agents.
* **Gate features.** 91's 18 gate features with 91's definitions, computed on that branch at the decision
  iteration; `d_n_cheap_fov` uses the CHEAP branch at iteration − 1. Provenance, registered in
  `rap.features` and checked with `assert_no_leakage`:
  * `cheap_det`: detections of the current iteration;
  * `cheap_prev`: `d_n_cheap_fov`;
  * `ego_state`: ego speed, ego acceleration, red lights. This is a new legal source, added to
    `LEGAL_SOURCES` in this commit.
* **R1 inputs.** 104's `track_vector`, the 25 nearest CHEAP-branch objects, 250 dims.
* **Cheap-side criticality.** `crit_cheap_sum` on the branch.
* **Uncertainty.** Excluded as privileged, not scored.
* **Diagnostics** (not deployable; must fail `assert_no_leakage`, a negative control):
  * reference criticality: 91's `crit_sum_gt` on the logged tracks;
  * ΔE as a missed-track count: |eligible in-camera tracks removed by CHEAP| − |removed by FULL| at the decision
    iteration;
  * E_risk: Σ criticality × (1[removed by CHEAP] − 1[removed by FULL]). Both ΔE and E_risk count misses only, as in 91.
* **Checks before scoring.**
  * The branch's object count equals the raw `n_tracks_cheap` on all 1,440 states.
  * Recomputed `crit_sum_gt` equals the existing benchmark column.

**Scoring** (`scripts/120_nuplan_real_allocation.py`, edge env) → `results/final/benchmark_table_nuplan_real.csv`.
* **Protocol.** 92's `evaluate`: exact tie expectation, 1,000 log-level bootstrap draws (seed 0), paired against
  random, draws with prize < ¼ of the full-sample prize dropped and counted. Quotas 10/20/30/50%.
* **Signals and splits.**

| signal | scored on |
|---|---|
| random | test and all |
| criticality_cheap | test and all |
| gate_ridge, gate_gbm (92's `gate_predictions`, fit on train ∪ val) | test only |
| R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf (103's `fit_score`) | test only |
| criticality_gt, dE_E1_fn_only, dE_E6_risk_weighted | test and all |
| oracle | test and all |

* **Columns.** nDG (η), CI, paired difference to random with CI, p ≤ random, responsive fraction, tie fraction.
  Per row: affected states on that split, V+ and V− counts.
* **Undefined rule.** nDG, its CI and the paired difference are **undefined** (NaN, with the reason) when the
  split's oracle prize at that quota is ≤ 1e-9, or when the split has **fewer than 10 affected states**.
  Gains and prizes are still written.

**Budget track** → `results/final/benchmark_budget_nuplan_real.csv`.
* **Protocol.** 93's two-level cascade protocol: budget = c₀ + f·c₁ for f ∈ {10, 20, 30, 50}%; the allocator's
  own overhead is charged; η is paired against random in the same draws.
* **Costs.** nuPlan's own Task 5 measurements (`results/raw/nuplan_task5/detect_summary.json`):

| mode | ms | mJ |
|---|---|---|
| 320 | 14.374 | 44.706 |
| 640 | 23.571 | 105.788 |

  These replace the KITTI profile. The definitions differ from the old costs, and the doc will say so:
  * ms is preprocess + inference + postprocess, without JPEG decode;
  * mJ is CPU+GPU rail power over idle.
* **Overheads.** Exactly those of the router run: `benchmark_budget_overheads_routers.json`, through 93's
  `signal_overhead` with the nuScenes feature-time proxy.
* **Signals.** random, oracle, criticality_cheap, gate_ridge, gate_gbm, gate_gbm_batched, R1 × 4. Diagnostics are
  written as infeasible.
* The same undefined rule applies.

**Reading** (descriptive; no pass/fail was registered for Task 7). `docs/iclr_nuplan_real_allocation.md` will
compare these cells with the detection-profile cells in `benchmark_table.csv`, `benchmark_table_routers.csv`
and `benchmark_budget_routers.csv`, on three things:
* which deployable signals beat random (paired lower bound > 0) at 20% quota and at the 20% ms budget;
* how many cells are undefined;
* the affected-state counts per split.

## 2026-09-14 21:59 — Task 7 results: the nuPlan allocation track on real-perception decision values

**Run.** Features: `results/raw/20260914_215224_nuplan_real_features`. Scoring: `*_nuplan_real_allocation`.
Outputs, all new; nothing existing was modified:
* `results/final/benchmark_table_nuplan_real.csv` (312 rows);
* `results/final/benchmark_budget_nuplan_real.csv` (416 rows);
* `docs/iclr_nuplan_real_allocation.md`.

**Checks.**
* The CHEAP-branch object count equals Task 5's `n_tracks_cheap` on 1,440/1,440 states.
* Reference criticality equals the existing column on 1,440/1,440 states.
* `assert_no_leakage` passes the 18 gate features and refuses crit_sum_gt, dE_E1_fn_only, dE_E6_risk_weighted and
  unc_proxy.

**Affected states, test / all** (detection profile in brackets):

| planner | safety | scalar_J |
|---|---|---|
| PDM-Closed | 27 / 45 (11 / 70) | 48 / 104 (42 / 175) |
| IDM | 6 / 6 (12 / 35) | 12 / 25 (26 / 96) |

IDM safety is **undefined** on both splits under the registered rule (< 10 affected states).

**Frame quota, test, 20%** — η, with the paired lower bound over random in brackets:

| cell | gate ridge | gate GBM | other |
|---|---|---|---|
| PDM-Closed safety | 0.99 [+0.58] | 0.99 [+0.64] | before: 0.78 [+0.20] and 0.66 [+0.13] |
| PDM-Closed scalar_J | 0.99 [+0.60] | 0.99 [+0.75] | — |
| IDM scalar_J | no win | no win | E_risk diagnostic 0.97 on 12 states |

* The gate now wins 2 of 4 nuPlan cells at 20% (before: 4 of 4).
* Across quotas, deployable wins on test: gates 18, R1 6, cheap-side criticality 1.
* E_risk on PDM-Closed falls from 0.89 to 0.14, consistent with the false-positive mechanism of Task 5.

**Budget.**
* 20% ms: only the batched GBM gate on PDM-Closed scalar_J beats random (η 0.57 [+0.015], escalates 5%). With the
  KITTI costs, nothing on nuPlan did.
* 20% mJ: nothing.
* 30–50%: the gates win on PDM-Closed (ms and mJ); gate ridge and R1-MLP-reg win on IDM scalar_J at 50%.

**Caveats recorded.**
* **Concentration.** 24 of 27 affected PDM-Closed safety test states come from one log (…_00152_00504), 17 from one
  scenario, and the gates rank that scenario first.
* **Dropped draws.** 330–368 of 1,000 bootstrap draws are dropped (the prize collapses without that log), so the
  intervals are conditional and too narrow.
* **Few training labels.** 18 affected PDM-Closed safety states in train ∪ val.
* **Descriptive sanity check.** Top-20% composition per gate and the ridge coefficient spread, computed from the
  saved scores, not a new registered analysis. It showed no sign of leakage.

## 2026-09-15 00:41 — Diagnosis: IDM is fed a route it cannot start from (harness bug; PDM-Closed unaffected)

**Question from review of Tasks 5 and 7.** Why is IDM nearly inert on nuPlan? Is it a code error or a real
property of the planner?

**Finding: a planner-input bug in our open-loop harness** (`82_nuplan_counterfactual.py`, and 115, which
reuses it). Perception, matching and scoring are not involved.

1. **The mechanism.** Every branch initialises `IDMPlanner` with `scenario.get_route_roadblock_ids()`, the
   route of the scenario.
   * The devkit's `IDMPlanner._get_starting_edge` looks for the ego's lane **only in the first two route
     roadblocks**, and otherwise takes the closest edge among them.
   * PDM-Closed runs tuPlan Garage's `route_roadblock_correction` on its first call; the devkit's IDM has no
     such step.
2. **Measured at all 1,440 benchmark states** (reference branch only, no perception involved;
   `results/raw/idm_route_diagnostics/route_geometry.csv`):
   * the ego is farther than 2 m from the first two route roadblocks in **836 states (58%, 42 scenarios)**;
   * in 193 of those (9 scenarios: pickup/dropoff, stationary) it lies on no route roadblock at all.

| ego position | states | IDM mean deviation from the log | IDM collision | PDM-Closed deviation / collision |
|---|---|---|---|---|
| within the first two roadblocks | 604 | 1.5 m | 4.5% | 2.0 m / 3.5% |
| beyond them | 836 | 30.2 m | 20.6% | 2.5 m / 5.3% |
| on no route roadblock | 193 | 104 m (median 47 m) | 13.5% | 3.1 m / 3.1% |

   * In the worst scenario (`e678935a`) IDM's plan starts 409 m from the ego.
   * Track B's route warnings ("could not find valid path") cover 245 states; the failures are broader
     than the warnings.
   * The position of a state within its scenario does not explain it: 17% of states are warned even at the
     first state.
3. **Consequence for registered results.**
   * **IDM-valid states** (ego within the first two roadblocks): real perception gives 0 affected safety
     states and 1 affected scalar_J state.
   * **The 836 affected states** hold **all** of IDM's real-perception decision value (6 safety, 24
     scalar_J).
   * **Detection-profile runs:** 22 of 35 IDM safety states fall in those 836 as well.
   * **Scope.** The IDM cells of Track B, Task 5 and Task 7, and the nuPlan IDM rows of the benchmark tables,
     therefore do not measure a functioning IDM. PDM-Closed's cells are unaffected: it corrects its route
     internally, and its deviation is 2–3 m everywhere.
   * **Caveat.** The 604 valid states are not a like-for-like subset (early-route geometry). They show where
     IDM works, not what its decision value is.
4. **Candidate fix, tried on train ∪ val states only** (`idm_fix_probe.py`, 21 states, reference branch).
   * Correct the route with tuPlan Garage's `route_roadblock_correction`, trim it to start at the ego's
     roadblock, and initialise IDM with that route. IDM's policy and parameters are unchanged.
   * Result: median deviation 6.3 → 1.7 m and maximum 413 → 13.9 m. On the 13.9 m state PDM-Closed deviates
     14.7 m too.
   * Correction alone fixes the off-route states but not the on-route states beyond the first two
     roadblocks (341 m remains on `c447cf02`). Trimming is what fixes those.

**Nothing registered is changed.** A corrected IDM rerun (Track B, Task 5, Task 7 IDM cells) has to be
pre-registered before it is scored.

## 2026-09-15 00:50 — Audit for similar errors, and pre-registration of the corrected IDM rerun

**Audit.** The error class is a published component fed inputs that break an assumption of its released code.
Checked, and cleared:

| component | check | result |
|---|---|---|
| IDM and PDM-Closed parameters | against the released YAML configs (devkit `idm_planner.yaml`, tuPlan Garage `pdm_closed_planner.yaml`) | identical |
| PDM-Closed's route | handling at a mid-scenario or off-route state | corrected internally by `route_roadblock_correction` on its first call; its reference deviation is 2–3 m everywhere |
| PKL's planner (Planner C, nuScenes): pixel→metre decoding and the frame of the true trajectory | tested on 400 **validation** rasters against the axis alternatives (`results/raw/idm_route_diagnostics/pkl_axis_probe.py`) | the pipeline's decoding (first axis = x) gives the lowest ADE: 2.39 m. y flipped: 2.61 m; axes swapped: 13.2 m; x flipped: 18.4 m. corr(x) = 0.916; the median forward position at 1 s is 4.60 m vs 4.63 m true; no false motion on 74 stationary frames |
| scoring | horizon alignment (trajectory time vs logged iteration), history buffer, traffic-light input | consistent |

PKL's planner still fails viability: ADE 2.39 m against 1.80 m for constant velocity on that chunk. That is a
property of the planner, not of the harness.

**The only error found is the IDM route input** (entry above).

**Fix.** A new `route_for()` in `82_nuplan_counterfactual.py`, also used by 115.
* For IDM, at every state: correct the scenario route with tuPlan Garage's `route_roadblock_correction`, then
  trim it to start at the ego's roadblock.
* PDM-Closed receives the scenario route unchanged, as before.
* IDM's policy and parameters are unchanged.
* `--no_route_fix` reproduces the original behaviour.
* `93_budget_allocation.py` gains `--reuse_overheads`, so the rerun keeps the measured allocator costs and no
  other cell moves.

**Rerun** (`scripts/supervise_idm_fix.sh`, detached, one job at a time). Everything that depends on IDM:
1. Track B IDM, 60 scenarios × 24 states;
2. 83 and 85;
3. 92 and 94;
4. 103, 93 (`--routers`, reused overheads) and 110;
5. Task 5 IDM reference and identity phase, then branches; 116 checks and cells; 117;
6. Task 7 (120);
7. 118 (budget curves).

The perception inputs and every design choice, variant and statistic stay as registered for Tracks B, 5 and 7.
The archived pre-fix outputs are in `results/archive/pre_idm_route_fix/`.

**Checks, fixed now.**
1. **After the Track B IDM run** (`121 --stage trackb`, stop on failure):
   * every branch computed a trajectory;
   * CHEAP, FULL and reference track counts equal the archive on all 1,440 states (the perception filter is
     untouched);
   * IDM's reference median mean log deviation ≤ 3 m;
   * ≤ 2% of states above 20 m (before the fix: median 2.4 m, 13.3% above 20 m).
2. **Task 5.** The IDM reference reproduces the corrected Track B values on all states, and the identity branch
   reproduces the reference (115's own checks).
3. **After the chain** (`121 --stage compare`, flag on failure): every PDM-Closed, nuScenes and KITTI row of the
   regenerated tables equals the archived row. The bootstrap streams run in cell order, and PDM-Closed precedes
   IDM.

**Readings.** Unchanged rules: Track B B-F1..B-F3; the Task 5 C3 reading; Task 7 descriptive. Results are reported
**before and after** the fix, labelled as a bug correction. The pre-fix readings are not overwritten in the log.

## 2026-09-15 03:42 — Results: corrected IDM rerun (pre-registered earlier today)

Chain `scripts/supervise_idm_fix.sh` ran 00:52–03:39. Every step exited 0 and every check passed.

**Checks.**

| check | result |
|---|---|
| Track B IDM reference (`check_trackb_idm.json`) | median log deviation **1.46 m** (was 2.44); **0%** above 20 m (was 13.3%); collisions **4.2%** (was 13.8%; PDM-Closed 4.5%) |
| perception filter | track counts equal the archive |
| Task 5 | IDM reference reproduces corrected Track B with 0 mismatches; identity equals reference |
| rows that must not move (`idm_route_fix_comparison.json`) | all PDM-Closed, nuScenes and KITTI rows identical in benchmark_table, benchmark_cells, benchmark_table_routers, benchmark_budget_routers, nuplan_real_perception_cells, benchmark_table_nuplan_real, benchmark_budget_nuplan_real and both PDM-Closed raw files |

**Changes (IDM only; full tables in `docs/iclr_idm_route_fix.md`).**
* **Track B, transported profile.**
  * safety: affected 35 → 47, harm 31.4% → 27.7%, all-FULL 3.9% → 16.8%, oracle@20 6.1% → 21.6%;
  * scalar_J: affected 96 → 105, harm 40.6% → 33.3%.
  * B-F1, B-F2 and B-F3 still do not fire.
  * Safety top-20 overlap 0.22 → 0.24; gamma 0.78 → 0.69.
* **Benchmark nuPlan IDM cells** (detection profile, test, 20%). Gate GBM no longer beats random: safety lower
  bound +0.27 → −0.20; scalar_J +0.26 → −0.05. The E_risk diagnostic still wins (+0.50). So the gate wins only the
  two PDM-Closed nuPlan cells.
* **Task 5, real perception** (primary, all).
  * IDM safety: 5 affected, harm 40%, ρ 0.055 (was 6 / 16.7% / 0.013);
  * IDM scalar_J: 30 affected, harm 33.3%, ρ 0.136 (was 25 / 48% / 0.062).
  * **Reading: intermediate, unchanged**, because IDM's ρ is below 0.20.
  * Non-deciding `nofp · test` now reads "falsifier fires" (≤ 1 affected safety state per planner).
* **Task 7.**
  * IDM safety is still undefined (4 affected test states).
  * IDM scalar_J (10 affected test states): R1-MLP-clf now beats random at every quota and every ms and mJ budget
    (η 0.94). Gate ridge's earlier wins at 30–50% are gone. This is fragile.

**Interpretation.**
* The fix changes IDM's transported-profile values substantially and removes the gate's IDM wins on the benchmark.
* It does not change the Task 5 reading or the planner-identity conclusion. Under real perception IDM is nearly
  insensitive to the 320/640 choice, and now that is not an artefact.
* Not rerun: the 6-state IDM wiring probe (superseded by the identity checks).
* The six hand-written reports with pre-fix IDM numbers now carry a pointer banner.
* The anonymous code release has not been updated.

## 2026-09-15 03:49 — Amendment: the IDM rerun chain missed one consumer; supplementary rerun pre-registered

**Bug in the rerun chain.** `supervise_idm_fix.sh` ran `93_budget_allocation.py` only with `--routers`.
* The registered primary budget run (no `--routers`) and its `--suffix _1thread` sensitivity also read the nuPlan
  IDM decision values.
* Their outputs are `benchmark_budget_two_level.csv` and `benchmark_budget_two_level_1thread.csv` (nuPlan IDM rows).
  Both, and the budget section that `94_benchmark_markdown.py` writes to `docs/benchmark_tables.md` from them, still
  hold pre-fix IDM values.
* Found while verifying outputs before commit. No row of these files had been read as a result.

**Consumer audit** (every script that reads an IDM-dependent file):

| scripts | why no rerun is needed |
|---|---|
| 90, 104, 114, 119 | use only scenario keys, state counts and CHEAP track counts, which the fix leaves unchanged (`check_trackb_idm.json`: track counts equal) |
| 102, 105 | write nothing that depends on IDM |
| 107 (R2) | KITTI and nuScenes only |
| figure scripts 53, 54, 63 | no nuPlan inputs |

**Supplementary rerun** (`scripts/supervise_idm_fix2.sh`, one job at a time):
1. 93 primary, with the archived `benchmark_budget_overheads.json`;
2. 93 `--suffix _1thread` under `OMP_NUM_THREADS=1`, with the archived `benchmark_budget_overheads_1thread.json`;
3. 94;
4. `121 --stage compare`.

**Checks, fixed now.** `121 --stage compare` now also requires:
* the non-IDM rows of both two-level files to be identical to the archive;
* both multi-fidelity files (brake and trajectory systems only) to be fully identical.

It also reports IDM before/after at the 20% budget.

## 2026-09-15 04:02 — Results: supplementary budget rerun (amendment above)

`scripts/supervise_idm_fix2.sh` ran 03:50–04:01. Steps 93 primary, 93 `_1thread`, 94 and `121 --stage compare` all
exited 0.

**Check passed.** All non-IDM rows are identical to the archive in 13 files. These are the 9 files of the first
comparison plus both two-level and both multi-fidelity budget tables.

**Result.** nuPlan IDM cells under measured cost, test split:
* In both the primary and 1-thread runs, no deployable allocator beats random at any budget level, in ms or mJ.
  The same was true before the fix.
* Only the η values move. Random at 20% ms goes from 0.134 to 0.198. Gate ridge at 20% mJ in the 1-thread run goes
  from 0.000 to 0.245 (lower bound −0.20).
* No budget claim changes.

**Docs.**
* `docs/benchmark_tables.md` is regenerated by 94 from the corrected tables.
* `docs/iclr_idm_route_fix.md` gains the budget rows and the corrected rerun scope.
* A seventh hand-written report, `iclr_corrected_results.md` (Track B IDM harm 41%, γ +0.70), gets the pointer
  banner.

## 2026-09-15 21:40 — Task 9 pre-registration: same architecture, different target (perception gain G vs decision value V)

**Question.** Does the allocation advantage come from the decision-value objective, or from the architecture? Every
learned allocator is retrained on a perception-gain label G with nothing else changed, and scored against V.

Written and committed before any G-target model is trained or scored. The official result files are not modified.
Code: `scripts/122_target_swap.py`.

### Architectures and inputs (reused unchanged)

| architecture | fitting code | inputs |
|---|---|---|
| `gate_ridge`, `gate_gbm` | 92's `gate_predictions` | core: 84's gate features; nuPlan real: the 18 `nr_*` features of 119 (120's `register_features`) |
| `R1_mlp_reg`, `R1_mlp_clf`, `R1_gbm_reg`, `R1_gbm_clf` | 103's `fit_score` | core: 225-dim CHEAP detection lists; nuPlan real: 250-dim branch track lists of 119 |

* Same hyperparameters and seeds.
* Fit on train ∪ val units of `configs/benchmark_splits.json`; scored once on test.

### Targets

**(a) V = J(CHEAP) − J(FULL), official models, not retrained.**
* Per-frame scores are loaded from the official runs:
  * R1, core: `20260915_014442_routers_r1`;
  * gates and R1, nuPlan real: `20260915_033311_nuplan_real_allocation`.
* Core gates: 92 does not save per-frame scores. They are regenerated by the same deterministic `gate_predictions`
  call on the same inputs (seed 0), i.e. the official model. Check S1 below must confirm this.

**(b) G, perception gain.** Same orientation as V: error of CHEAP minus error of FULL.

| G | core (KITTI, nuScenes) | nuPlan real (Task 5 primary branches) |
|---|---|---|
| **primary (deciding)** | `dE_E5_combined` (combined FN/FP/localisation error) | [missed(CHEAP) + FP(CHEAP)] − [missed(FULL) + FP(FULL)] |
| secondary | `dE_exact` (column `dE`, exact FN count) | missed-track change; must equal 119's `dE_E1_fn_only` |
| secondary | `dE_E6_risk_weighted` | 119's `dE_E6_risk_weighted` |

* **nuPlan primary G, definitions** (from `data/cache/nuplan_real/branches.pkl`, decision iteration):
  * missed = eligible in-camera tracks removed from that mode's branch;
  * FP = false-positive agents inserted into that branch.
  * Checked before registration: FP count = `n_tracks − n_tracks_nofp` on all 2,880 state-modes.
* **Core labels** come from the same frame tables 92 reads; no NaNs (checked).
* **G is a training label only.** Asserted: no G column is in any feature list or input matrix.
* **Model form:**
  * regression on G for `gate_ridge`, `gate_gbm`, `R1_*_reg`;
  * classification on 1[G > 1e-9] for `R1_*_clf`;
  * 103's rule applies unchanged: a single training class gives constant scores.

### Cells

The 14 official held-out cells:
* 10 core: nuScenes oracle and mono × brake / plan_ade / plan_fde; KITTI oracle and mono × brake / traj;
* 4 nuPlan real: PDM-Closed safety and scalar_J, IDM safety and scalar_J.

Task 7's undefined rule applies to every nuPlan row (< 10 affected test states, or zero prize), so IDM safety stays
undefined.

### Statistics

**Frame quotas.**
* nDG against V on the test split at 10/20/30/50%, with 92's exact tie expectation.
* 1,000 bootstrap draws, seed 0. In each draw the test units are resampled once per dataset (nuScenes scenes, KITTI
  sequences, nuPlan logs). The same resample is applied to every cell of that dataset, to both targets and to all
  architectures, so per-cell, paired and pooled statistics share one set of draws.
* A cell's draw is dropped when its prize falls below 25% of the full-sample prize (92's rule).

**Reported for each G variant × architecture × cell × quota:**
* nDG(V-target) and nDG(G-target), each with a 95% percentile CI and with its difference to random (mean and CI);
* the paired difference Δ = nDG(V) − nDG(G), with 95% CI and P(Δ ≤ 0);
* absolute loss reduction = nDG × oracle reduction at that quota, for both targets.

**Budget (not deciding).**
* The 20% measured-latency (ms) budget:
  * core with 93's `profile_costs`, and the router run's overheads via `signal_overhead`;
  * nuPlan real with Task 5's detector costs, as in 120.
* Each architecture's escalated fraction comes from its own official overhead, identical for (a) and (b).
* `gate_gbm` is charged its per-call overhead. `gate_gbm_batched` (same scores) is reported as well.
* Primary G only.

### Pooled test (deciding)

* **Scope:** quota 20%, primary G, 12 cells = 10 core + PDM-Closed safety + PDM-Closed scalar_J.
* **Statistic per architecture:** D = mean over the 12 cells of Δ_c.
* **Bootstrap:** in each draw, the mean of Δ_c over the cells not dropped in that draw; 95% percentile CI.
  * The number of draws with at least one cell left out is reported.
* **Also reported:** the number of cells with Δ_c > 0, and the numbers whose Δ CI lies entirely above or below 0.

### Reading rule (primary G only; secondary variants reported, not deciding)

| reading | condition |
|---|---|
| **objective matters** | pooled CI > 0 for at least one detection-list router variant (R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf) **and** at least one gate (gate_ridge, gate_gbm) |
| **architecture-driven** | pooled CI includes 0 for all six architectures |
| **mixed** | otherwise, including an architecture whose pooled CI lies entirely below 0 |

"At least one R1 variant" is lenient across four variants. Per-variant results are reported so this can be judged.

### Checks, before any G result is read

| check | requirement |
|---|---|
| **S1** | (a) reproduces the official test nDG to 3 decimals for every architecture × cell × quota (`benchmark_table.csv`, `benchmark_table_routers.csv`, `benchmark_table_nuplan_real.csv`) and the official 20% ms budget nDG (`benchmark_budget_routers.csv`, `benchmark_budget_nuplan_real.csv`). Otherwise the script stops before training any G model. |
| **S2** | the V stored with the official scores equals the recomputed V; frame order and keys are equal |
| **S3** | no G column in any feature list; the official leakage guards run unchanged |

**Descriptive, to explain any similarity:** train-split sign agreement P(sign G = sign V | both non-zero) per cell and
G variant, on train units and on train ∪ val, with the number of pairs.

### Outputs

* `results/final/benchmark_target_swap.csv`: per-cell rows.
* `results/final/benchmark_target_swap_summary.json`: sanity, agreement, pooled results, reading, and every row where
  the G-target beats the V-target significantly (Δ CI entirely below 0).
* `docs/iclr_target_swap.md`.

## 2026-09-15 22:02 — Task 9 results: target swap (pre-registered, commit 33f96a4)

`scripts/supervise_task9.sh` ran 21:43–22:00, exit 0, one job with the fan at 100%. Outputs:
`results/final/benchmark_target_swap.csv` (1,106 rows) and `benchmark_target_swap_summary.json`. Report:
`docs/iclr_target_swap.md`.

**Checks.**
* S1: the official V-target scores reproduce all 434 official nDG values (quotas and the 20% ms budget) to 3
  decimals; maximum absolute difference 1e-16.
* S2 and S3: asserted.
* In 382 of 1,000 draws at least one of the 12 pooled cells was left out at 20% (prize under a quarter of the
  full-sample prize).
  * Per cell: both PDM-Closed cells 319 draws each (their prize sits in one log), KITTI oracle traj 85, KITTI mono
    traj 29, every other cell 7 or fewer.

**Pooled test (primary G, 20%, 12 cells). Reading: architecture-driven.**

| architecture | mean Δ (V − G) | 95% CI | cells with V > G |
|---|---|---|---|
| gate_ridge | +0.118 | [−0.080, +0.275] | 9 |
| gate_gbm | +0.106 | [−0.058, +0.217] | 7 |
| R1_mlp_reg | +0.040 | [−0.083, +0.141] | 6 |
| R1_mlp_clf | +0.117 | [−0.037, +0.197] | 9 |
| R1_gbm_reg | −0.069 | [−0.186, +0.131] | 6 |
| R1_gbm_clf | +0.013 | [−0.070, +0.164] | 7 |

* Every pooled CI includes 0. The secondary G variants (`dE_exact`, `dE_E6_risk_weighted`) also read
  architecture-driven.
* Point estimates favour V for five of six architectures (both gates and R1_mlp_clf by about 0.12), but none is
  significant.

**Significant cells, primary G, 20%.**

| direction | cells |
|---|---|
| G-target beats V-target | R1_gbm_reg and R1_gbm_clf on both PDM-Closed cells (Δ −0.63 and about −0.41) |
| V-target beats G-target | gate_ridge and gate_gbm on both PDM-Closed cells (Δ +0.56 to +0.71); R1_mlp_clf on PDM-Closed scalar_J; gate_gbm and R1_mlp_clf on KITTI oracle traj; R1_gbm_clf on KITTI oracle brake |
| V-target beats G-target, not pooled | IDM scalar_J (10 affected test states): R1_mlp_reg, R1_mlp_clf (0.94 vs 0.03), R1_gbm_clf |

* At other quotas, G beats V on KITTI mono traj with gate_ridge (30% and 50%).
* No nuScenes cell differs significantly at 20% with the primary G.

**Budget, 20% ms (primary G).**
* On core cells the gates and GBM routers escalate no frames: their overhead exceeds the budget. There V and G are
  identical by construction.
* Where frames are escalated, V beats G significantly in 10 rows and G beats V in 1 (IDM scalar_J batched gate,
  Δ −0.001).

**Train-split sign agreement P(sign G = sign V | both ≠ 0), primary G.**

| cells | agreement |
|---|---|
| nuScenes | 0.49–0.58 |
| KITTI | 0.62–0.85 |
| PDM-Closed | 0.73 (11 pairs, safety), 0.68 (28 pairs, scalar_J) |

**Interpretation.**
* By the registered rule the pooled advantage cannot be attributed to the decision-value objective.
* The objective does matter in individual cells, in both directions:
  * **V-trained gates win PDM-Closed.** This rests on one scenario (Task 7 caveat).
  * **G-trained GBM routers win PDM-Closed.** A dense perception label is learnable where the V label has only 18 or
    56 affected training states.

## 2026-09-15 22:18 — Task 9 audit plan (requested after the results were read)

**Request.** After seeing the result, the user asked whether anything was computed wrongly. This audit looks for
computation errors only. It cannot change the registered reading. Anything beyond A1–A4 is labelled exploratory.

**Script:** `scripts/123_target_swap_audit.py`. It changes no result file.

| check | question |
|---|---|
| **A1** | The G-target code path (92's `gate_predictions` and 103's `fit_score` on 122's inputs and masks), given V as label, must reproduce the official V-target scores in every cell. Only then are the G-target fits known to differ from the official ones in the label alone. |
| **A2** | G labels per cell: orientation, non-zero shares, Spearman correlation with V on the fitting units, and the nDG of the label itself against V on test (the ceiling for a G-trained model) |
| **A3** | Are any G-target scores degenerate (constant, few distinct values, ties at the 20% cut)? Does the recomputed nDG equal 122's? |
| **A4** | per-cell V-target CIs at 20% from 122's joint bootstrap against the official per-cell CIs |
| **A5** | 122's pooled bootstrap re-run with the same seed must reproduce its CIs. Exploratory: the pooled CI on draws with all 12 cells present, core 10 only, and the two PDM-Closed cells only |

## 2026-09-15 22:27 — Task 9 audit results: no computation error found

`scripts/supervise_task9_audit.sh` ran 22:19–22:25, exit 0. Outputs in `results/raw/20260915_221911_target_swap_audit/`.

| check | result |
|---|---|
| **A1** G-target code path with V as label | reproduces the official V-target scores in 84/84 architecture × cell rows (83 bit-identical, 1 differing by 1e-16); nDG at 20% identical. The G-target models differ from the official ones in the label alone. |
| **A3** degenerate scores | no constant G-target scores; no row with a tie share above 0.5 at the 20% cut; all 252 G-target nDG values recomputed from the saved scores equal 122's. The only constant scores are two official V-target classifiers on IDM safety, which is undefined. |
| **A4** joint vs official per-cell bootstrap | V-target CI width ratio median 1.00 (range 0.58–1.09). At 20%, 13 of 78 rows beat random against 12 in the official tables. The joint bootstrap does not inflate per-cell uncertainty. |
| **A5** pooled bootstrap re-run with the same seed | reproduces 122's pooled CIs exactly for all six architectures |
| **A2** labels | equal the official diagnostic columns (label nDG equals the official `dE_*` rows, 30/30) |

**A2 in detail: how the G labels behave.**
* **nuScenes `dE_E5_combined` is negative on 49% of fitting frames** (mean −0.07 to −0.09), while `dE_exact` is positive
  (mean +0.85, 3% negative). So the sign convention is right. At 640, nuScenes detections carry more
  false-positive and localisation error, which outweighs the fewer misses in E5.
* **G carries little information about V.**
  * Spearman(G, V) on the fitting units: nuScenes −0.01 to 0.09, KITTI 0.06–0.22, PDM-Closed 0.02–0.06.
  * The G label itself, ranked against V on test at 20%: nuScenes −0.08 to 0.19, KITTI 0.16–0.66, PDM-Closed 0.30.

**Why the pooled test is null** (A5 exploratory, not part of the registered test):

| subset | gate_ridge | gate_gbm | R1_mlp_reg | R1_mlp_clf | R1_gbm_reg | R1_gbm_clf |
|---|---|---|---|---|---|---|
| core 10 cells, mean Δ [CI] | +0.000 [−0.111, +0.186] | +0.014 [−0.097, +0.144] | +0.061 [−0.083, +0.155] | +0.121 [−0.052, +0.209] | +0.043 [−0.108, +0.162] | +0.097 [−0.013, +0.186] |
| PDM-Closed 2 cells, mean Δ [CI] | +0.704 [+0.393, +0.924] | +0.566 [+0.315, +0.704] | −0.067 [−0.148, +0.072] | +0.098 [−0.057, +0.292] | −0.631 [−0.684, −0.531] | −0.410 [−0.547, −0.211] |
| 618 draws with all 12 cells present, CI | [+0.005, +0.284] | [+0.009, +0.231] | [−0.086, +0.123] | [−0.010, +0.195] | [−0.179, +0.033] | [−0.071, +0.105] |
| pooled mean, PDM-Closed in / out of the draw | 0.136 / 0.024 | 0.109 / 0.020 | 0.027 / 0.046 | 0.090 / 0.092 | −0.072 / 0.034 | 0.015 / 0.090 |

* **On the 10 core cells the target makes no difference to the gates** (Δ ≈ 0). The R1 classifiers lean towards V by
  0.10–0.12, not significantly.
* **The gates' pooled lean comes entirely from the two PDM-Closed cells**, whose prize sits in one scenario.
* **Restricting to draws that contain all 12 cells would put both gate CIs above 0.** That subset is conditional on
  resampling PDM-Closed's dominant log, so it is not an unbiased interval. It does not change the registered
  reading.
* **The official V-target allocators themselves beat random in only 12 of 78 rows at 20%**, none on nuScenes. There
  is little V-target advantage for a G-target to fall short of.

**Two design choices in the pre-registration to flag** (not errors; both were registered before the run):
1. **Resampling.** Units are resampled once per dataset per draw and shared by all cells of that dataset. The Task 9
   specification's "resample units within every cell in each draw" could also be read as independent resampling
   per cell. That would treat cells sharing the same scenes and sequences as independent and give a narrower
   pooled CI.
2. **Dropped cells.** The pooled mean in a draw averages only the cells not dropped by the 25%-prize rule. A 382-draw
   mixture of 10-cell and 12-cell means widens the CI.

Neither changes the reading without a new pre-registration.

## 2026-09-16 07:16 — Task 11 pre-registration: causal streaming allocation with a calibrated threshold

**Question.** The benchmark scores an allocator as a ranking that fills a budget over the whole test split. What
happens when the same scores are applied causally: one threshold frozen before the test stream, then applied in
timestamp order?

Written and committed before the script is run. No official result file is modified. CPU only, cached scores.
Code: `scripts/124_causal_threshold.py`. Outputs: `results/final/causal_threshold.csv`,
`docs/iclr_causal_threshold.md`.

### Cells and signals

* The 14 official held-out cells: 10 core (nuScenes oracle/mono × brake/plan_ade/plan_fde; KITTI oracle/mono ×
  brake/traj) and 4 nuPlan real-perception cells. IDM safety keeps Task 7's undefined rule (< 10 affected test
  states), so its nDG stays undefined; realised rates are still reported.
* Signals: `random`, `uncertainty`, `criticality_cheap`, `gate_ridge`, `gate_gbm`, `R1_mlp_reg`, `R1_mlp_clf`,
  `R1_gbm_reg`, `R1_gbm_clf`, and `R2_cnn_clf` on core cells.
* **Availability, checked before registering:**
  * `uncertainty` does not exist on the nuPlan real cells (no `unc_proxy` among 119's signals, and 120's leakage
    guard refuses it): reported as unavailable there.
  * **R2's cached scores cover the test frames only.** A threshold cannot be calibrated for it on validation or
    train ∪ val without re-running the image model, which needs a GPU. R2 therefore appears only in the official
    top-k row and in the sanity check; its V1 and V2 rows are written as unavailable with that reason.
* Target rates: 10, 20, 30, 50%.

### Calibration variants (both reported)

| variant | model | threshold calibrated on |
|---|---|---|
| **V1 (primary)** | refit on TRAIN units only (gates: 92's model spec; R1: 103's `fit_score`) | the VALIDATION units (nuScenes 16 scenes, KITTI 4 sequences, nuPlan 6 logs) |
| **V2** | the official model, fit on train ∪ val | grouped 5-fold cross-fitting inside train ∪ val: units sorted and assigned round-robin to folds, each fold scored by a model refit on the other four, thresholds calibrated on the pooled out-of-fold scores |

V2 isolates whether any change comes from refitting. For `uncertainty` and `criticality_cheap` nothing is fitted, so
V2 calibrates on all train ∪ val scores.

**tau_k.** On the calibration scores (n values, rounded to 9 decimals as `topk_expect` does), tau is the value at
rank ceil(k·n) from the top, and p = (k·n − #{s > tau}) / #{s = tau}, clipped to [0, 1]. On test a frame escalates
iff s > tau, or s = tau and a seeded Bernoulli(p) fires. This is `topk_expect`'s randomised tie convention, so the
target rate is hit in expectation.

### Policies on the test split

| policy | rule |
|---|---|
| **A** | frozen threshold: escalate iff the tie-broken score passes tau_k |
| **B** | A plus a causal budget: inputs of each test unit are processed in timestamp order; escalate iff A fires **and** the running count of escalations in that unit is < floor(1 + k·t), where t is the 1-based index of the current input. No future information and no knowledge of the unit's length |
| **C** | the official top-k over the whole test split: the hindsight reference row |

`random` escalates Bernoulli(k) under A, and the same draw under B's cap; it is simulated with 32 seeded
realisations and reported as their mean.

**Stream order.** Core cells: `(seq, frame)` ascending. nuPlan: within a log, scenarios ordered by their `t0` in
`configs/benchmark_nuplan_scenarios.csv`, then by iteration. Scenario windows inside one log can overlap in time
(5.2–9.1 s, recorded with the splits); this ordering is the registered convention.

### Reported per cell × signal × rate × policy × variant

* realised escalation rate on test: overall, and the minimum and maximum across test units;
* realised decision value: the raw gain Σ V over escalated frames, in loss units and as a share of the all-cheap
  loss on test;
* nDG against the official oracle prize at the target rate: the denominator is `topk_expect` at
  k_n = max(round(k·n), 1), the official rule;
* a paired unit-level bootstrap against `random` under the same policy: 1,000 draws, units resampled once per
  dataset per draw and shared by every cell, signal and policy, so all differences are paired; the official 25%
  prize filter applies and the number of dropped draws is reported.

### Measured-budget variant

93's costs and overheads (`profile_costs`, `signal_overhead`, `benchmark_budget_overheads_routers.json`; nuPlan
uses the KITTI profile, as in the official budget table). At the 20% and 50% ms budgets the target rate is the
feasible escalation share after the allocator's own overhead, k = max((budget − cheap − overhead)/full, 0), and
policy B is run at that rate. Two baseline rows are added:

* **all-cheap**: nothing escalates;
* **uniform full fidelity**: FULL on every frame, feasible only when the budget covers a full pass, that is from
  budget level 1 − cheap/full = 28.6% on KITTI and nuPlan and 34.3% on nuScenes (checked against 93's costs).

### Primary statistic and reading rule (registered before running)

At the 20% rate, variant V1, over the 10 core cells plus the two PDM-Closed cells, the mean paired difference
nDG(policy B) − nDG(official top-k), bootstrapped jointly. The primary averages over the six learned deployable
signals (both gates and the four R1 routers) as well as over the 12 cells; the per-signal pooled intervals are
reported beside it.

| reading | condition on the pooled 95% CI |
|---|---|
| **streaming holds** | lower bound above −0.05 |
| **streaming costs** | the interval lies entirely below −0.05 |
| **inconclusive** | otherwise |

### Checks

* **Sanity (deciding):** calibrating tau on the TEST scores instead, policy A's expected gain must reproduce the
  official top-k nDG to 3 decimals, for every cell, signal and rate.
* Refitting on train ∪ val must reproduce the official scores (the same code path as 92 and 103).
* Every realised rate that deviates from its target by more than 5 percentage points is listed.

## 2026-09-16 07:57 — Task 11 results: causal streaming allocation (pre-registered, commit 780efbe)

Run 07:27–07:53, exit 0. `results/final/causal_threshold.csv` (3,464 rows), report `docs/iclr_causal_threshold.md`.
No official result file was touched.

**Two fixes after the pre-registration commit, before any result was read.**
1. The debug path (`--max_cells`, added for a 2-cell smoke run) did not write the refit-check file; it does now.
2. The sanity variant calibrated tau at the fractional count k·n, while the official top-k row uses the integer
   max(round(k·n), 1). Three of 72 smoke rows then missed the 3-decimal target at the 30% rate. The sanity variant
   now calibrates at the official integer count, which is what the registered check meant.

**Checks.**
* Refitting on train ∪ val reproduces the official scores: maximum absolute difference 1.11e-16 over 64 rows.
* A threshold calibrated on the test scores reproduces the official top-k nDG to 3 decimals in **444 of 444 rows**.

**Primary statistic (V1, 20%, 12 cells, six learned signals): −0.089, CI [−0.143, −0.024]. Reading: inconclusive.**
The interval straddles the registered −0.05 line, so neither "streaming holds" nor "streaming costs" fires. Per
signal: gate_ridge −0.090 [−0.226, +0.097], gate_gbm −0.096 [−0.195, +0.035], R1_mlp_reg −0.015 [−0.088, +0.086],
R1_mlp_clf −0.095 [−0.176, −0.015], R1_gbm_reg −0.114 [−0.218, −0.020], R1_gbm_clf −0.124 [−0.207, −0.050]. 382 of
1,000 draws left a pooled cell out.

**What the decomposition shows (20%, pooled cells, learned signals).**

| policy | mean nDG | vs official |
|---|---|---|
| official top-k | +0.205 | — |
| A, frozen threshold | +0.217 | +0.012 |
| B, frozen threshold + causal cap | +0.105 | −0.100 |

* A beats the official ranking in 41 of 72 cell × signal pairs, B in 21. **The frozen threshold is free; the causal
  cap costs about half the decision value**, because escalations arrive in bursts that floor(1 + k·t) refuses.
* Refitting is not the cause: V2 (official models, cross-fitted threshold) gives −0.081 against V1's −0.095.
* Worst cells: PDM-Closed scalar_J −0.254, PDM-Closed safety −0.206, nuScenes oracle brake −0.183, KITTI oracle
  traj −0.180. Two cells gain: nuScenes oracle plan_ade +0.061, KITTI mono traj +0.018.

**Realised rates.** 1,195 of 1,840 rows miss the target by more than 5 pp (median 7.4–8.5 pp, worst 35.5 pp), in
both directions, and the miss grows with the target rate. Only `random`, which needs no calibration, holds its rate.

**Significance against random (V1, all rates).** Official 71 rows; A 106 rows (55 verdicts change); B 81 rows (64
change). Streaming promotes some allocators and demotes others.

**Measured budgets.** At 20% ms both gates and both GBM routers have zero feasible rate after their overhead, so
they escalate nothing; mean policy-B nDG is +0.026. At 50% ms the mean is +0.110, and **uniform full fidelity beats
the best allocator in 11 of 14 cells** on loss reduction (for example KITTI oracle traj 51.8% against 30.6%).

**Reading for the paper.** The benchmark's rows are hindsight top-k. The frozen threshold reproduces them, the
causal cap does not, and at deployable latency budgets the ranking often matters less than simply running FULL.

## 2026-09-16 08:02 — Task 12 pre-registration: statistics hardening

**Question.** Do the held-out conclusions survive without the prize denominator, without one dominant test unit,
and against trivial predictors?

Written and committed before the script runs. No official result file is modified. CPU only.
Code: `scripts/125_statistics_hardening.py`. Outputs: `results/final/statistics_hardening.csv`,
`docs/iclr_statistics.md`.

**Cells.** The 14 official held-out cells: 10 core and 4 nuPlan real-perception cells. Core cells are compared
against `benchmark_table.csv` and `benchmark_table_routers.csv`; nuPlan real cells against
`benchmark_table_nuplan_real.csv`.

**Signals.** Every signal the official table carries for that cell: `random`, `uncertainty`, `criticality_cheap`,
`criticality_gt`, `dE_exact`, `dE_E1..E6`, `PKL`, `TIP`, `gate_ridge`, `gate_gbm`, `oracle`, the four R1 routers and
`R2_cnn_clf` on core cells; on the nuPlan real cells the 12 signals of its own table. Cached scores throughout;
gates are the official deterministic fit.

### (a) Paired intervals on raw gain, not on nDG

For every cell, signal and quota (10/20/30/50%): the realised decision value of the signal minus that of random,

* in loss units: Δ = topk_expect(score) − k_n/n · ΣV, with the official k_n = max(round(q·n), 1);
* as a share of the all-cheap loss: Δ / Σ J_cheap on test.

A 1,000-draw unit-level paired bootstrap, units resampled once per dataset per draw and shared by every cell and
signal. **Every draw is kept**, since the quantity is not divided by the prize. The same statistic with the official
25% prize filter applied is reported in separate columns, together with the number of draws that filter would drop.

### (b) Leave-one-test-unit-out influence

At the 20% quota, models and thresholds unchanged, each test unit (scene, sequence or log) is dropped in turn and
nDG and the raw gain are recomputed on the remainder (k_n and the prize recomputed on the reduced split). Reported
per cell and signal: the minimum, the maximum, the unit whose removal moves nDG most, and the value with that unit
removed. Reported for every signal, and called out for the signals the paper counts as wins (the official rows with
`minus_random_lo` > 0 at 20%) and for the two PDM-Closed cells where the gates reach 0.99.

### (c) Trivial baselines, same protocol as the official signals

| baseline | core cells | nuPlan real cells |
|---|---|---|
| ego speed alone | `v_ego` | `nr_ego_speed` |
| number of cheap detections alone | `feat_n_det` | `nr_n_cheap` |
| cheap-side risk alone | `feat_crit_sum` | `nr_crit_cheap_sum` |
| largest cheap detection area alone | `feat_area_frac_max` | not available: the nuPlan real features carry no area |

All four quotas, the paired bootstrap against random, and the raw-gain statistic of (a). Note, registered here: on
core cells `feat_crit_sum` is exactly the column the official `criticality_cheap` signal uses, so that baseline
reproduces an official row by construction; the same holds for `nr_crit_cheap_sum` on the nuPlan real cells.

### (d) Harm as a share of all inputs

Per cell and split (test and all units): affected inputs, harmed inputs, harmed / affected (the existing reading)
and harmed / all inputs, with counts, where harmed means V < −1e-9.

### Check

Recomputing the official nDG through this script must match the released tables to 3 decimals for every cell,
signal and quota.

## 2026-09-16 08:11 — Task 12 results: statistics hardening (pre-registered, commit e627e60)

Run 08:04–08:08, exit 0. `results/final/statistics_hardening.csv` (2,552 rows), report `docs/iclr_statistics.md`.
No official result file was touched.

**Check.** The recomputed official nDG matches the released tables to 3 decimals in **980 of 980** comparisons.

**(a) Raw gain instead of nDG.** Paired unit bootstrap, every draw kept.

| quota | rows | beat random, every draw | beat random, 25% filter | verdicts changed |
|---|---|---|---|---|
| 10% | 294 | 18 | 28 | 10 |
| 20% | 294 | 18 | 24 | 6 |
| 30% | 294 | 28 | 34 | 6 |
| 50% | 294 | 36 | 55 | 19 |

* **The prize filter only ever adds wins**, never removes one, and it acts exactly where the prize sits in one unit
  (KITTI oracle traj 85 dropped draws; the PDM-Closed cells 319).
* At 20%, of the 12 official wins among 118 deployable pairs: **8 survive on raw gain, 4 do not** (PDM-Closed safety
  gate_ridge and gate_gbm, PDM-Closed scalar_J gate_gbm, IDM scalar_J R1_mlp_clf). The three gate rows have a lower
  bound of exactly 0.000, because draws without the dominant log give a difference of exactly zero.
* Surviving wins are worth 3.1–19.5% of the all-cheap loss on KITTI and 30.5% on PDM-Closed scalar_J (gate_ridge).

**(b) Leave-one-test-unit-out at 20%.** Of the 36 official win rows, 2 lose more than half their nDG and 9 have a
leave-one-out range wider than 0.2. **The PDM-Closed gates keep nDG 0.99 while losing 93% of their raw gain** when
log `2021.05.12.23.36.44_veh-35_00152_00504` is removed (162.9 → 12.0 on safety, 171.3 → 12.9 on scalar_J): the
prize collapses with the log, so the ratio cannot show the dependence. Largest swings: PDM-Closed safety R1_gbm_reg
+0.072 → −0.922; IDM scalar_J R1_gbm_clf +0.059 → +0.999; IDM scalar_J R1_mlp_clf 0.938 → 0.318; on core, trivial
ego speed on KITTI mono traj +0.554 → −0.260 without sequence 0008.

**(c) Trivial baselines.** **No trivial baseline beats random on raw gain at 20% (0 of 38 rows)**; across all quotas
only three rows do. But on nDG, **ego speed alone outranks every learned allocator on all four KITTI cells**
(0.402–0.784 against 0.220–0.575). On the nuPlan cells the learned allocators are clearly ahead of the trivial ones
(gates 0.99 against 0.29 for detection count), and that lead is the single-log one from (b). Cheap-side risk alone
reproduces the official `criticality_cheap` row by construction, as registered.

**(d) Harm shares.** Mean over cells on test: 39.4% of affected inputs are harmed, which is 8.9% of all inputs
(whole split: 40.9% and 8.7%). Range on test: 10.0–53.3% of affected, and 0.3–29.9% of all inputs.

**Reading for the paper.** Report the raw-gain interval beside nDG; drop or report the 25% prize filter, which only
inflates significance; state the single-log dependence of the nuPlan gate claims; and report both harm shares with
the trivial baselines in their own row.

## 2026-09-16 09:34 — Task 13 Part A pre-registration: consumer transfer matrix

**Question.** How much of an allocator's value survives when the downstream consumer changes?

Written and committed before the script runs. No official result file is modified. CPU only, cached scores.
Code: `scripts/126_consumer_transfer.py`. Outputs: `results/final/consumer_transfer.csv`,
`docs/iclr_consumer_transfer.md`.

**Groups and consumers.** One square matrix per (track, geometry):

| group | consumers |
|---|---|
| nuScenes oracle, nuScenes mono | brake (J), plan_ade (JC_ade), plan_fde (JC_fde) |
| KITTI oracle, KITTI mono | brake (J), traj (JB) |
| nuPlan real perception | PDM-Closed safety, PDM-Closed scalar_J, IDM scalar_J |

The nuPlan consumers are the real-perception cells, the ones the paper's held-out set uses; IDM safety is absent
because its nDG is undefined there (4 affected test states), which matches the consumer list.

**Signals.** The four cached R1 routers and the two gates.
* R1 scores come from the cached `results/raw/*_routers_r1/scores__*.npz` for core cells and from the
  nuPlan real-perception run for nuPlan. Key names differ between runs, so the identifier columns are detected
  from the file (`seq`/`frame` or `scenario`/`iteration`) rather than assumed.
* Gates are refit from the cached features with the official hyperparameters.

**Deviation, registered and reported.** The brief asks for gate scores out of fold by unit; the sanity check asks
the diagonal to reproduce the official held-out nDG to 3 decimals. These conflict, because the official gate is fit
on train ∪ val and scored on test. The official protocol is kept: a gate is fit on the train ∪ val units of the
consumer it is trained for and applied to the evaluation cell's frames. Every evaluated test unit is therefore
outside the fitting set, which is what out-of-fold protects against.

**Entry (A, B).** The allocator trained for consumer A, evaluated against consumer B's decision value on B's frozen
test split, at quotas 10/20/30/50% with the exact tie expectation.

**Frame coverage.** A gate model scores every frame of B. Cached R1 scores exist only for A's own frames: where A
does not cover a B test frame (nuScenes plan cells hold fewer frames than brake cells), that frame is given A's
minimum score, so it is never escalated, and the coverage fraction is reported per entry. Entries below full
coverage are flagged in the table.

**Reported per entry.** nDG; transfer regret nDG(A→B) − nDG(B→B); realised gain as a share of the all-cheap loss;
a paired unit bootstrap over 1,000 draws against random with **every draw kept**, with the 25% prize filter
verdict beside it (Task 12 showed the filter only ever adds wins); and whether A→B beats random.

**Per group:** the diagonal, the median transfer regret over off-diagonal entries, and how many off-diagonal
entries still beat random.

**Check.** The diagonal must reproduce the official held-out nDG to 3 decimals.

## 2026-09-16 09:45 — Task 13 Part A results: consumer transfer matrix (pre-registered, commit 3a656a5)

Run 09:39–09:42, exit 0. `results/final/consumer_transfer.csv` (845 rows), report `docs/iclr_consumer_transfer.md`.
No official result file was touched.

**Check.** The diagonal reproduces the official held-out nDG to 3 decimals in **312 of 312** comparisons.

**Note on the brief's premise.** It said the identifier columns differ between the two router runs. On disk all
three `*_routers_r1` runs carry `seq`/`frame` for the core cells and `scenario`/`iteration` for the nuPlan cells.
The script detects them anyway, as registered.

**Transfer regret** (nDG(A→B) − nDG(B→B), 528 off-diagonal entries over all quotas): median −0.002, IQR
[−0.082, +0.063], range −1.062 to +0.969; 38.3% of entries land within 0.05 nDG of the diagonal.

* **Core tracks: the consumer barely matters.** Median regret at 20% is −0.002 (KITTI mono), +0.009 (KITTI oracle),
  −0.036 (nuScenes mono), +0.009 (nuScenes oracle).
* **nuPlan is bimodal.** An allocator trained for IDM scores 0.000 on both PDM-Closed cells (regret −0.99), while
  allocators trained for PDM-Closed reach 0.94 on IDM scalar_J against IDM's own 0.000 (regret +0.94).
* **Off-diagonal entries beat random at least as often as the diagonal**: 18 of 132 against 9 of 78 at 20%, and the
  same pattern at every quota. 13 of the 18 are KITTI transfers in both directions.

**Reading.** Planner-conditionality is a property of the decision values, not of the allocators: V differs by
consumer, but the rankings these allocators produce mostly do not. Fitting one allocator per consumer is not
supported on the core tracks; on nuPlan the useful direction is PDM-Closed → IDM, never the reverse.

**Hygiene.** The 64 nuScenes plan→brake R1 entries cover 78.7% of the evaluation frames (cached scores exist only
for the training consumer's frames); uncovered frames are scored below the threshold. The 25% prize filter changes
66 of 840 verdicts, every one of them by adding a win, 64 of those on nuPlan: the Task 12 finding repeats here.

## 2026-09-16 09:47 — Task 13 Part B pre-registration: release hygiene, second pass

No experiment. The anonymous release is at `037fb53`; work happens in a clean clone and is pushed with the
anonymous identity only. Items 1, 2, 3, 6, 7 and 8 of the brief were already applied in Task 10 and are verified
again here rather than redone; item 5 is already satisfied (no Task 9 artefact exists in the release).

**What changes in this pass.**
1. **Add the new work the appendix cites** (item 4): `scripts/124_causal_threshold.py`,
   `scripts/125_statistics_hardening.py`, `scripts/126_consumer_transfer.py`, their three CSVs, and
   `docs/iclr_causal_threshold.md`, `docs/iclr_statistics.md`, `docs/iclr_consumer_transfer.md`.
2. **Three new cached-tier stages** (C14 causal threshold, C15 statistics hardening, C16 consumer transfer) so
   `--verify` covers them, plus README and script-index entries.
3. **One code change, registered here:** 124 and 125 hard-code the run directory names
   `20260915_014442_routers_r1` and `20260915_033311_nuplan_real_allocation`. The release ships exactly those
   runs, so results are unaffected, but a reviewer re-running the full tier would produce new run names and break
   them. Both are changed to pick the latest run of the tag, as 126 already does. This alters file lookup only;
   the three CSVs must reproduce byte-identically under `--verify`.
4. **Item 5 re-checked:** the final tree must contain no `122`, no `target_swap` artefact and no RESEARCH_LOG.
5. **Item 9:** the identity scan (names, emails, /home/, claude, session URLs) over the final tree, then push.

**Checks before pushing.** `python reproduce.py --tier cached --verify` on a copy must report every stage
identical, including the three new ones; the identity scan must come back empty.

## 2026-09-16 10:05 — Task 13 Part C pre-registration: gated nuPlan expansion

**Status: gate only.** Nothing is run against decision values in this part until the gate below is reported and
reviewed. If the gate opens, any expansion writes to a separate directory and no official result file changes.

**What the gate measures**, all from the existing selection rule and measured timings, not from estimates of
convenience:
1. **Pool size.** Every scenario in the nuPlan mini split the existing rule admits, and how many logs they span,
   against the frozen 60 scenarios over 34 logs. Recorded as `logs/task13c_enum_uncapped.json`.
2. **Reproducibility of the rule.** Whether the first 60 scenarios of an uncapped enumeration reproduce the frozen
   set. A capped probe (cap 1500) already covers 35 logs in its first 60 against the frozen set's 34, so the
   builder's order appears to depend on the cap itself. If the frozen 60 are not a prefix of the enumeration, an
   expansion cannot be defined by raising `limit_total_scenarios`, and a new explicitly listed target set must be
   frozen before any outcome is seen.
3. **Download.** Camera-image volume for the new scenario windows, from the measured 12,921 images / 2.766 GB for
   the current 60 windows, against free disk. The Task 5 fetch ran under a registered 3.5 GB member-byte cap; a
   larger fetch needs that cap raised, which is a change to the registered data policy and is recorded here.
4. **Wall clock.** From the measured Task 5 chain on 60 scenarios / 1,440 states: fetch 12.4 min, detection
   12.6 min, projection 5.4 min, IDM reference 18.3 min, PDM-Closed reference 39.8 min, IDM branches 76.3 min,
   PDM-Closed branches 186.2 min, cells and markdown 0.7 min.

**Decision rule, fixed before the numbers are read.** If the additional scenarios admitted by the existing rule
need more than 12 hours of compute, or their images do not fit on free disk, STOP and report the gate only. A
feasible bounded subset may be reported as information, but is not run under this pre-registration.

**Why this matters for the paper.** The nuPlan track's prize mass concentrates in a single test log. An expansion
is worth running only if it adds test logs; adding scenarios inside the existing 34 logs would not address the
concentration. Whether it does is an outcome, not a gate criterion.

## 2026-09-16 10:22 — Correction to the Task 13 Part C pre-registration (commit 8e175d2)

The motivation sentence in that entry said the nuPlan track's prize mass "concentrates in a single test log".
That was written from memory and is not what the data says. Checked against the Task 12 influence section
(`results/final/statistics_hardening.csv`): nuPlan has 9 test units, and 17 of its 37 rows with positive nDG fall
to <= 0 when a single test log is dropped; the same count is 10 of 92 on KITTI and 36 of 99 on nuScenes.

The defensible statement is **single-test-unit fragility, worst on nuPlan**, not that the prize sits in one log.
The gate criteria and the decision rule of that pre-registration are unchanged.

## 2026-09-16 10:15 — Task 13 Part D pre-registration: does a perception-centric objective pick a different allocator?

**Relation to Task 9.** Task 9 asked which *training target* produces a better model and returned a pre-registered
null. That null stands and is not revisited. This asks a different question: with the method pool and the protocol
held fixed, which method does each *evaluation objective* select?

**Objectives.** Both use the existing selection protocol, the exact tie expectation (`topk_expect`) and the
cluster bootstrap over units, on the test split of the 14 official held-out cells, at quotas 10/20/30/50%.
* `E_dec`: realised decision value / decision-value oracle prize at the budget — the official nDG.
* `E_perc`: identical, with the per-input value replaced by a perception gain and normalised by the
  perception-gain oracle at the same budget. Run twice: `G^dE = dE_E1_fn_only` (missed objects) and
  `E_risk = dE_E6_risk_weighted`.

**Method pool, in three tiers, reported separately; the headline is tiers 1 and 3.**
1. Target-free, non-circular core: `random`, `uncertainty`, `criticality_cheap`, `trivial_ego_speed`.
2. V-trained, **flagged as advantaged under E_dec**: `gate_ridge`, `gate_gbm`, the four R1 routers, `R2_cnn_clf`.
3. Perception diagnostics: `dE_exact`, `dE_E1_fn_only`, `dE_E6_risk_weighted`, `PKL`, `TIP`.

Known unavailability, recorded now so it cannot be mistaken for a result: on the nuPlan real-perception cells
`uncertainty` (no cheap-detection uncertainty in those features), `R2_cnn_clf` (trained on nuScenes and KITTI
images), `dE_exact`, `PKL` and `TIP` are undefined; on KITTI `PKL` and `TIP` are undefined. Under `E_perc` with
value `G`, the signal equal to `G` scores 1.0 by construction; that is expected, not a finding.

**Reported per cell and budget.** Kendall tau between the two induced method rankings; whether the argmax differs,
and in how many cells; the **selection regret** — decision value realised by the method `E_perc` selects minus
that of the method `E_dec` selects, in nDG and as a share of the all-cheap loss, with a paired unit bootstrap of
1,000 draws where **every draw is kept**, the official 25% prize filter reported alongside as a count, never
substituted; and how often the `E_perc` winner is worse than random on decision value. Selection is fixed from the
point estimate; the bootstrap measures the two selected methods.

**Sanity check to report.** The `E_dec` column must reproduce the official held-out nDG to 3 decimals, against
`benchmark_table.csv`, `benchmark_table_routers.csv` and `benchmark_table_nuplan_real.csv`. The run asserts this.

**Outputs.** `results/final/objective_swap.csv` (sections `score`, `compare`, `sanity`) and
`docs/iclr_objective_swap.md`. No official result file is modified. Report whatever comes out.

## 2026-09-16 11:15 — Task 13 Part C results: the expansion gate is CLOSED

Enumeration recorded in `logs/task13c_enum_uncapped.json` (the existing rule: `ScenarioFilter` with every filter
`None`, `shuffle=False`, uncapped; 21.6 s).

| quantity | value |
|---|---|
| scenarios the rule admits | 219,607 over 54 logs (mini split has 64 db files) |
| currently frozen | 60 scenarios over 34 logs |
| **additional** | **219,547 scenarios, 20 logs** |
| scenarios per log | median 4,000, max 7,200 |
| frozen tokens all present in the pool | yes |
| **the frozen 60 are the first 60 of the enumeration** | **no** — the first 60 span 36 logs, not 34 |

**Measured per-scenario cost** from the Task 5 chain on 60 scenarios / 1,440 states: fetch 12.4 min, detection
12.6, projection 5.4, IDM reference 18.3, PDM-Closed reference 39.8, IDM branches 76.3, PDM-Closed branches
186.2, cells and markdown 0.7 — **351.7 min for 60 scenarios, i.e. 5.86 min per scenario** and 46.1 MB of CAM_F0
images per 20 s window (2.766 GB for the current 60).

**Gate arithmetic.** Running the additional scenarios the rule admits would take 219,547 × 5.86 min ≈ **21,400
hours** and need ≈ **10.1 TB** of camera images against **102 GB** free. Both limits of the pre-registered
decision rule are exceeded by orders of magnitude, so per that rule: **STOP, report the gate only.** Nothing was
run, and no directory was created for it.

**Two findings worth carrying forward, neither of them a result.**
1. **The existing rule cannot be extended by raising its cap.** The frozen 60 are not a prefix of the uncapped
   enumeration (the first 60 span 36 logs; a cap of 1,500 gives 35), so the builder's order depends on the cap
   itself. Any expansion must freeze an explicit token list before any outcome is seen — expressible in the same
   machinery through `ScenarioFilter(scenario_tokens=...)`.
2. **A bounded expansion is affordable, and is not authorised by this pre-registration.** At the measured rate,
   12 hours buys about **122 scenarios** (≈5.6 GB of images, which fits). That would also require raising the
   registered 3.5 GB member-byte fetch cap. Reported as information for a future decision.

## 2026-09-16 11:20 — Task 13 Part D results: perception-centric evaluation selects a different allocator

Pre-registered at commit `2f1c219`, run afterwards. Output `results/final/objective_swap.csv` (3,388 rows),
report `docs/iclr_objective_swap.md`. No official result file was touched.

**Sanity.** `E_dec` reproduces the official held-out nDG in **688 of 688 comparable rows** (max abs diff
1.1e-16). The other 40 sanity rows have a NaN official value: they are the nuPlan IDM safety cell, which the
benchmark declares undefined (4 affected test states < 10). My script computes a number there because its prize
is positive; the cell is flagged `ndg_defined=False` and is excluded from every summary. The printed sanity line
says 728/728 because NaN comparisons fail silently — corrected here and in the report.

**Result (headline pool = tiers 1 and 3, 13 defined cells × 4 budgets = 52 per variant).**

| | `E_perc_dE` | `E_perc_risk` |
|---|---|---|
| argmax differs | 42/52 | 40/52 |
| Kendall tau, median | +0.171 | 0.000 |
| selection regret, median nDG | −0.104 | −0.122 |
| worse than random on decision value | 16/52 | 22/52 |
| regret intervals excluding zero | 4/52 | **0/52** |

Under `E_perc` the winner is a perception diagnostic in 104/104; under `E_dec` it splits 52 target-free / 52
diagnostics, with `trivial_ego_speed` winning 34. With the V-trained tier added, 87/104 argmaxes differ, median
regret −0.229, worse-than-random 38/104, and `E_dec` picks a V-trained method in 44/104.

**Reading, and its limit.** The objective changes the selected method in most cells, and the perception-selected
method often fails to beat random on the decision. But with 24, 6 and 9 test units the per-cell regret is mostly
not separable from zero — 4 of 52 intervals exclude it for the missed-object variant and none for the
risk-weighted one. What the data supports is the disagreement in *selection*; the *size* of the loss is not
established. Reported as it came out.

## 2026-09-16 14:55 — Task 14: ship Part D, and harden the verifier's JSON comparison

**Part D artefacts shipped** to the anonymous release as cached-tier stage C17, with README and script-index
entries: `scripts/127_objective_swap.py`, `results/final/objective_swap.csv`, `docs/iclr_objective_swap.md`.

**Two corrections to Part D, made before shipping.**
1. The `compare` rows carried no definedness flag. Regret is measured in decision value, so the nuPlan IDM
   safety cell — undefined under `E_dec` with 4 affected states — produced regret numbers that looked usable.
   Added `e_dec_defined` and `n_affected_dec` to every compare row: 16 of 224 rows are now flagged False, all of
   them that cell. Note the flag is objective-specific by construction: under both `E_perc` variants the same
   cell *is* defined, because the perception gain is non-zero on many more inputs.
2. The script printed `sanity: 728/728`, folding in 40 rows whose official value is NaN, because a NaN
   comparison fails silently. It now prints 688/688 comparable rows plus the count with no official value. The
   report and this log already carried the corrected figure; the script did not. The table is byte-identical
   before and after this change, checked with `cmp`.

**Verifier.** `reproduce.py`'s JSON branch decided equality with `round(x, 9)`, which turns a difference in the
last bits into a verdict whenever two values straddle a rounding boundary. Replaced with a recursive comparison:
NaN equals NaN in the same position, numbers compare at `rtol=1e-9, atol=1e-12`, booleans by identity, dicts by
key set, lists by order. The CSV branch already used exactly this tolerance and is unchanged, so the documented
platform differences for C2, C12 and C14 are unaffected — they are orders of magnitude larger.

**What I could not reproduce.** The task described C10 reporting DIFFERS on an unchanged tree. On this machine it
does not: `reproduce.py --tier cached --verify --only C10` reports `identical` for both outputs, and the shipped
and regenerated `nuplan_real_perception_checks.json` are byte-identical (27,881 bytes, same single `NaN` token,
zero fields differing at rel 1e-9). So the change above is a robustness fix against the described mechanism, not
a confirmed repair of an observed failure here. Recorded rather than presented as a fix.

## 2026-09-16 15:40 — Task 14 results: 17-stage verify, and a false claim of mine corrected

**Per-stage verdicts** (`reproduce.py --tier cached --verify`, 17 stages, fresh clone, rebuilt env):
C1–C13 identical, **C14 DIFFERS**, C15 identical, C16 identical, **C17 identical**. Exit 1 from C14 alone.

* **C10 is identical**, and was identical before the comparator change too: `--only C10 --verify` on the
  unpatched tree already reported `identical`, with byte-identical files. The DIFFERS described in the task does
  not reproduce here, so the comparator change is a robustness fix, not a confirmed repair.
* **C17 reproduces byte-identically** and prints the corrected sanity line: 688/688 comparable rows, 40 further
  rows with no official value.
* **C2, C12, C15, C16 are identical on this machine.** The README documents C2 and C12 as differences that *can*
  appear on other platforms, so this is consistent with it.

**C14, and a claim I got wrong.** In Task 13 I wrote into the release README that "each number quoted in
`docs/iclr_causal_threshold.md` … is unchanged at the three decimals it is quoted to". That is false. This run
moves 72 of 3,464 rows (Task 13's moved 71), all of them learned signals, 45 with `ndg` NaN in the undefined
cell — a nondeterministic refit, not a data change. Checking every pooled row at 3 dp:

* the pooled headline is stable: −0.089007 → −0.089056, CI [−0.142568, −0.023583] → [−0.142610, −0.023838], i.e.
  −0.089 [−0.143, −0.024] either way;
* `gate_ridge`, `gate_gbm`, `R1_mlp_reg`, `R1_gbm_reg`, `R1_gbm_clf` are bit-stable;
* **`R1_mlp_clf`'s upper bound moves −0.015 → −0.016**, and the report quotes that endpoint.

The README note and the report now say this. Also recorded: the per-policy means and the "A beats the official
ranking in 41 of 72 pairs, B in 21" counts are **not stored** in `causal_threshold.csv` and script 124 does not
emit them, so `--verify` cannot cover them; a like-for-like recomputation moved by one pair between the two runs.
I could not reproduce the report's exact 72-pair selection from the shipped tables in three attempts (I get 78
pairs joining against the official eta, and the report's "official top-k" appears to be 124's own policy-C
recomputation), so I state only what was checked rather than restating that table.

**A recurring bug of mine, twice now.** Joining on the nuPlan geometry `"n/a"` fails silently because `read_csv`
parses it as NaN; it cost 12 of 72 pairs here. Task 9 hit the same thing, and script 127 already guards with
`fillna("n/a")`. Worth a helper rather than a third occurrence.

## 2026-09-16 17:10 — Task 16 Part A pre-registration: skipping cost accounting for the pixel router

**What changes.** A second cost-accounting variant, run alongside the shipped cascade and never in its place. Nothing
is retrained; R2's model and its cached scores (`results/raw/20260914_125344_router_r2`) are unchanged. No official
result file is modified.

* **Cascade (shipped).** Per-input cost Cs + Cc + f·Cf, so the escalated share is f_c = max((B − Cc − Cs)/Cf, 0)
  (`93_budget_allocation.py:342`).
* **Skipping.** An escalated input runs FULL instead of CHEAP: Cs + (1−f)·Cc + f·Cf = Cs + Cc + f·(Cf − Cc), so
  f_s = clip((B − Cc − Cs)/(Cf − Cc), 0, 1).
* **Budget** B = Cc + q·Cf for q in {10, 20, 30, 50}% (`93:317`), the benchmark's own, so both variants spend the
  same budget.
* **Constants**, exactly those of the shipped budget track. Cc, Cf come from `profile_costs` (`93:56`): median
  end-to-end latency and GPU-rail energy per frame. Cs comes from `signal_overhead` for R2 (`93:168-169`): resize,
  host-to-device copy and engine call, with energy = Cs ms × CPU+GPU rails over idle. **Image decoding is charged in
  neither variant.** The profiler decodes images before its clock starts (`01_profile_jetson.py:75-85,153`), and
  R2's cost excludes decoding. Under skipping an escalated input still needs a decoded image that nobody pays
  for. That is stated, not changed.
* **Decision values are the same in both designs.** Escalation replaces CHEAP's output with FULL's either way, so
  V = J(CHEAP) − J(FULL) is unchanged. Only the cost differs.

**Eligibility.** R2 only: it reads the raw camera frame, which exists before CHEAP runs. R1 reads CHEAP's detection
list (`103_routers_r1.py:43-56`), and the gates read features of CHEAP's detections, so neither can run before
CHEAP. Both are structurally ineligible and are not implemented.

**Expected shares.** This is arithmetic on the measured constants, not an outcome:

| track | unit | Cc | Cf | Cs | cascade 10/20/30/50% | skipping 10/20/30/50% |
|---|---|---|---|---|---|---|
| nuScenes | ms | 12.710 | 19.351 | 9.766 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 (Cs + Cc = 22.476 > B = 22.386 at 50%) |
| KITTI | ms | 13.179 | 18.469 | 4.817 | 0 / 0 / 3.92 / 23.92% | 0 / 0 / **13.69 / 83.51%** |
| nuScenes | mJ | 23.706 | 66.661 | 15.792 | 0 / 0 / 6.31 / 26.31% | 0 / 0 / **9.79 / 40.83%** |
| KITTI | mJ | 13.858 | 33.050 | 7.789 | 0 / 0 / 6.43 / 26.43% | 0 / 0 / **11.08 / 45.52%** |

The task's expected ms shares are confirmed. The energy constants support skipping, so the energy budgets are
evaluated too. 28 (cell, unit, budget) rows have a positive skipping share: KITTI ms 4 cells × 2 budgets,
KITTI mJ 4 × 2, nuScenes mJ 6 × 2.

**Validation gate, run first; the script stops if it fails.**
1. The cascade branch must reproduce `escalated_frac` of all 80 R2 rows in `results/final/benchmark_budget_routers.csv`
   exactly, including 0.0% / 0.0% at the 20% ms budget and 0.0% / 23.9% at the 50% ms budget (nuScenes / KITTI).
2. It must also reproduce the point `eta` of every R2 row with a positive share, to 1e-9 relative. This checks the
   gain and oracle code path against the shipped convention, whose denominator is the oracle at the nominal
   share q (`93:328`).

**Evaluation, for every row with f_s > 0.**
* Test split of `configs/benchmark_splits.json`, the benchmark's cells, R2 scores aligned by (seq, frame).
* k = floor(f_s·n + 1e-9) (`93:321`). R2's gain uses the exact tie expectation (`topk_expect`). Random's gain is
  k/n · ΣV.
* **Denominator: the oracle escalating the same k**, topk_expect(V, V, k), i.e. the budget-constrained oracle at
  the realised share. It equals the capacity oracle Σmax(V, 0) exactly when #positive ≤ k ≤ #non-negative. It is
  smaller when the share cannot reach every positive input, and smaller again when k exceeds the non-negative
  inputs, which forces negative V in. For every row the capacity oracle, the share of non-negative V, the share
  of positive V and a flag for each case are reported. The shipped nominal-share denominator is not used,
  because f_s can exceed q.
* **Paired cluster bootstrap.** 1,000 draws, seed 0, test units resampled with replacement. R2, random and the
  oracle share the same draws, with k_t = floor(f_s·n_t + 1e-9). **Every draw is kept**: no prize filter. Draws
  with an oracle ≤ 1e-9 have an undefined nDG and are counted; they still count in the gain difference.
* **Gain as a share of the all-cheap loss**: gain / ΣJ_cheap on the test split, for R2 and for random.
* **Beats random** when the 2.5th percentile of the paired gain difference (R2 − random) over all 1,000 draws is
  above 0. **Loses** when the 97.5th percentile is below 0. On draws with a positive oracle this has the same sign
  as the nDG difference.

**Expectation, registered.** Skipping changes how many inputs R2 can escalate, not which ones it picks. R2 beats
random nowhere under the cascade, even when charged nothing, so it should track random here too. If it beats
random, that is a finding and will be stated plainly.

**Outputs.** `scripts/128_skip_accounting.py`; `results/final/skip_accounting.csv` (sections `validation`,
`shares`, `evaluation`); `docs/iclr_skip_accounting.md`. The anonymous release gets the same files as cached-tier
stage C18.

**Part B**, no experiment: relabel R2 and correct items 1, 3, 4, 5 and 6 of `docs/iclr_router_implementation.md` in
`scripts/107_router_r2.py` and `docs/iclr_routers.md`. Items 2 and 7 are left alone.

## 2026-09-16 21:50 — Task 16 Part A: the validation gate fired on the smoke run (reading bug, criterion unchanged)

The first run (`--debug --nboot 50`, run `20260916_214811_skip_accounting`) stopped at the registered gate: the
cascade share equalled the shipped `escalated_frac` in 56 of 80 R2 rows, while `eta` matched in 80 of 80. Nothing
was evaluated.

**Cause: my script's reading of the shipped table, not the computation.** The 24 unequal rows are exactly the
positive shares, apart from the four KITTI mJ rows at 50%, and differ by at most 9.7e-17 (one unit in the last
place, relative 1.5e-15). pandas' default CSV float parser is not round-trip exact. Re-reading the same file with
`float_precision="round_trip"` gives **80 of 80 exactly equal**. Identical `eta` in all 80 rows already implied
identical k.

**Change.** The script now reads the shipped table with `float_precision="round_trip"`. The gate is unchanged:
exact equality of the share, `eta` to 1e-9. No tolerance was added. The medians the task asked for were already
right on the failed run: 0.0% / 0.0% at 20% and 0.0% / 23.9% at 50% (nuScenes / KITTI, ms).

## 2026-09-16 22:00 — Task 16 Part A results: R2 under skipping accounting tracks random

Registered run `20260916_215103_skip_accounting` (`--nboot 1000`), after the logged gate deviation. Output
`results/final/skip_accounting.csv` (80 validation, 18 share and 28 evaluation rows); report
`docs/iclr_skip_accounting.md`. No official result file was touched.

**Validation passed.** The cascade branch reproduces all 80 shipped R2 rows: share exactly equal 80/80, `eta` within
1e-9 80/80. That includes 0.0% / 0.0% at 20% ms and 0.0% / 23.9% at 50% ms (nuScenes / KITTI).

**Shares confirmed as registered.** The task's expected ms shares are right: nuScenes 0% at every budget; KITTI 0,
0, 13.69 and 83.51%. Energy gives nuScenes 9.79 / 40.83% and KITTI 11.08 / 45.52% at 30 / 50%.

**Evaluation, 28 rows with a positive skipping share.**
* **R2 beats random in 0 and loses in 2**, both at the 50% energy budget:
  * nuScenes oracle braking: nDG −0.172 against +0.168, difference −0.340 [−1.171, −0.025];
  * KITTI oracle Planner B: +0.200 against +0.453, difference −0.253 [−0.395, −0.061].
* Point nDG is below random in 18 of 28.
* Latency rows (KITTI only): 8 rows, 0 wins, 0 losses. At the 83.5% share, R2 reaches 0.31–0.70 nDG and random
  0.29–0.83.
* No bootstrap draw had a non-positive oracle; every draw was kept.

**Protocol check.** The denominator is the oracle escalating the same k in every row. The regimes: at capacity in
21 rows; share below the positive share in 6 (oracle 0.863–0.998 of capacity); **share above the non-negative
share in exactly 1**, KITTI mono braking at 50% ms, 83.5% against a non-negative share of 79.0%, oracle 0.996 of
capacity. The other KITTI cells' non-negative shares are 88.5, 94.6 and 98.8%.

**Reading, as registered.** Skipping changes how many inputs R2 escalates, not which. The ranking that did not
beat random under the cascade does not beat it here, and loses significantly in two energy rows. With 28 one-sided
tests, about 0.7 wins and 0.7 losses are expected by chance. No surprise to report.
