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
