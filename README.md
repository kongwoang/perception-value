# risk-aware-perception

**Research question.** Does downstream task criticality provide information about the
marginal value of additional visual perception compute that visual uncertainty and
scene complexity do not?

**Target venue.** CVPR 2027. **Current stage:** Phase 0 — characterization only. No
learned routing method is built until Phase 0 passes its GO criteria.

## The Phase-0 claim under test

One detector, two fidelities (same weights, same architecture, different input
resolution). For a frame `t`:

```
Risk(t, m)      = sum_i criticality(i, t) * perception_error(i, t, m)
Value_task(t)   = Risk(t, CHEAP) - Risk(t, FULL)
Value_visual(t) = Error_std(t, CHEAP) - Error_std(t, FULL)      # every criticality = 1
```

`Value_task` is an **oracle** built from GT 3D geometry. The predictors that must
estimate it see only what a deployed gate would: the CHEAP detections of this frame,
the CHEAP detections and downscaled image of the previous frame, and the camera
calibration. `rap.features.assert_no_leakage` enforces that against a provenance
registry rather than trusting the caller.

## Layout

```
src/rap/
  kitti.py      KITTI tracking loader; rectified-camera -> IMU/ego transform
  geometry.py   GT ego-frame geometry, range rate, TTC, criticality models
  mono.py       the same quantities ESTIMATED from cheap 2D boxes (deployable)
  detect.py     two-fidelity YOLOv8s, keeps the full per-box class distribution
  trt.py        TensorRT runner (PyTorch eager has a ~19 ms overhead floor on Xavier)
  risk.py       matching, per-object error, frame risk
  features.py   deployable features + provenance registry + leakage guard
  tables.py     per-frame table: oracle targets + features
  predict.py    leave-one-sequence-out predictability
  budget.py     budgeted selection at matched FULL-compute quota
  pairs.py      uncertainty-matched pairs (the core novelty test)
  viz.py        figures
scripts/
  00_build_engines.py   ONNX export + FP16 TensorRT engines, one per mode
  01_profile_jetson.py  latency / peak memory / rail power on this board
  02_run_detection.py   both fidelities over every frame -> detection cache
  03_build_tables.py    per-frame analysis table
  04_analysis.py        heterogeneity, predictability, budget, matched pairs
  05_figures.py         all figures
  06_sensitivity.py     every result re-run under alternative risk definitions
  07_exemplars.py       qualitative matched-pair figure
  99_synthetic_pipeline_check.py   negative control (see below)
```

## The negative control

`99_synthetic_pipeline_check.py` replays the entire pipeline on real GT labels with
*simulated* detections whose recall depends only on apparent box size — so there is
no true criticality mechanism to find. It is run as a regression check because two of
the analyses behave very differently on it:

* criticality features still predict `Value_task` (rho ~ 0.3), because `Value_task`'s
  *scale* is set by how much criticality is present in a frame at all;
* the uncertainty-matched pair test correctly returns null.

That is why the matched-pair test, not the raw predictability delta, is treated as the
decisive evidence for the novelty claim.

## Reproducing

```bash
P=/home/kongwoang/miniforge3/envs/edge/bin/python
$P scripts/00_build_engines.py
$P scripts/01_profile_jetson.py --backend trt --images <dir of real frames>
$P scripts/02_run_detection.py --backend trt
$P scripts/03_build_tables.py
$P scripts/04_analysis.py --lat_cheap <ms> --lat_full <ms>
$P scripts/05_figures.py && $P scripts/07_exemplars.py
$P scripts/06_sensitivity.py
$P -m pytest tests/ -q
```

Every run directory under `results/raw/` stores its resolved config, git commit and
environment, so any number traces back to the code that produced it.

Findings: `docs/cvpr_phase0_findings.md`. Chronology: `RESEARCH_LOG.md`.
