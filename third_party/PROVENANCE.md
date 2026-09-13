# Prior-work code used in Phase 0F

Nothing here is modified except where stated. Scoring definitions are untouched.

## planning-centric-metrics (PKL) — Philion, Kar, Fidler, CVPR 2020
- repo: https://github.com/nv-tlabs/planning-centric-metrics
- commit: `f6865f2b473303f2ff01a477bf6de4dce7109742` (2020-10-25)
- used via: `planning_centric_metrics.calculate_pkl`, unmodified
- **weights problem**: PKL's own Google Drive links for `planner.pt` and
  `masks_trainval.json` no longer resolve. Both return a Google sign-in page; `gdown`
  reports "Cannot retrieve the public link of the file". `calculate_pkl`'s auto-download
  therefore writes an HTML page and `torch.load` fails with `invalid load key, '<'`.
- **resolution**: the weights are taken from the TIP repository, which vendors them and
  states it is adapted from PKL. Verified compatible: the checkpoint loads into PKL's
  `compile_model(cin=5, cout=16, with_skip=True)` with `strict=True` — 149 keys, zero
  missing, zero unexpected — and `masks_trainval.json` is the expected (16, 256, 256)
  binary array.
- **our deviation**: `nuscenes.eval.common.loaders.create_splits_scenes` is patched so the
  "val" split contains exactly our 85 scenes. Without this, `load_gt` loads ground truth
  for all ~28k trainval samples and exhausts this board's 15 GB of unified memory
  (observed: 11.4 GB resident, then CUDA OOM). The box construction inside `load_gt` runs
  unmodified.

## TIP — Transcendental Idealism of Planner, ICML 2023
- repo: https://github.com/qcraftai/tip
- commit: `e52fd48de0624a9c54e93c8436f6bd7529b2a5d3` (2023-09-01)
- vendors `planner.pt` (75 MB) and `masks_trainval.json` (5.3 MB)

## nuscenes-devkit
- pip package `nuscenes-devkit` 1.1.11
- **environment note**: on this aarch64 board the devkit's late `import sklearn` fails with
  "cannot allocate memory in static TLS block". All scripts that touch the devkit run
  through `scripts/py`, which preloads sklearn's bundled libgomp.

## Data
- nuScenes trainval01 blob (85 scenes, 3,376 CAM_FRONT keyframes) + trainval metadata
- nuScenes map expansion v1.3

## Dependency versions actually used (conda env `edge`, Jetson AGX Xavier, L4T R35.6.5)
| package | version |
|---|---|
| torch | 2.1.0a0+41361538.nv23.6 |
| torchvision | 0.16.0+fbb4cc5 |
| numpy | 1.24.4 |
| pandas | 2.0.3 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |
| Shapely | 1.8.5.post1 |
| seaborn | 0.13.2 (TIP's `utils` imports it; installed with `--no-deps`) |
| ultralytics | 8.3.40 |

## Sign conventions
PKL and TIP are both *divergences from the planner's ground-truth output*: **higher is
worse**. TIP's `calculate_tip` states it converts its scores "to comply with PKL's
definition", so the two share the direction. The allocation signal is therefore

    G_metric(frame) = score_cheap(frame) - score_full(frame)

which is **positive when the expensive mode helps this frame**, matching the sign of the
project's own dE and dJ gains. No other transformation is applied.

## Second deviation, forced by the board
PKL/TIP are evaluated in slices of scenes (`--nchunks 6`), one process per slice, with only
the map expansions that slice needs. Holding the trainval tables, four map expansions, GT
and predictions at once left the planner 1.3 GB and the first BEV upsample raised CUDA OOM
even at `bsz=1`. Both metrics are per-sample scores computed from that sample's own boxes,
so slicing cannot change any frame's score; `bsz=1, nworkers=1` throughout.

## Third deviation: strictly one job at a time
On this board GPU allocations come out of the same physical RAM as host allocations, and the
driver needs a large contiguous free block. Twice a metric chunk died with `CUDA out of
memory` while asking for only 20-128 MiB with 1-3 GiB nominally free, because an unrelated
process (a Planner B sweep the first time, a 48 MB `git push` the second) had fragmented the
largest free block. Both chunks succeeded unchanged on a retry with the board otherwise idle.
The sweep therefore runs exactly one process at a time, retries a failed chunk, and skips
chunks whose CSV already exists. No scoring code is affected: the retried chunks produce the
same per-sample scores, since PKL and TIP are computed per sample from that sample's own
boxes.

## Phase 0G Track B — external planners
- **nuplan-devkit**: https://github.com/motional/nuplan-devkit, commit
  `e9241677997dd86bfc0bcd44817ab04fe631405b` (2025-08-27), installed with `--no-deps`.
- **tuPlan Garage** (PDM-Closed): https://github.com/autonomousvision/tuplan_garage, commit
  `b51d5d04fac1bd4389653b9ab2ff73ea88f435a3` (2024-11-28), installed with `--no-deps`.
- **Data**: `nuplan-v1.1_mini.zip` (8.0 GB) and `nuplan-maps-v1.1.zip` (927 MB) from the
  official S3 bucket. The mini **camera** blobs were not downloaded: they are nine shards of
  45-54 GB (~450 GB) against 138 GB of free disk, and they are split by blob rather than by
  log, so an arbitrary shard need not contain a single complete scenario window. What replaces
  them is recorded in the Phase 0G pre-registration.

### Environment deviation, and why it is safe
The devkit requires Python >= 3.9; this project's `edge` environment is 3.8.20 and is tied to
NVIDIA's Jetson torch build, which exists only for 3.8. A separate CPU-only `nuplan`
environment was built with Python 3.9 from conda-forge (448 packages). **No torch is
installed in it**: nuplan-devkit keeps its torch dependencies in a separate
`requirements_torch.txt`, and both planners under test are rule-based, so the simulation stack
runs without it and the Jetson torch build is untouched.

Native dependencies (GDAL through Fiona/rasterio/pyogrio, Shapely 2, rtree, pyarrow, casadi)
come from conda-forge aarch64 builds. Version pins the devkit requests that conda-forge cannot
satisfy on aarch64 were not forced; conda resolved compatible versions instead. The planners'
own code and configuration are unmodified.

### Getting the official simulation to run on aarch64 — every compatibility step
B2 required reproducing one untouched official simulation per planner before anything was
modified. It took five fixes, none of which touches the planners or their configs:

1. **Python 3.9 environment** (§ above). The assumption that Track B needs no torch was wrong:
   `run_simulation.py` imports `pytorch_lightning` at module level, before any planner is
   selected, so the entry point needs it even for rule-based planners.
2. **torch via pip, not conda.** `mamba install pytorch-cpu` solved for 59 minutes without
   finishing and was abandoned; `pip install "torch>=2.0,<2.4" pytorch-lightning timm` completed
   in four minutes from `manylinux2014_aarch64` wheels. Installed: torch 2.3.1,
   pytorch-lightning 2.6.0 (the devkit's `requirements_torch.txt` asks for 1.3.8 and torch
   1.9.0+cu111, neither of which exists for this platform — recorded as a deviation).
3. **`scripts/pynuplan`, an LD_PRELOAD wrapper.** cv2, reached through
   `nuplan.database.utils.image`, failed with `libgomp.so.1: cannot allocate memory in static
   TLS block` — the same aarch64 static-TLS defect the `py` wrapper handles for the `edge`
   environment. Two libgomp copies with *different sonames* are in play (conda's `libgomp.so.1`
   for cv2 and torch's vendored `libgomp-<hash>.so.1.0.0`), so both must be preloaded;
   preloading one leaves the other to fail.
4. **Missing pure-Python dependencies**: pytest (imported by
   `nuplan/database/utils/pointclouds/lidar.py`), tensorboard, plus ray, selenium, testbook,
   hypothesis, pyinstrument, retry, guppy3, control. `urllib3` pinned below 1.27 for botocore,
   which conflicts with selenium's requirement; selenium is only used for bokeh PNG export and
   is not on the simulation path.
5. No change to PDM-Closed, IDMPlanner, their configs, or any scoring code.

**B2 result**: `closed_loop_nonreactive_agents` on nuPlan mini, `scenario_filter=one_continuous_log`
limited to 4 scenarios. IDMPlanner 4/4 succeeded, PDM-Closed 4/4 succeeded, on the **same**
scenario tokens (`000d90717e5e569d`, `3ec7324424685846`, …), which is what the PDM-vs-IDM
comparison requires. Official nuPlan metrics were written for both.
