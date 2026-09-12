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
