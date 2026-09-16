# What our two routers actually do

**What this is.** A description of the detection-list router (R1) and the pixel router (R2) as the code
implements them, written so they can be compared with the methods they are named after. Desk work only: no model
was trained, no experiment was run and no result file was touched. The only computations were read-only
inspections of shipped caches, score files and run metadata, reported where they are used below.

**References.** `file:line` refers to this repository at the commit that adds this document. The anonymous
release carries the same scripts with paths made configurable, so a line can shift by one or two there.

## Short answers

| question | answer from the code |
|---|---|
| **Q1.** Does R1 estimate a per-input reward and rank by it? | **Two of the four variants do.** `R1_mlp_reg` and `R1_gbm_reg` regress V = J(CHEAP) − J(FULL), the loss reduction FULL buys on one consumer, and inputs are ranked by that estimate. `R1_mlp_clf` and `R1_gbm_clf` estimate P(V > 0), whether escalation helps at all, and rank by that probability: not a reward estimate. On nuPlan the input is a track list, not a detector's output. |
| **Q2.** Can R2 skip the cheap pass? | **No.** The cascade runs CHEAP on every input and R2 only decides whether FULL is *added*. Nothing in the implementation or the cost accounting skips the weak pass. This is a structural difference from a weak-skipping router. |
| **Q3.** What does the estimator cost? | R1-MLP **0.649 ms** (nuScenes) and **0.654 ms** (KITTI) per input: 0.106 / 0.111 ms to build the vector plus 0.543 ms for a single-row scikit-learn call, on the Jetson AGX Xavier CPU. R1-GBM 15.9 ms. R2 9.77 / 4.82 ms, with image decoding excluded. Batch 1, Python, median over repeated calls; the basis is detailed under Q3. |

Seven places where a shipped document says less than, or something other than, the code are listed at the end.
None was changed.

---

## R1 — detection-list router

Script `scripts/103_routers_r1.py`. The nuPlan real-perception cells, which are the benchmark's official nuPlan
cells, reuse its `fit_score` from `scripts/120_nuplan_real_allocation.py:132`.

### 1. Inputs

**nuScenes and KITTI: the CHEAP detections of the current frame.**

* **Source.** The cached output of the CHEAP TensorRT YOLOv8s engine: `data/cache/nusc_det_tv/ns_cheap_320`
  (network input 192 × 320) and `data/cache/det/cheap_320` (96 × 320) (`src/rap/detect.py:45,52`). The engine
  keeps COCO classes person, bicycle, car, motorcycle, bus and truck, mapped to person / cyclist / vehicle
  (`detect.py:23-25`), with confidence floor 0.10, NMS IoU 0.65 and at most 100 detections (`detect.py:33-35`).
  `DetCache.det` applies no further filter (`src/rap/cache.py:59-65`). The lowest confidence found in either
  cache is 0.1002.
* **Selection and order.** A stable sort on descending confidence, first 25 (`103:45-46`). Ties keep the cache
  order.
* **Fields, 9 per detection** (`103:50-55`): confidence; x1/W, y1/H, x2/W, y2/H; one-hot over vehicle, person and
  cyclist (`cache.py:8`); box area / (W·H). W and H are the frame's cached image size, with fallbacks of
  1600 × 900 and 1242 × 375 (`103:65`). The KITTI cache carries four real sizes, from 1224 × 370 to 1242 × 375.
* **Padding.** Slots past the last detection stay zero (`103:48`). There is no presence flag: an empty slot is
  recognisable only by confidence 0, since real detections are at least 0.10. The 25 × 9 matrix is flattened
  row-major to 225 values (`103:56`).
* **In practice** (all cached frames): KITTI has a median of 6 detections per frame and a maximum of 36; 0.5% of
  frames are truncated at 25, 99.3% are padded and 8.0% are empty. nuScenes has a median of 4 and a maximum of 27;
  0.1% of frames are truncated, 99.9% padded and 9.7% empty.
* **Pipeline position.** After the CHEAP pass (post-NMS), before any escalation decision. Current frame only.
  The cache also stores per-detection entropy and margin, which R1 does not use.

**nuPlan: two variants, both track lists.** nuPlan detections carry no confidence, so R1 takes objects from the
planner's observation.

| variant | cells | objects | vector built at |
|---|---|---|---|
| Track B (transported detection profile) | `benchmark_table_routers.csv`, nuPlan rows | logged tracks at the decision iteration that the miss model keeps for CHEAP (`104:73-79`; `src/rap/nuplan_perception.py:64-86`) | `scripts/104_nuplan_track_lists.py:79` |
| **Real perception (official)** | `benchmark_table_nuplan_real.csv` | the CHEAP branch observation: logged tracks, minus the eligible in-image tracks the real 320 detections did not match, plus false-positive agents lifted from unmatched detections (`scripts/115_nuplan_real_counterfactual.py:79-86`) | `scripts/119_nuplan_real_features.py:110,128` |

* **Vector** (`104:35-54`). Objects are sorted by Euclidean distance from the ego rear axle, nearest first
  (`104:41`; distance at `nuplan_perception.py:43-56`); the first 25 are kept. Each slot holds 10 values:
  presence; forward offset / 80; lateral offset / 40; length / 10; width / 5; one-hot over vehicle, pedestrian,
  bicycle and static; length · width / 50. That gives 250 values. Classes map through `NUPLAN_TO_COARSE`, and any
  unlisted type becomes "static" (`src/rap/detector_model.py:43-46`; `nuplan_perception.py:53`).
* **Geometry is logged geometry.** Every kept or matched object is the logged track, so its position, box and
  class are exact, not a detection estimate. Only false-positive agents carry lifted estimates (their pose and
  box come from 114, velocity 0: `115:69-77`).
* **Most of the vector is never touched by the perception intervention.** The intervention can remove only
  vehicles, pedestrians and bicycles: inside the 60° / 80 m front cone for Track B (`nuplan_perception.py:40-41,
  58-74`), and with a projected box inside the front image for real perception (`114:55, 341-343, 359`). Static
  objects and everything else pass through to CHEAP and FULL alike. Measured on the shipped vectors, 1,440 states
  each:

  | | Track B | real perception |
  |---|---|---|
  | occupied slots behind the rear axle or static, never eligible (exact lower bound) | 73.4% | 73.1% |
  | occupied slots behind the rear axle | 46.6% | 46.4% |
  | occupied slots that are static | 44.8% | 44.7% |
  | occupied slots the intervention can act on | 7.4% (exact, same cone) | about 10% (the cone as a proxy for image projection; at most 26.9%) |
  | states with all 25 slots occupied | 86% | 87% |

### 2. Architecture and hyperparameters, as instantiated

One model per cell (consumer × target), fitted independently (`103:81-90`). Four variants (`103:72-78`):

| variant | model |
|---|---|
| `R1_mlp_reg` | `Pipeline(StandardScaler, MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=300, random_state=0))` |
| `R1_mlp_clf` | the same pipeline with `MLPClassifier` |
| `R1_gbm_reg` | `HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0)` (`src/rap/predict.py:32-35`), no scaling |
| `R1_gbm_clf` | the same settings in `HistGradientBoostingClassifier` (`predict.py:42-45`) |

Everything else is scikit-learn's default: ReLU activation, Adam at learning rate 1e-3, batch size min(200, n),
no early stopping. Some MLP fits reach the 300-iteration cap before converging: the cached-tier re-run of stage
C14, which refits these models, printed four `ConvergenceWarning` messages. scikit-learn 1.3.2 is installed in the main environment; the run metadata does not record
the version.

### 3. Training target, loss and split

* **Label.** V = J(CHEAP) − J(FULL) for the cell's consumer and target (`103:125`; `120:129`). Positive V
  means FULL lowers the loss.
* **Regressors** (`_reg`) fit V directly with squared error (the MLP adds L2 `alpha`).
* **Classifiers** (`_clf`) fit 1[V > 1e-9] (`103:84`; `EPS` at `92_benchmark_table.py:56`) with log-loss and
  score `predict_proba[:, 1]` (`103:89`).
* **Split.** Fitted on the train ∪ val units of `configs/benchmark_splits.json` (`103:126`, `120:130`), scored on
  every input, evaluated on test only. No hyperparameter was selected on any split.
* **Single-class guard** (`103:85-87`). A classifier whose training labels contain one class returns zeros for
  every input, a constant ranking that is random with full ties. In the shipped scores this fires for both
  classifiers on **nuPlan real-perception IDM safety** (0 positives among 1,056 training states), a cell the
  benchmark already declares undefined. Two other real-perception cells train their classifiers on very few
  positives: IDM scalar_J on 11 and PDM-Closed safety on 7. The Track B scores have no constant column.

### 4. Measured cost

Measured by `measure_router_overheads` (`scripts/93_budget_allocation.py:183-221`) in run
`20260914_132102_benchmark_budget_routers`, and reused unchanged by the later run `20260915_014949` (its
`reuse_overheads`). Platform: Jetson AGX Xavier, MAXN, Python 3.8.20. Values are in
`results/final/benchmark_budget_overheads_routers.json`.

| component | what is timed | nuScenes | KITTI |
|---|---|---|---|
| vector construction | `det_list_features` on 400 cached frames already in memory, 3 passes, median (`93:193-195`, `104-111`) | 0.106 ms | 0.111 ms |
| inference | single-row `Pipeline.predict` on the CPU; model fitted on synthetic labels, timing only; 1,200 calls, median (`93:196-202`) | MLP 0.543 ms, GBM 15.8 ms | same models |
| **charged per input** | vector + inference (`93:165-167`) | **MLP 0.649 ms, GBM 15.9 ms** | **MLP 0.654 ms, GBM 15.9 ms** |
| energy | CPU rail over idle during a single-row MLP loop (`93:215-221`) | +8.93 W | same figure |

* **Pre-processing versus inference.** R1 has no transfer: vector construction is the pre-processing (0.106 /
  0.111 ms), and the rest is the scikit-learn call. The detector's own decoding and NMS, and reading the cache,
  are excluded because CHEAP already pays for them.
* **Only the regressors were timed.** The classifier variants are charged the regressor's number (`93:166`).
* **nuPlan was not timed.** Its charge reuses the nuScenes vector time (`93:161-166`), so building 104's or 119's
  track vector from planner objects has never been measured.

### 5. Inputs not available before escalation

* **nuScenes and KITTI: none.** Only the current frame's CHEAP detections; no FULL output, ground truth or later
  frame.
* **nuPlan: nothing from the FULL branch or from the outcome.** However, the vector carries logged-track
  information that no camera detector produces: exact pose, size and class of every kept or matched object, and
  every logged track the intervention cannot act on, at least 73% of occupied slots. The CHEAP and FULL branches
  share this information, so it exists before escalation under the benchmark's counterfactual. It is still not
  the output of a weak detector.
* **Registry: no entry covers R1.** The leakage registry in `src/rap/features.py:50-70` holds named gate-feature
  columns, and `cheap_det` is a legal source there (`features.py:28`). But R1's 225- and 250-value vectors are
  never registered, and `assert_no_leakage` is never called on them; the only call on this path guards 120's
  gate columns (`120:60-68`). R1's freedom from leakage rests on how the vector is built, not on the registry.

---

## R2 — pixel router

Script `scripts/107_router_r2.py`, shipped run `results/raw/20260914_125344_router_r2`. nuScenes and KITTI only.

### 1. Inputs

* **Source.** The raw camera image of the same frame, not anything a detector produced: the nuScenes CAM_FRONT
  keyframe JPEG, 1600 × 900 (`107:67,70`; `src/rap/nusc.py:46-56,76`), and the KITTI `image_02` PNG, about
  1242 × 375 (`107:41,70`). It covers every frame in the CHEAP detection cache (`107:65-74`).
* **Decode and resize.** `cv2.imread` in BGR (`107:71`), then `cv2.resize` to 128 × 128 with `INTER_AREA`
  (`107:73`). **Aspect ratio is not preserved**: nuScenes frames are squashed from 16:9 and KITTI frames from
  about 3.3:1.
* **Colour order.** Channels are reversed to RGB (`107:73`) and stored as uint8, height × width × channels
  (`107:75`).
* **Tensor.** NCHW float, divided by 255, then normalised with the ImageNet mean (0.485, 0.456, 0.406) and
  standard deviation (0.229, 0.224, 0.225) in RGB order (`107:47-48,169`). In the served engine, the division
  and normalisation are part of the exported graph (`107:206-215`). The engine therefore takes a
  1 × 3 × 128 × 128 float tensor in [0, 255]. No augmentation.

### 2. Architecture and hyperparameters, as instantiated

* **Network.** `torchvision.models.mobilenet_v2(width_mult=1.2, num_classes=<number of cells>)` with random
  initialisation (`107:141`). One network per dataset: 6 outputs for nuScenes, 4 for KITTI.
* **Head.** torchvision's default classifier: `Dropout(p=0.2)`, then **one `Linear(1536 → one logit per cell)`**. The
  backbone and that single layer are shared by all the cells of a dataset. The nuScenes network has 3,198,214
  parameters.
* **Width.** The value in {1.0, …, 1.4} closest to 0.15 GFLOPs, counted by a hook on `Conv2d` and `Linear`
  multiplies (`107:79-96,139-140`). Both datasets give 1.2, at 0.1513 GFLOPs (`r2_*_meta.json`).
* **Serving** (`107:214-234`). ONNX opset 13, then TensorRT FP16 via `trtexec` with a 512 MiB workspace; the
  sigmoid is inside the engine; batch 1. Test scores come from the engine. The Spearman correlation with PyTorch
  scores is 0.9996-0.9999 per output.

### 3. Training target, loss and split

* **Target.** One output per cell, 1[V > 1e-9] (`107:137`). Where a cell has no label for a frame (its planner
  covers fewer frames), the entry is NaN and masked (`107:137,156,171`). There is no regression variant: the only
  shipped variant is `R2_cnn_clf`.
* **Loss.** `BCEWithLogitsLoss` with `pos_weight` = negatives / positives per output, counted on train ∪ val
  (`107:145-146,160`). The loss is the masked sum over labelled entries divided by their count, so all cells of
  a dataset train jointly (`107:171`). Shipped `pos_weight`: nuScenes 19.3, 3.3, 9.3, 11.0, 2.5, 7.0; KITTI 5.1,
  8.0, 3.4, 6.6.
* **Optimisation** (`107:46,157-175`). AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, 30 epochs, cosine
  annealing over all steps, seed 0, `cudnn.benchmark = False`. No early stopping, no model selection.
* **Split.** Train ∪ val frames (`107:138`): 2,421 for nuScenes and 4,472 for KITTI.
* **nuScenes weights** were trained in run `20260914_114453` and loaded through `--reuse_pt` after a board reboot
  (`107:149-152`; `weights_reused_from` in the metadata).

### 4. Measured cost

Same measurement run and platform as R1 (`93:223-258`).

| component | what is timed | nuScenes | KITTI | charged? |
|---|---|---|---|---|
| decode | `cv2.imread` of 200 full-resolution images, one pass, median (`93:237`) | 31.0 ms | 15.7 ms | **no**: treated as already paid by CHEAP |
| **pre-processing and transfer** | resize with `INTER_AREA`, channel flip, transpose, host-to-device copy, float cast, synchronise (`93:230-235,239`) | **8.22 ms** | **3.27 ms** | yes |
| **inference** | engine call with synchronise, on a zero tensor already on the GPU; 50 warm-up calls, then 300 calls, median (`93:240-248`); includes normalisation and sigmoid | **1.55 ms** | 1.55 ms | yes |
| **charged per input** | pre-processing + inference (`93:168-169`) | **9.77 ms** | **4.82 ms** | |
| energy | CPU + GPU rails over idle during a pre-processing and engine loop on nuScenes (`93:251-258`) | +1.62 W | same figure | |

The timing images are the first 200 nuScenes CAM_FRONT files in sorted order and the first 200 frames of KITTI
sequence 0000 (`93:226-227`). They are not the benchmark frames. **Inference was timed once, with the nuScenes
engine**, and that number is charged to both datasets.

### 5. Inputs not available before escalation

* **None.** The router sees the raw camera image of the current frame, which exists before either detector
  pass. Labels are used only in training, on train ∪ val.
* **Registry: no entry covers R2 either.** `cheap_image` is a legal source (`features.py:28`), but the tensor is
  never registered and `assert_no_leakage` is never applied to it.

---

## Q1. Does R1 estimate a per-input reward and rank by it?

| variant | target | ranks by | per-input reward estimate? |
|---|---|---|---|
| `R1_mlp_reg` | V = J(CHEAP) − J(FULL) | predicted V | **yes**, in decision loss for one consumer |
| `R1_gbm_reg` | V | predicted V | **yes**, same |
| `R1_mlp_clf` | 1[V > 1e-9] | P(V > 0) | **no**: probability that escalation helps at all |
| `R1_gbm_clf` | 1[V > 1e-9] | P(V > 0) | **no**, same |

How the scores are used: under a frame quota, inputs are ranked by score and the top k escalated, with exact
expectation over ties (`92_benchmark_table.py:74-138`). Under a measured budget, the escalated share becomes
(budget − CHEAP − router overhead) / FULL, and the top of the same ranking is escalated (`93:342-343`).

What the code establishes for the comparison:

* **The regressors match the mechanism** "estimate a per-input reward from the weak detector's output, then
  offload the best under a budget". Two differences are visible from our side. The reward is a downstream
  decision loss, not a detection metric, and it is learned separately for each consumer and target. And V can be
  negative, because FULL can make a decision worse; the regressors see those negative values.
* **The classifiers are structurally different.** They rank by the probability of any improvement, so they
  ignore how large the improvement is and how large a harm is.
* **On nuPlan, neither variant consumes a detector's output.** The input is a track list, mostly logged tracks
  the perception intervention cannot change (section R1.1).

## Q2. Can R2 skip the cheap pass?

**No. As implemented, R2 never skips the CHEAP pass.**

* **The cascade.** It is defined as "CHEAP runs on every frame; the allocator adds its own per-frame overhead;
  escalated frames additionally run the chosen higher fidelity" (`93:8-9`). The budget is CHEAP + f · FULL
  (`93:317`), and escalations are what remains after CHEAP and the overhead (`93:342`). The nuPlan track uses
  the same accounting (`120:208,233`).
* **R2's score only decides whether FULL is added.** Every input pays CHEAP whatever R2 predicts.
* **R2's measured cost assumes CHEAP runs.** It leaves out image decoding (31.0 / 15.7 ms) because CHEAP is
  taken to pay for it (`93:169`; `docs/iclr_routers.md` §2).
* **Its input would allow skipping.** R2 uses the raw image and nothing CHEAP produces, so it could run before
  CHEAP. Neither the pipeline nor the cost model uses that ordering. A skipping design would charge an escalated
  input router + FULL instead of CHEAP + router + FULL; CHEAP costs 12.8 ms on nuScenes and 13.2 ms on KITTI
  (`93:50-51`).

This is a structural difference between our baseline and a weak-skipping router, not a matter of tuning. The
target also differs: R2 predicts whether FULL lowers one consumer's decision loss (V > 0), not whether the
strong detector improves per-frame AP.

## Q3. Estimator cost, and whether it is like for like

| router | charged per input | of which pre-processing / transfer | of which inference |
|---|---|---|---|
| R1-MLP | 0.649 ms (nuScenes), 0.654 ms (KITTI) | 0.106 / 0.111 ms (vector construction; no transfer) | 0.543 ms |
| R1-GBM | 15.9 ms | 0.106 / 0.111 ms | 15.8 ms |
| R2 | 9.77 ms (nuScenes), 4.82 ms (KITTI) | 8.22 / 3.27 ms (resize + host-to-device copy) | 1.55 ms |
| R1 on nuPlan | not measured; the nuScenes figure is charged | — | — |

The basis, axis by axis, for comparing with a published figure such as about 0.5 ms:

* **Device.** Jetson AGX Xavier in MAXN; R1 on the ARM CPU, R2 pre-processing on the CPU and inference on the
  GPU.
* **Batching.** One input per call throughout. Batching is measured only for the 65-feature GBM gate
  (0.018 ms per input over 1,000 rows, `93:206-212`), not for R1.
* **Runtime.** Python 3.8 and scikit-learn for R1, with a `StandardScaler` step inside the timed pipeline;
  TensorRT FP16 for R2. **0.543 of R1-MLP's 0.649 ms is the interpreter and library call on a single row,** not
  the arithmetic of a 225 → 64 → 64 → 1 network. A compiled evaluator or a batched call would change this number
  far more than the model would.
* **What is included.** For R1, building the vector from detection arrays already in memory; not the detector's
  decoding or NMS, and not reading the cache. For R2, resize and transfer are included and decoding is not.
* **Statistic.** Median over repeated warm calls: 1,200 for R1, 200 and 300 for R2's two parts.
* **Coverage.** Only the regression variants were timed. nuPlan was not timed.

Whether this is like for like therefore depends on how the published figure treats four points: the device,
batch size 1, whether building the input from the weak detector's output is inclusive, and the runtime. Those
four are fixed on our side as stated.

---

## Where shipped documents say less than, or something other than, the code

Reported and left unchanged.

1. **R2 is labelled "weak skipping" but never skips the weak pass.** See `scripts/107_router_r2.py:2` and
   `docs/iclr_routers.md` §1 ("R2 (weak skipping)"). Neither says that CHEAP runs on every input.
2. **The nuPlan R1 input is described only as a nearest-object list.** `docs/iclr_routers.md` §1 says "25 nearest
   CHEAP-kept tracks" and `docs/iclr_nuplan_real_allocation.md:24` says "the 25 nearest branch objects". Neither
   says that object geometry is the logged track's, or that at least 73% of occupied slots are behind the rear
   axle or static, where the perception intervention cannot act.
3. **R2's "one head per cell" is one shared output layer.** `docs/iclr_routers.md` §1 describes separate heads;
   the code has a shared MobileNetV2 with one `Linear(1536 → one logit per cell)`, trained jointly under a
   masked, multi-task loss.
4. **R2's input is the raw camera frame, squashed.** `docs/iclr_routers.md` §1 says "CHEAP camera frame resized
   to 128×128". The image is not a CHEAP product, and the resize does not preserve aspect ratio.
5. **Only the R1 regressors were timed.** `docs/iclr_routers.md` §2 lists "R1 MLP / GBM inference, one row"
   without saying so; the classifiers are charged the same figure.
6. **R2's TensorRT cost was measured once.** `docs/iclr_routers.md` §2 lists 1.55 ms per dataset; it was timed
   with the nuScenes engine on a zero tensor and charged to both.
7. **The R1 classifiers are constant on nuPlan real-perception IDM safety** (no positive training label).
   No shipped document mentions this. The cell is already declared undefined, so no reported comparison changes.
