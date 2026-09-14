# Lightweight routers on the benchmark and budget tracks

> **Note (2026-09-15).** IDM numbers in this document predate the IDM route fix and are superseded. In 58% of nuPlan states the pipeline gave IDM a route it could not start from. Corrected values and the before/after comparison are in `docs/iclr_idm_route_fix.md`. PDM-Closed, nuScenes and KITTI numbers are unaffected.

Pre-registration: `RESEARCH_LOG.md`, 2026-09-14 "Task 2 pre-registration", committed before any run.
Four operational incidents are logged there, none affecting design or results: an R1 file-name crash,
two board reboots, and the R2 restaging. Every table at every quota: `docs/routers_tables.md`, generated
from `results/final/benchmark_table_routers.csv` and `benchmark_budget_routers.csv` by
`scripts/110_routers_markdown.py`.

## 1. What was tested

The splits, cells and statistics are the benchmark's own:
* test split only;
* fit on train ∪ val;
* exact tie expectation;
* unit bootstrap, paired against random.

No router hyperparameter was changed after a test score was seen.

| router | input | model | score |
|---|---|---|---|
| **R1** (ORIC-style) | top 25 CHEAP detections by confidence (conf ≥ 0.10): conf, normalised box, class one-hot, area, 225 dims. nuPlan: 25 nearest CHEAP-kept tracks, 250 dims | MLP 2×64 or the benchmark GBM | V (regression) or P(V>0) (classification) |
| **R2** (weak skipping) | CHEAP camera frame resized to 128×128 | MobileNetV2 ×1.2 = **0.151 GFLOPs**, trained from scratch, one head per cell, served as a TensorRT FP16 engine | P(V>0) |

R2 check: TensorRT and PyTorch scores agree to Spearman 0.9996–0.9999 per head.

R2 training data and fit:
* nuScenes: 2,421 train ∪ val frames, 6 heads, loss 1.20 → 0.015;
* KITTI: 4,472 frames, 4 heads, loss 0.0028 at epoch 30 — close to memorised.

## 2. Measured cost on this board

Single-frame medians. CHEAP's decode is shared and not charged to R2.

| component | nuScenes | KITTI |
|---|---|---|
| FULL pass (for scale) | 19.35 ms | 18.47 ms |
| 65-feature gate features | 3.52 ms | 3.52 ms |
| ridge / GBM inference, one row | 0.41 / 17.9 ms | same |
| GBM inference, amortised in a 1,000-row batch | 0.018 ms | same |
| **R1 features** | **0.106 ms** | **0.111 ms** |
| R1 MLP / GBM inference, one row | 0.54 / 15.8 ms | same |
| R2 resize + upload | 8.22 ms | 3.27 ms |
| R2 TensorRT inference | 1.55 ms | 1.55 ms |
| (image decode, already paid by CHEAP) | 31.0 ms | 15.7 ms |

Totals per frame:
* **R1-MLP: 0.65 ms**, 3.4–3.5% of a FULL pass. The first learned allocator here that is cheap
  relative to the escalation it buys.
* The gate with batched GBM: 3.54 ms, ~18–19% of FULL. Its features, not its model, are the cost.
* R2: 9.8 ms on nuScenes (50% of FULL) and 4.8 ms on KITTI (26%). The network is cheap (1.55 ms); the
  resize of a full-resolution frame and the host-to-device copy are not.

Rail power over idle while running:
* R1 loop: CPU rail 8.9 W;
* R2 loop: CPU + GPU rails 1.6 W.

## 3. Under a frame quota (no cost charged)

A signal beats random when its paired 95% lower bound is above zero; it loses when the upper bound is
below zero.

| signal | tests | beats random | loses to random | KITTI wins | nuScenes wins | nuPlan wins |
|---|---|---|---|---|---|---|
| R1 MLP, V | 56 | 7 | 0 | 6 / 16 | 1 / 24 | 0 / 16 |
| R1 MLP, P(V>0) | 56 | 9 | 0 | 7 / 16 | 2 / 24 | 0 / 16 |
| R1 GBM, V | 56 | 11 | 1 | 8 / 16 | 3 / 24 | 0 / 16 |
| R1 GBM, P(V>0) | 56 | 8 | 0 | 8 / 16 | 0 / 24 | 0 / 16 |
| **R2 CNN** | 40 | **0** | **4** | 0 / 16 | 0 / 24 | — |
| gate GBM (65 features) | 56 | 14 | 0 | 7 / 16 | 0 / 24 | 7 / 16 |
| gate ridge | 56 | 6 | 1 | 0 / 16 | 0 / 24 | 6 / 16 |
| cheap-detection uncertainty | 56 | 0 | 7 | 0 | 0 | 0 |
| cheap-side criticality | 56 | 1 | 10 | 0 | 1 | 0 |

With 56 one-sided tests at 2.5%, about 1.4 wins per signal are expected by chance. R1's 7–11 and the
gate's 14 are well above that; R2's 0 is not.

### What the table says

* **Detection-list routers learn something real, and mostly on KITTI.**
  * At 20%, a router beats random in 3 cells, all on KITTI: oracle Planner B, oracle braking and mono
    braking.
  * The 65-feature gates beat random in 5 cells: KITTI oracle Planner B and all four nuPlan cells.
  * On KITTI oracle Planner B, R1 reaches η@20 0.46–0.49; the gate reaches 0.58.
* **nuScenes stays unsolved.**
  * No router and no gate beats random at 20%.
  * R1 separates only at 30–50% on braking: GBM on V, mono, +0.34 [+0.07, +0.62] over random at 50%.
  * R1 also separates in two single planner cells: FDE at 10% and ADE at 50%.
* **On nuPlan the raw track list loses to the engineered features.** R1 wins nothing there. The
  18-feature nuPlan gate wins in all four cells. The gate's corridor gap and cheap criticality carry
  information that a distance-sorted list of 25 tracks does not expose to these models.
* **Raw pixels at 0.15 GFLOPs do not work here.**
  * R2 never beats random.
  * It is significantly worse than random on nuScenes oracle braking (20%, 30%, 50%) and on KITTI
    oracle Planner B (50%).
  * On KITTI it memorised its training frames (final loss 0.003) without generalising.

## 4. Under measured latency and energy budgets

Cascade cost model and budgets as in `93_budget_allocation.py`. Each allocator is charged its own
measured overhead. The batched-GBM gate is the same ranking as the GBM gate, charged batched inference;
it is evaluated, not computed analytically. Full tables: `docs/routers_tables.md`.

**Share of frames an allocator can still escalate** (median over cells; random escalates 20% and 50%):

| allocator | overhead | 20% ms budget, nuScenes / KITTI | 50% ms budget | 20% mJ budget |
|---|---|---|---|---|
| cheap-detection uncertainty | 0.21–0.29 ms | 18.5% / 18.8% | 48.5% / 48.8% | 17.0% / 15.7% |
| cheap-side criticality | 1.30–1.47 ms | 12.4% / 12.9% | 42.4% / 42.9% | 5.2% / 0% |
| **R1 MLP** | **0.65 ms** | **16.6% / 16.5%** | **46.6% / 46.5%** | 11.3% / 2.3% |
| gate, GBM batched | 3.54 ms | 1.7% / 0.8% | 31.7% / 30.8% | 0% / 0% |
| gate, ridge | 3.93 ms | 0% / 0% | 29.7% / 28.7% | 0% / 0% |
| gate GBM / R1 GBM, one row per call | 21.4 / 15.9 ms | 0% | 0% | 0% |
| R2 CNN | 9.8 / 4.8 ms | 0% / 0% | 0% / 23.9% | 0% / 0% |

**Allocators that still beat random under measured cost** (paired lower bound above zero):

| budget | allocator | cell | η | lower bound over random |
|---|---|---|---|---|
| **20% ms** | **R1 MLP, P(V>0)** | **KITTI oracle · Planner B** | **0.41** | **+0.11** |
| 10% ms | R1 MLP, P(V>0) | nuScenes oracle · planner FDE | 0.21 | +0.004 |
| 10% ms | R1 MLP, V | nuScenes mono · planner FDE | 0.16 | +0.011 |
| 30% ms | R1 MLP, P(V>0) | KITTI oracle · Planner B | 0.59 | +0.10 |
| 30% ms | gate GBM batched | nuPlan IDM · safety, scalar_J | 0.64, 0.62 | +0.006, +0.12 |
| 50% ms | R1 MLP (both targets) | KITTI oracle · Planner B; KITTI mono · braking | 0.76–0.84; 0.32–0.37 | +0.01 to +0.10 |
| 50% ms | R1 MLP, P(V>0) | nuPlan IDM · scalar_J | 0.97 | +0.017 |
| 50% ms | gate GBM batched | KITTI oracle · Planner B | 0.82 | +0.058 |
| 50% ms | gate ridge | nuPlan PDM-Closed · safety, scalar_J | 0.89, 0.88 | +0.095, +0.077 |
| 50% ms, 50% mJ | cheap-side criticality | nuScenes mono · braking | 0.38, 0.48 | +0.001, +0.050 |

**Pre-registered reading.** At the 20% ms budget, a router beats random in **1** cell and a 65-feature
gate in **0**. Under a 20% frame quota the counts were 3 and 5.

Under energy budgets no learned allocator beats random at any level. The single-row MLP call drew 8.9 W
over idle on the CPU rail, which makes R1 cost 5.8 mJ per frame, 18% of a KITTI FULL pass.

## 5. What this changes for the paper

1. **Charging measured cost reorders the allocators.**
   * The strongest frame-quota allocator, the 65-feature GBM gate with 14 wins, cannot escalate anything
     when charged its single-row inference. Charged batched inference, it wins 3 rows, all at 30–50% budgets.
   * R1-MLP is the only learned allocator that beats random at a 20% latency budget. Its inputs — a list
     of 25 boxes — cost 0.11 ms, against 3.5 ms for the engineered features.
   * This is the "overhead decides feasibility" finding again, now with an allocator that clears it.
2. **Cheap inputs win once cost is counted; pixels lose either way.**
   * R2, at the requested 0.15 GFLOPs, never beats random under a quota and is worse than random in 4 rows.
   * Its preprocessing alone — resizing a full-resolution frame and copying it to the GPU — costs 26–50%
     of a FULL pass.
3. **nuScenes remains open.** No deployable allocator — engineered, detection-list or pixel — beats random
   at 20% on nuScenes, under a quota or a budget. That is the benchmark's hardest cell, not a failure of one
   method.
4. **On nuPlan structure matters.** The 18-feature gate (corridor gap, cheap criticality) wins all four
   cells under a quota, and a distance-sorted track list wins none. Any router claim should be per track.

## 6. Caveats

* **Multiple tests.** 56 tests per signal and table, so about 1.4 wins are expected by chance at the
  one-sided 2.5% level. R1's 7–11 quota wins and the MLP's 9 latency-budget wins (6 for P(V>0), 3 for V)
  are above that, but modest. Several lower bounds sit within 0.01 of zero.
* **Implementation-dependent overheads.** They are for this Python, scikit-learn and TensorRT code. The
  scikit-learn MLP's multithreaded single-row call inflates its CPU power, and a compiled evaluator would
  change the energy rows.
* **R2 was trained exactly as pre-registered**: no augmentation, fixed 30 epochs. On KITTI it memorised its
  training set. A regularised pixel router might do better, but was not the registered test.
* **nuPlan R1 feature time** is proxied by the nuScenes R1 feature time (tracks were not timed); the R1 GBM
  rows cannot escalate regardless.
* **Board incidents.**
  * The board rebooted twice during the Task 2 chain. The telemetry showed no memory or heat cause at
    either reboot; see RESEARCH_LOG 12:53 and 13:20. Every result here comes from a run that completed.
  * The fan was switched to 100% at 13:21, during the budget run's overhead measurement. Latency is
    unaffected. The fan runs off SYS5V, not the CPU or GPU rails used for energy.
