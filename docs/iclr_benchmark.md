# Benchmark v0 — frozen splits, one protocol, every baseline

> **Note (2026-09-15).** IDM numbers in this document predate the IDM route fix and are superseded. In 58% of nuPlan states the pipeline gave IDM a route it could not start from. Corrected values and the before/after comparison are in `docs/iclr_idm_route_fix.md`. PDM-Closed, nuScenes and KITTI numbers are unaffected.

Pre-registration: `RESEARCH_LOG.md`, 2026-09-13 23:55 (commit `cd67b27`, pushed before any
re-run). Splits: `configs/benchmark_splits.json`. One run produced every number below:
`scripts/92_benchmark_table.py` → `results/final/benchmark_{table,cells,self_agreement}.csv`.
All tables, at every quota, are rendered from those CSVs in `docs/benchmark_tables.md`
(`scripts/94_benchmark_markdown.py`); this page quotes them and says what they mean.

## 1. What is frozen

| track | unit | train / val / test units | test frames or states |
|---|---|---|---|
| nuScenes | scene, stratified by location | 45 / 16 / 24 | 955 (752 with trajectory truth) |
| KITTI | tracking sequence | 11 / 4 / 6 | 3,536 |
| nuPlan | **log**, stratified by map | 19 / 6 / 9 | 384 states (16 scenarios) |

Rule: sort units by (stratum, name); position mod 10 → 0–4 train, 5–6 val, 7–9 test. No RNG.
nuPlan is split by log because 3 of 34 same-log scenario pairs overlap in time by 5.2–9.1 s.
Two accepted consequences: KITTI test holds 44% of frames, and nuPlan test is Las Vegas only.

**Protocol.** η at 10/20/30/50% of test frames; a signal's gain is its exact expectation over
uniformly random tie-breaks (checked against 20,000 random orders, ≤0.5% deviation, which is
Monte-Carlo noise); bootstrap over test units, 1,000 draws, paired difference to random inside each
draw; learned gates fit on train ∪ val and scored on test once. `tie_frac` and `responsive_frac` are
in every row of the CSV.

**Cells.** nuScenes × {oracle, mono} × {brake, plan_ade, plan_fde}; KITTI × {oracle, mono} ×
{brake, traj = Planner B}; nuPlan × {PDM-Closed, IDM} × {safety, scalar_J}. PKL's planner
(`plan_*`) fails the viability test and carries that flag in `system_note`. Signals that cannot
exist in a cell are rows with `available=False` and a reason; nothing is left blank.

## 2. Harm is on the test split too — with one exception that is too small to read

Share of affected frames that escalation **hurts**, and destroyed-benefit ratio
D = Σ max(−V, 0) / Σ max(V, 0):

| cell | test: harmed / D | all units: harmed / D |
|---|---|---|
| nuScenes braking, oracle | 46.8% / 0.59 | 48.5% / 0.60 |
| nuScenes braking, mono | 47.5% / 0.65 | 45.5% / 0.54 |
| PKL planner ADE, oracle / mono | 50.0% / 0.91 · 53.3% / 0.55 | 51.4% / 0.82 · 51.6% / 0.72 |
| PKL planner FDE, oracle / mono | 42.5% / 0.73 · 49.4% / 0.65 | 50.2% / 0.88 · 50.3% / 0.83 |
| KITTI braking, oracle / mono | 45.8% / 0.34 · 49.1% / 0.65 | 34.1% / 0.09 · 40.7% / 0.19 |
| KITTI Planner B, oracle / mono | 17.2% / 0.00 · 40.2% / 0.33 | 9.5% / 0.00 · 35.2% / 0.06 |
| nuPlan PDM-Closed, safety | **0.0% / 0.00 (11 affected states)** | 37.1% / 0.30 |
| nuPlan IDM, safety | 16.7% / 0.32 (12 affected states) | 31.4% / 0.36 |

Two corrections to how D was described earlier. D is **not** uniformly large: KITTI with oracle
geometry destroys almost nothing (0.004 for Planner B, 0.09 for braking over all units), because
with true ranges the FULL detector rarely makes a decision worse. And the nuPlan test split has
so few affected states — 11 for PDM-Closed safety, none of them harmed — that no nuPlan test-split
statistic should carry a claim by itself.

## 3. Baselines on the test split, η@20

The full grid (10 signals × 14 cells × 4 quotas, plus all 9 ΔE variants in the CSV) is in
`docs/benchmark_tables.md`. What it says:

**Deployable signals that beat random** (paired 95% interval above zero):

| cell | signal | η@20 | minus random |
|---|---|---|---|
| KITTI Planner B, oracle | gate GBM | 0.58 | +0.41 [+0.18, +0.73] |
| nuPlan PDM-Closed, safety | gate ridge | 0.78 | +0.58 [+0.20, +0.80] |
| nuPlan PDM-Closed, safety | gate GBM | 0.66 | +0.43 [+0.13, +0.63] |
| nuPlan PDM-Closed, scalar_J | gate ridge | 0.76 | +0.57 [+0.20, +0.79] |
| nuPlan IDM, safety | gate GBM | 0.64 | +0.57 [+0.27, +1.15] |
| nuPlan IDM, scalar_J | gate GBM | 0.62 | +0.58 [+0.26, +1.16] |

Only learned gates, and none on nuScenes or on KITTI with monocular geometry. The nuPlan rows
rest on 11–42 affected test states; intervals reaching above +1 show how thin that is.

Across all four quotas, 21 deployable rows beat random on test. 20 are learned gates: GBM on KITTI
Planner B, oracle geometry, at every quota; GBM on KITTI braking at 30% and 50% (oracle geometry) and
at 50% (mono); and gates on nuPlan at 10–50%. The one that is not is **cheap-side criticality on
nuScenes braking with mono geometry at 30%** (0.50, +0.38 [+0.07, +0.61]), which does not separate at
10, 20 or 50%. Cheap-detection uncertainty beats random on no cell at any quota.

**The nuScenes braking/mono gate does not survive the frozen split.** Leave-one-scene-out it was
+0.360 [+0.171, +0.556] over random (§5 of `iclr_corrected_results.md`); trained on 61 scenes and
scored on the 24 test scenes it is 0.27, +0.21 [−0.15, +0.49]. The benchmark number is the
test-split one. The development result says the signal is learnable; the benchmark says 24 scenes
cannot confirm it.

**Deployable signals that are worse than random** (paired interval below zero): cheap-detection
uncertainty on PKL's planner under mono (ADE −0.14 [−0.25, −0.01], FDE −0.17 [−0.29, −0.05])
and on KITTI Planner B with oracle geometry (−0.17 [−0.20, −0.13]); cheap-side criticality on KITTI oracle braking (−0.13 [−0.20, −0.03]) and Planner B
(−0.17 [−0.20, −0.14]) and on nuPlan PDM-Closed scalar_J (−0.19 [−0.20, −0.02]). On nuPlan,
cheap-side criticality selects states where V is exactly zero (`responsive_frac` 0.000 on safety):
when CHEAP already sees a critical object, the planner is already reacting to it and FULL changes
nothing. The nuPlan gate learned a different rule. Its most important inputs (permutation
importance on train ∪ val, descriptive only) are the nearest gap in the corridor and the maximum
cheap-side criticality, and ridge puts a negative weight on tracks within 20 m of the corridor.
Affected states have a longer empty corridor (mean nearest gap 71–75 m vs 55–56 m), lower summed
cheap-side criticality (0.50–0.62 vs 0.84–0.85) and higher ego speed (5.2–5.3 vs 3.6 m/s): FULL
matters when CHEAP reports the road ahead as clear while the ego is moving.

**Diagnostics** (need FULL or ground truth). Three separate from random on test: ΔE E6 on nuScenes
oracle braking (0.67, paired +0.17 to +0.92), exact ΔE on KITTI oracle braking (0.29, paired +0.07
to +0.41), and ΔE E6 on nuPlan at 0.87–0.97 — near-tautological there, since in the simulator V is
generated by missed tracks and E6 counts criticality-weighted missed tracks. PKL and TIP do not
separate from random on any test cell; on PKL's own planner under mono PKL is 0.20
[−0.09, +0.38], and under oracle geometry its truth-referenced η on the 24 test scenes is −0.01.

## 4. Self-agreement share S = 1 − η_truth / η_self

Self = the same PKL planner's path deviation; truth = the real trajectory; identical frames.

| | PKL, ADE | TIP, ADE | PKL, FDE | TIP, FDE |
|---|---|---|---|---|
| oracle, all 85 scenes | 0.56 [0.36, 0.82] | 0.55 [0.33, 0.81] | 0.67 [0.44, 1.00] | 0.73 [0.47, 1.05] |
| mono, all 85 scenes | 0.48 [0.30, 0.69] | 0.47 [0.25, 0.70] | 0.61 [0.44, 0.81] | 0.66 [0.46, 0.89] |
| oracle, test (24) | 1.01 [0.55, 1.65] | 1.04 [0.61, 1.70] | 0.98 [0.43, 1.68] | 1.02 [0.49, 1.76] |
| mono, test (24) | 0.75 [0.45, 1.05] | 0.81 [0.44, 1.11] | 0.81 [0.62, 1.15] | 0.95 [0.60, 1.31] |

Over all scenes, **about half** of PKL's capability on its own planner is self-agreement against
ADE, and **about two thirds** against FDE. On the test split it is essentially all of it, because
the truth-referenced η is near zero there — with intervals that wide, the all-scene figure is the
one to cite, with the test figure as the caution that it varies by scene.

## 5. Allocation under measured latency and energy

`scripts/93_budget_allocation.py` → `results/final/benchmark_budget_{two_level,multifidelity}.csv`
and `benchmark_budget_overheads.json`. Cascade cost model: CHEAP on every frame, the allocator's
own overhead on every frame, the chosen higher fidelity on escalated frames. Budgets are mean cost
per frame at the level where a zero-overhead allocator escalates 10/20/30/50% of frames.

**Measured on this board** (single frame, CPU): cheap-detection uncertainty 0.24 ms, cheap-side
criticality 1.43 ms, the full 65-feature vector 3.6 ms (nuScenes) / 3.9 ms (KITTI), ridge inference
0.39 ms, **GBM inference 16.1 ms**; CPU rail 7.4 W over idle while features and GBM run. FULL costs
19.35 ms / 66.7 mJ per frame on nuScenes and 18.47 ms / 33.0 mJ on KITTI.

### 5.1 Two fidelity levels: the overhead eats the budget

With two levels a ms budget is a relabelling of a frame quota, so the only question is how many
escalations a signal's own cost removes.

| allocator | per-frame overhead | escalated at the 20% ms budget | at 50% |
|---|---|---|---|
| random | 0 | 20% | 50% |
| cheap-detection uncertainty | 0.24 ms | 18.7–18.8% | 48.7–48.8% |
| cheap-side criticality | 1.43 ms | 12.2–12.6% | 42.2–42.6% |
| gate, ridge | 4.0–4.2 ms | **0%** | 27.0–29.4% |
| gate, GBM | 19.7–20.0 ms | **0%** | **0%** |
| any diagnostic (ΔE, PKL, TIP, GT criticality) | FULL on every frame | infeasible | infeasible |

The 65-feature vector alone is 19–21% of a FULL pass, so **at a 20% budget no learned gate as
implemented can escalate a single frame**, and the GBM gate cannot at any budget up to 50%. Every
"deployable signal beats random" row in §3 was a learned gate. Under measured cost, over all cells,
both units and all four budgets, four rows still beat random (paired lower bound above zero), all at
the 50% budget: ridge on nuPlan PDM-Closed in ms (safety +0.39 [+0.08, +0.50], scalar_J +0.39
[+0.08, +0.49]), and cheap-side criticality on nuScenes braking with mono geometry, in ms
(+0.27 [+0.01, +0.63]) and in mJ (+0.32 [+0.04, +0.71]). Under a 50% frame quota that criticality
signal did not separate (+0.22 [−0.04, +0.57]); charged its 1.43 ms it escalates 42.6% of frames
instead of 50%. Its frame-quota η already falls from 0.50 at 30% to 0.38 at 50% — past about a third
of frames, further escalations it ranks next are net harmful — so being able to afford fewer of them
helps.
124 rows are *worse* than random, most of them gates that can afford no escalation at all. Under
energy budgets the CPU overhead is charged at 7.4 W and ridge becomes infeasible almost everywhere.

What this is and is not. It is the cost of *this* implementation — Python feature extraction and
scikit-learn's per-call inference path. Two follow-ups separate the parts:

* **Inference is almost entirely per-call overhead.** The same fitted GBM costs 16.05 ms on one frame
  but 0.021 ms per frame inside a 1,000-frame batch; ridge 0.42 ms vs 0.0015 ms
  (`95_gate_inference_batch_timing.py`, `benchmark_gate_inference_batch_timing.json`). A compiled
  evaluator would make inference negligible.
* **Feature extraction is the binding cost, and it does not go away.** 3.6 ms on nuScenes against a
  20% budget of 3.87 ms leaves room to escalate ~1.4% of frames even with free inference; on KITTI
  3.9 ms already exceeds the 3.69 ms budget.
* **Declared sensitivity run, `OMP_NUM_THREADS=1`** (amendment 2026-09-14 00:40, `*_1thread` files):
  single-frame GBM 12.9 ms, CPU rail 1.5 W over idle instead of 7.4 W. Latency results are
  essentially unchanged — GBM still escalates nothing up to 50%. Energy overheads fall about fivefold,
  so under mJ budgets ridge becomes feasible (10.8% of nuScenes frames escalated at 20%). Rows beating
  random under measured cost go from four to five: ridge on nuPlan PDM-Closed at 50% now also in mJ
  (paired lower bounds +0.13 for safety and +0.10 for scalar_J), and cheap-side criticality on nuScenes braking/mono keeps
  its ms row but loses the mJ one, because cheaper energy lets it escalate 46.7% of frames instead of
  34.1%, close to the 50% random does.

The finding that survives both: a gate must be cheap *relative to the escalation it buys*, and at tight
budgets that ratio — set mainly by feature extraction here — not ranking quality, decides whether it
helps. A benchmark that charges quota in frames cannot see this.

### 5.2 Several fidelity levels: measured cost changes the decision

KITTI, monocular geometry, YOLOv8s cascade 320 → {384, 512, 640}. Latency and energy disagree about
the intermediate levels: 384 costs 0.78 of 640 in ms but 0.58 in mJ.

| KITTI mono, 20% budget | braking, ms | braking, mJ | Planner B, ms | Planner B, mJ |
|---|---|---|---|---|
| oracle share at 384 / 512 / 640 | 9.2 / 6.6 / 6.9% | **11.5** / 7.2 / 7.4% | 6.4 / 2.0 / 1.2% | 6.4 / 2.0 / 1.2% |
| best 640-only allocation, as a share of the multi-level oracle | 0.742 | 0.740 | 0.776 | 0.776 |
| the plan's spend in the *other* unit, relative to that unit's budget | 0.88 of the mJ budget | **1.14 of the ms budget** | under both | under both |

* **Choosing among fidelities is worth about a third more** than choosing only whether to escalate:
  the best 640-only allocation reaches 0.74 (braking) and 0.78 (Planner B) of the multi-level optimum
  at the same budget. The greedy oracle is within 0.01% of the LP bound at 20%.
* **The unit matters when the budget binds.** For braking, the energy-optimal plan moves frames to
  384 and overruns the latency budget by 14% (4.22 ms extra per frame against 3.69 ms); the
  latency-optimal plan leaves 12% of the energy budget unspent. For Planner B the budget does not
  bind — only 9.6% of frames have any positive-gain level — so both units give the same plan.
* Deployable multi-level gates pay three model evaluations per frame: ridge escalates nothing below the
  30% budget and reaches η 0.068 (braking) and 0.126 (Planner B) at 50%; GBM escalates nothing.

## 6. What this changes for the paper

1. **The benchmark table is `benchmark_table.csv`, on the test split.** The all-unit numbers remain
   the right ones for the descriptive claims (harm, D, S) because the test split is small.
2. **Harm replicates on the frozen test split** in every cell that has enough affected frames; D
   ranges from ~0 (KITTI, oracle geometry) to 0.91, so "harmful escalations cancel X% of the benefit"
   must be stated per system, not as one range.
3. **No existing score is deployable.** The existing scores that separate from random on the test
   split are all diagnostics (ΔE E6 on nuScenes oracle braking and on nuPlan, exact ΔE on KITTI oracle
   braking). Learned cheap-side gates separate on nuPlan and on KITTI (Planner B at every quota,
   braking at 30–50%), not on nuScenes, where the only deployable signal that separates at any quota
   is cheap-side criticality on braking/mono at 30%. The nuScenes braking/mono gate's development
   result does not survive the split.
4. **Cheap-side heuristics can be worse than random** (uncertainty on PKL's planner under mono; cheap
   criticality on KITTI and nuPlan), which is a stronger motivation for the task than "they fail".
5. **Measured cost is now a result axis, with two findings**: allocator overhead decides feasibility at
   tight budgets — it erases every learned-gate win at 10–30% and reorders which allocators beat random
   at 50% — and with several fidelities the optimal allocation depends on whether the budget is
   latency or energy.

## 7. Caveats that belong in the paper

* Test splits are small: 24 nuScenes scenes, 6 KITTI sequences, 9 nuPlan logs with 11–42 affected
  states. Several intervals exceed ±1 in η units.
* nuPlan test is Las Vegas only; nuPlan V comes from a transported detector-miss model, which makes
  ΔE E6 near-tautological there and makes the nuPlan uncertainty signal privileged (flagged).
* nuPlan costs are Track B's external cost, not the official nuPlan metrics.
* PKL's planner fails the viability test; its cells are kept and flagged.
* Overheads are for this Python/scikit-learn implementation (inference overhead shown to be per-call;
  feature extraction is not); FULL/CHEAP costs are per-mode medians,
  not per-frame measurements; nuPlan uses the KITTI detector's costs and nuScenes feature timing.
