# IDM route fix: what was wrong, what was rerun, what changed

**Bug.** Every nuPlan branch initialised the devkit's `IDMPlanner` with the scenario route.
* IDM looks for the ego's lane only in the **first two** route roadblocks.
* In 836 of the 1,440 benchmark states (58%), the ego is beyond those two.
* In 193 of them, the ego is on no route roadblock at all (pickup/drop-off and stationary scenarios).
* In those 836 states IDM's reference plans deviated from the log by 30.2 m on average (up to 413 m) and collided in
  20.6% of them. Over all 1,440 states, the reference collision rate was 13.8%.

PDM-Closed was never affected; it corrects its own route. Perception, matching and scoring were not involved.
The diagnosis is in `results/raw/idm_route_diagnostics/`.

**Fix.** For IDM only, at every state:
1. correct the route with tuPlan Garage's `route_roadblock_correction`, the function PDM-Closed applies to itself;
2. trim it to start at the ego's roadblock.

IDM's policy and parameters are unchanged. The fix lives in `route_for()` in `scripts/82_nuplan_counterfactual.py`
and is also used by 115. Passing `--no_route_fix` reproduces the old behaviour.

**Audit for similar errors.**

| component | check | result |
|---|---|---|
| planner parameters | against the released configs | identical |
| PDM-Closed route | handling at mid-scenario and off-route states | corrected internally |
| PKL's planner (nuScenes) | pixel→metre decoding and trajectory frame, tested on validation rasters | the used convention is correct (lowest ADE; alternatives 2.6–18.4 m). Its viability failure is genuine |
| scoring | horizon alignment, history buffer, traffic lights | consistent |

**Rerun.** Everything that depends on IDM, in two detached chains, one job at a time.
* Main chain (00:52–03:39): Track B IDM, 83, 85, 92, 94, 103, 93 `--routers` (measured overheads reused), 110,
  Task 5 IDM reference and branches, 116, 117, 120, 118.
* Supplementary chain (03:50–04:01): 93 primary and 93 `--suffix _1thread` (archived overheads reused), then 94.
  The main chain had missed these two budget runs; the omission is logged as an amendment (commit `2e2ed7c`).
Pre-fix outputs are archived in `results/archive/pre_idm_route_fix/`. The machine-readable comparison is in
`results/final/idm_route_fix_comparison.json`.

## Checks (all passed)

| check | result |
|---|---|
| IDM reference plans after the fix (1,440 states) | median log deviation **1.46 m** (was 2.44 m); **0%** of states above 20 m (was 13.3%); collisions **4.2%** (was 13.8%). PDM-Closed: 1.81 m, 0%, 4.5% |
| perception filter untouched | CHEAP, FULL and reference track counts equal the archive on all states |
| Task 5 reference reproduces corrected Track B; identity reproduces reference | 0 mismatches; 1,440 identity observations equal; 120 planned states equal |
| rows that must not move | all non-IDM rows identical to the archive in 13 files: benchmark table and cells, router table, four budget tables (routers, nuPlan real, primary two-level, 1-thread two-level), both multi-fidelity tables, Task 5 cells, Task 7 table, both PDM-Closed raw files |

## Before and after, IDM only

### Track B: transported detection profile, all 1,440 states

| aggregate | affected (V+ / V−) | harm rate | all-FULL reduction | oracle@20 reduction | selective extra share |
|---|---|---|---|---|---|
| safety, before | 35 (24 / 11) | 31.4% | 3.9% | 6.1% | 36.4% |
| safety, **after** | **47 (34 / 13)** | **27.7%** | **16.8%** | **21.6%** | **22.0%** |
| scalar_J, before | 96 (57 / 39) | 40.6% | 2.0% | 3.2% | 37.1% |
| scalar_J, **after** | **105 (70 / 35)** | **33.3%** | **14.0%** | **18.1%** | **22.6%** |

A functioning IDM gains much more from FULL under the transported profile.

**Registered falsifiers.**
* B-F1, B-F2 and B-F3 fire neither before nor after.

**Planner comparison, same objective** (PDM-Closed vs IDM):

| aggregate | top-20 overlap | strict-pair disagreement | gamma |
|---|---|---|---|
| safety | 0.22 → **0.24** | 0.11 → **0.16** | 0.78 → **0.69** |
| scalar_J | 0.28 → **0.32** | 0.33 → **0.26** | 0.35 → **0.49** |

**2×2** (`phase0g_external_2x2.csv`): the IDM-vs-PDM safety gamma drops from 0.78 to 0.69; the progress-deviation
gamma rises from 0.34 to 0.50.

### Benchmark table, nuPlan IDM cells (detection profile, test split, 20%)

| target | IDM test cell | gate GBM η (lower bound vs random) | gate ridge | E_risk (diagnostic) |
|---|---|---|---|---|
| safety, before | 12 affected, all-FULL 2.5% | **0.64 (+0.27)** | 0.32 (−0.20) | 0.97 (+0.74) |
| safety, **after** | 14 affected, all-FULL 21.7% | 0.49 (−0.20) | 0.50 (−0.20) | **0.87 (+0.50)** |
| scalar_J, before | 26 affected | **0.62 (+0.26)** | 0.32 (−0.19) | 0.93 (+0.67) |
| scalar_J, **after** | 28 affected | 0.62 (−0.05) | 0.50 (−0.20) | **0.85 (+0.49)** |

**The earlier claim that the gate wins all four nuPlan cells does not survive.**
* After the fix, the gate wins the two PDM-Closed cells but neither IDM cell at 20%.
* The PDM-Closed rows are unchanged.
* R1 routers win neither IDM cell at 20%, before or after.

### Measured-cost budgets, nuPlan IDM cells (detection profile, test split)

| run | deployable allocators that beat random, any budget level, ms or mJ | random η at 20% ms, before → after |
|---|---|---|
| primary two-level (`benchmark_budget_two_level.csv`) | none before, none after | 0.134 → 0.198 |
| 1-thread sensitivity (`benchmark_budget_two_level_1thread.csv`) | none before, none after | 0.134 → 0.198 |
| routers (`benchmark_budget_routers.csv`), 20% ms | none before, none after | — |

The budget conclusion for IDM does not change: only the η values move.

### Task 5: real YOLOv8s perception, IDM

| variant · split · aggregate | before: affected, harm, ρ | **after: affected, harm, ρ** |
|---|---|---|
| primary · all · safety | 6, 16.7%, 0.013 | **5, 40.0%, 0.055** |
| primary · all · scalar_J | 25, 48.0%, 0.062 | **30, 33.3%, 0.136** |
| primary · test · safety | 6, 16.7%, 0.013 | **4, 25.0%, 0.044** |
| primary · test · scalar_J | 12, 50.0%, 0.043 | **10, 10.0%, 0.076** |
| nofp · all · safety | 3, 33.3%, 0.01 | **2, 0%, 0** |
| s1 · all · safety | 5, 40.0%, 112 | **7, 71.4%, 3.1** |
| iou50 · all · scalar_J | 28, 46.4%, 0.06 | **35, 40.0%, 0.155** |

* **Pre-registered reading (primary, all): intermediate, before and after.**
  * PDM-Closed is unchanged and clears both bars.
  * IDM's ρ stays below 0.20 on both aggregates (0.055 and 0.136).
* **Readings on other variants** (not deciding): unchanged, except `nofp · test`, which now reads "falsifier fires".
  There, IDM has one affected safety state and PDM-Closed none, and neither has a harmful one.
* **With a functioning IDM, the Task 5 result stands for the right reason.** Under real perception, IDM's decision
  value is small and sparse: 5 affected safety states and 1.4% all-FULL reduction. Before the fix, those numbers
  came from IDM plans that had broken down.

### Task 7: allocation on real-perception decision values, IDM

| cell | before | **after** |
|---|---|---|
| safety, test | 6 affected → undefined | **4 affected → undefined** |
| scalar_J, test | 12 affected. No deployable signal beats random at 20%. Gate ridge wins at 30% and 50%; R1-MLP-reg at 50%. E_risk (diagnostic) 0.97 at 20% | **10 affected. R1-MLP-clf beats random at every quota (η 0.94, +0.64 at 20%) and at every ms and mJ budget level; R1-GBM-clf also wins at 50%. Gate ridge wins nowhere. E_risk 0.06 at 20%** |

The R1 result rests on 10 affected test states and should be read as fragile.

## What this changes for the paper

1. **Report the corrected IDM numbers.** The pre-fix IDM rows measured a planner that planned along wrong paths in
   58% of states.
2. **Transported profile.**
   * IDM now responds strongly: oracle@20 is 22% on safety.
   * The two published planners still disagree about which states matter (overlap 0.24, B-F3 does not fire).
   * Planner-conditionality survives the correction.
3. **Real perception.**
   * The conclusion is unchanged, and now clean: PDM-Closed shows benchmark-level sign variation, while IDM is
     nearly insensitive to the 320/640 choice.
   * The reading stays intermediate.
4. **Allocation claims.**
   * On the detection-profile benchmark, the gate wins only the PDM-Closed nuPlan cells.
   * Under measured cost, no allocator beats random on IDM, before or after.
   * Under real perception, IDM scalar_J is won by R1-MLP-clf on a handful of states.

## Not rerun, and why

* `phase0g_external_idm_sanity_raw.csv`: the 6-state wiring probe predates the fix and is superseded by the
  identity checks.
* Seven hand-written reports that quote pre-fix IDM numbers carry a pointer to this page. Generated tables (`docs/*_tables.md`)
  were regenerated.
* The anonymous code release has **not** been updated.
