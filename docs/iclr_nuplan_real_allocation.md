# nuPlan allocation track on real-perception decision values

> **Note (2026-09-15).** IDM numbers in this document predate the IDM route fix and are superseded. In 58% of nuPlan states the pipeline gave IDM a route it could not start from. Corrected values and the before/after comparison are in `docs/iclr_idm_route_fix.md`. PDM-Closed, nuScenes and KITTI numbers are unaffected.

**What this is.** A re-run of the benchmark's nuPlan cells with Task 5 real-perception labels (primary variant), so
the allocation track matches the real-perception result.

**Provenance.**
* Pre-registration: `RESEARCH_LOG.md`, Task 7 (commit `5c1ac60`), committed before any feature was rebuilt or any
  signal scored.
* Outputs: `results/final/benchmark_table_nuplan_real.csv` (frame quotas) and
  `results/final/benchmark_budget_nuplan_real.csv` (measured budgets).
* No existing result file was modified. The detection-profile cells stay in `benchmark_table.csv`,
  `benchmark_table_routers.csv` and `benchmark_budget_routers.csv`.

## 1. What changed and what was checked

**Labels.** V = J(CHEAP) − J(FULL) from the Task 5 primary branches, for PDM-Closed and IDM, on the safety and
scalar_J targets. Same frozen log split: fit on 25 train ∪ val logs, score once on 9 test logs.

**Pre-escalation inputs.** Everything is rebuilt from the real CHEAP branch, exactly as the planner saw it: kept
tracks plus the 320 detections' false-positive agents (`scripts/119_nuplan_real_features.py`).
* **Gate features:** 91's 18 features.
* **R1 inputs:** the 25 nearest branch objects.
* **Cheap-side criticality:** as defined in 91, but on the branch.
* **Diagnostics:** reference criticality, ΔE (missed-track count), and E_risk.

**Checks.**
* The branch object count equals Task 5's `n_tracks_cheap` on all 1,440 states.
* Recomputed reference criticality equals the existing benchmark column on all 1,440 states.
* `assert_no_leakage` passes the 18 gate features. Their provenance is cheap_det, cheap_prev and `ego_state` (a new
  legal source, declared in the pre-registration).
* `assert_no_leakage` refuses all three diagnostics and `unc_proxy`.

**Protocol.** The benchmark's own code (`scripts/120_nuplan_real_allocation.py`):
* exact tie expectation;
* 1,000 log-level bootstrap draws, paired against random;
* 92's gates and 103's R1 fitting.

**Registered undefined rule.** nDG is not reported when a split has fewer than 10 affected states or a zero oracle
prize.

**Budget track.** nuPlan's own Task 5 detector costs replace the KITTI profile. Allocator overheads are those of
the router run.

| detector cost | KITTI profile (old) | nuPlan, Task 5 (new) |
|---|---|---|
| 320 | 13.18 ms, 13.9 mJ | 14.37 ms, 44.7 mJ |
| 640 | 18.47 ms, 33.0 mJ | 23.57 ms, 105.8 mJ |

The definitions differ between the two columns:

| | old | new |
|---|---|---|
| latency | end-to-end median | preprocessing + inference + postprocessing, no JPEG decode |
| energy | GPU rail | CPU+GPU rails over idle |

## 2. How much there is to allocate

| cell | test, detection profile | test, real | all logs, detection profile | all logs, real |
|---|---|---|---|---|
| PDM-Closed · safety | 11 | **27** | 70 | **45** |
| PDM-Closed · scalar_J | 42 | **48** | 175 | **104** |
| IDM · safety | 12 | **6** → nDG undefined | 35 | **6** → nDG undefined |
| IDM · scalar_J | 26 | **12** | 96 | **25** |

Numbers are affected states (V ≠ 0), out of 384 test and 1,440 total.

* **IDM safety is undefined on both splits.** All six affected states are in the test logs.
* **IDM scalar_J's test prize comes from 12 states.**

## 3. Frame quota: test split at 20%

Each entry is η, with the paired lower bound over random in parentheses. **Bold** marks rows that beat random
(lower bound > 0).

| cell | labels | criticality (cheap) | gate ridge | gate GBM | best R1 | E_risk (diagnostic) |
|---|---|---|---|---|---|---|
| PDM-Closed · safety | detection profile | 0.00 (−0.20) | **0.78 (+0.20)** | **0.66 (+0.13)** | 0.33 (−0.20) | **0.89 (+0.56)** |
| PDM-Closed · safety | **real** | 0.07 (−0.14) | **0.99 (+0.58)** | **0.99 (+0.64)** | 0.14 (−0.13) | 0.14 (−0.14) |
| PDM-Closed · scalar_J | detection profile | 0.00 (−0.20) | **0.76 (+0.20)** | 0.55 (−0.20) | 0.33 (−0.02) | **0.87 (+0.53)** |
| PDM-Closed · scalar_J | **real** | 0.07 (−0.14) | **0.99 (+0.60)** | **0.99 (+0.75)** | 0.28 (−0.05) | 0.14 (−0.13) |
| IDM · safety | detection profile | 0.00 (−0.20) | 0.32 (−0.20) | **0.64 (+0.27)** | 0.34 (−0.98) | **0.97 (+0.74)** |
| IDM · safety | **real** | undefined | undefined | undefined | undefined | undefined |
| IDM · scalar_J | detection profile | 0.00 (−0.20) | 0.32 (−0.19) | **0.62 (+0.26)** | 0.34 (−0.18) | **0.93 (+0.67)** |
| IDM · scalar_J | **real** | 0.00 (−0.20) | 0.98 (−0.19) | 0.00 (−0.21) | ≈ 0 | **0.97 (+0.77)** |

### Deployable rows that beat random across all quotas (test split)

| signal | wins | where |
|---|---|---|
| gate ridge and gate GBM | 18 | 16 on PDM-Closed (both targets, every quota); 2 on IDM scalar_J (gate ridge at 30% and 50%) |
| R1 | 6 | MLP-clf on PDM-Closed scalar_J at 10/30/50%; MLP-reg on PDM-Closed at 50% (both targets) and on IDM scalar_J at 50% |
| cheap-side criticality | 1 | PDM-Closed safety at 50% |

### What the comparison shows

1. **At 20%, the 18-feature gate now wins 2 of 4 cells, not 4 of 4.**
   * **PDM-Closed:** the gates win both cells, and more strongly than before.
   * **IDM safety:** has no measurable prize.
   * **IDM scalar_J:** no gate wins. Gate ridge's η of 0.98 is not significant; its paired lower bound is −0.19.
2. **The miss-based diagnostic stops explaining PDM-Closed.**
   * E_risk falls from 0.89 to 0.14, and ΔE (missed-track count) wins only at 10% on safety.
   * This matches Task 5: PDM-Closed's real-perception value runs mostly through false positives.
   * A miss-only error measure cannot see that channel.
   * On IDM scalar_J, E_risk still reaches 0.97, but on only 12 affected states.
3. **Detection-list routers still lag the engineered features on this track.** R1 wins only at 30–50% quotas.

## 4. Measured budgets

| budget | allocators that beat random |
|---|---|
| **20% ms** | only the batched-inference GBM gate on PDM-Closed scalar_J: η 0.57, lower bound +0.015, escalates 5% of frames. On PDM-Closed safety the same gate reaches 0.50 (lower bound −0.004). |
| 30% and 50% ms | the gates on both PDM-Closed cells; R1-MLP-clf on PDM-Closed scalar_J (and R1-MLP-reg at 50%); on IDM scalar_J at 50%, gate ridge and R1-MLP-reg |
| 20% mJ | nothing |
| 30% and 50% mJ | the gates on PDM-Closed; gate ridge on IDM scalar_J at 50% |

**Why the 20% ms row now has a winner.** Under the detection-profile costs, nothing on nuPlan beat random at 20% ms.
With nuPlan's own, larger 640 cost (23.6 ms), a gate's 3.5–3.9 ms overhead takes a smaller share of one escalation.
The batched GBM gate can still escalate 5% of frames, and on PDM-Closed those 5% capture about half of the prize.

**Energy.** On energy, gate overheads (24–26 mJ) remain too large at 20%.

## 5. Caveats

* **The test prize is concentrated in one log.**
  * On PDM-Closed safety, 24 of the 27 affected test states come from log `2021.05.12.23.36.44_veh-35_00152_00504`,
    and 17 from one scenario, `b36903c077285e20`.
  * The gates put that scenario's states first: moving ego, 8.5 against 3.9 m/s, and many objects.
  * They hold 20 of the 27 affected states in their top 20%, and all of the positive V mass.
  * No feature dominates the ridge gate and none carries FULL or ground-truth information. Still, the η of 0.99 is
    largely one scenario ranked correctly, not a broad result.
* **About a third of bootstrap draws are dropped.**
  * PDM-Closed safety drops 330 of 1,000 draws, scalar_J 354, IDM scalar_J 368. In those draws the prize falls
    below a quarter of its full-sample value, typically because the dominant log was not resampled.
  * The intervals are therefore conditional on draws that contain that log's prize. They are narrower than the
    fragility warrants.
* **Few training labels.** Train ∪ val holds only 18 affected PDM-Closed safety states and 56 scalar_J states.
* **The budget comparison mixes cost definitions** (§1), so only within-table comparisons are exact.

## 6. What this changes in the benchmark tables

* **Replace the nuPlan cells of the allocation track with these.**
* **Gate claim, old:** "The 18-feature nuPlan gate wins all four nuPlan cells."
* **Gate claim, new:** "Under real perception the gate wins both PDM-Closed cells at every quota. IDM safety has too
  few affected states to score (6). IDM scalar_J is won only by gate ridge at 30–50%. The PDM-Closed test prize
  sits mostly in one log and one scenario."
* **Budget claim.** One PDM-Closed cell survives a 20% latency budget with nuPlan's own detector costs.
* **Diagnostic claim.** Miss-count and risk-weighted miss diagnostics no longer track PDM-Closed's value under real
  perception. This agrees with the false-positive mechanism found in Task 5.
