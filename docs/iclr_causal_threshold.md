# Causal streaming allocation with a calibrated threshold

**What this is.** The benchmark scores an allocator as a ranking that fills the budget over the whole test split.
This measures the same cached scores applied causally: one threshold, frozen before the test stream, applied in
timestamp order, with and without a running budget cap.

**Provenance.**
* Pre-registration: `RESEARCH_LOG.md`, Task 11, committed before the script ran (commit `780efbe`).
* Code: `scripts/124_causal_threshold.py`. Output: `results/final/causal_threshold.csv`.
* No official result file was changed. CPU only, cached scores.

**Policies.**

| policy | rule |
|---|---|
| official (C) | top-k over the whole test split: the hindsight reference the benchmark reports |
| A | frozen threshold: escalate iff the score passes tau_k, calibrated before the stream |
| B | A plus a causal cap: at most floor(1 + k·t) escalations after t inputs of the unit, in timestamp order |

**Calibration variants.** V1 (primary): models refit on the train units, threshold calibrated on the validation
units. V2: official models (train ∪ val), threshold from grouped 5-fold cross-fitting inside train ∪ val.

**Coverage.** `R2_cnn_clf`'s cached scores exist for test frames only, so no threshold can be calibrated for it
without re-running the image model; it appears in the official row only. `uncertainty` does not exist on the
nuPlan real-perception cells.

## 1. Checks

| check | result |
|---|---|
| refitting on train ∪ val reproduces the official scores | maximum absolute difference 1.11e-16 over 64 architecture × cell rows |
| a threshold calibrated on the test scores reproduces the official top-k nDG to 3 decimals | 444 of 444 rows |
| bootstrap | 1,000 draws, units resampled once per dataset per draw and shared by every cell, signal and policy; the official 25% prize filter applies. Dropped draws per cell: nuPlan 319–348, KITTI oracle traj 85, KITTI mono traj 29, every other cell ≤ 7 |

## 2. Pre-registered reading: **inconclusive**

Primary statistic: at the 20% rate, variant V1, the mean paired difference nDG(policy B) − nDG(official top-k) over
the 10 core cells plus the two PDM-Closed cells, averaged over the six learned deployable signals and bootstrapped
jointly.

| statistic | value | 95% CI |
|---|---|---|
| **pooled B − official (six learned signals)** | **−0.089** | **[−0.143, −0.024]** |
| gate_ridge | −0.090 | [−0.226, +0.097] |
| gate_gbm | −0.096 | [−0.195, +0.035] |
| R1_mlp_reg | −0.015 | [−0.088, +0.086] |
| R1_mlp_clf | −0.095 | [−0.176, −0.015] |
| R1_gbm_reg | −0.114 | [−0.218, −0.020] |
| R1_gbm_clf | −0.124 | [−0.207, −0.050] |

**Reproducibility of this table.** The six learned signals are refit inside the stage, and those fits are not
bit-reproducible across runs or platforms. The figures below come from six runs on the reference platform (the
shipped run, three regenerations, and two more with BLAS and OpenMP pinned to 1 and to 4 threads) and one run on a
second platform, macOS, for which only some figures were reported.

A figure is **safe** at three decimals if it rounds the same in every run and lies farther from its rounding
boundary than it has drifted in any run. It is **fragile** if it has rounded the same only because its drift pointed
away from the boundary, and **not safe** if two runs round it differently. The reading below, that the interval
straddles −0.05, holds in every run.

| signal | figure | as quoted | reference platform, 6 runs | second platform | at three decimals |
|---|---|---|---|---|---|
| **pooled, six signals** | point | −0.089 | −0.0892 to −0.0890 | same at 3 dp | safe |
| **pooled, six signals** | lower | −0.143 | −0.1427 to −0.1426 | −0.142 | **not safe** (−0.143 / −0.142) |
| **pooled, six signals** | upper | −0.024 | −0.0240 to −0.0236 | same at 3 dp | **fragile** |
| `gate_ridge` | point | −0.090 | identical | same at 3 dp | safe |
| `gate_ridge` | lower | −0.226 | identical | not reported | safe |
| `gate_ridge` | upper | +0.097 | identical | not reported | safe |
| `gate_gbm` | point | −0.096 | identical | same at 3 dp | safe |
| `gate_gbm` | lower | −0.195 | identical | not reported | safe |
| `gate_gbm` | upper | +0.035 | identical | not reported | safe |
| `R1_mlp_reg` | point | −0.015 | identical | same at 3 dp | safe |
| `R1_mlp_reg` | lower | −0.088 | identical | not reported | safe |
| `R1_mlp_reg` | upper | +0.086 | identical | not reported | safe |
| `R1_mlp_clf` | point | −0.095 | −0.0958 to −0.0949 | same at 3 dp | **not safe** (−0.096 / −0.095) |
| `R1_mlp_clf` | lower | −0.176 | −0.1767 to −0.1764 | not reported | **not safe** (−0.177 / −0.176) |
| `R1_mlp_clf` | upper | −0.015 | −0.0167 to −0.0155 | not reported | **not safe** (−0.017 / −0.016 / −0.015) |
| `R1_gbm_reg` | point | −0.114 | identical | same at 3 dp | safe |
| `R1_gbm_reg` | lower | −0.218 | identical | not reported | safe |
| `R1_gbm_reg` | upper | −0.020 | identical | not reported | safe |
| `R1_gbm_clf` | point | −0.124 | identical | −0.125 | **not safe** (−0.125 / −0.124) |
| `R1_gbm_clf` | lower | −0.207 | identical | not reported | safe |
| `R1_gbm_clf` | upper | −0.050 | identical | not reported | safe |

What to quote:
* **At three decimals:** the pooled point estimate, −0.089, and `gate_ridge`, −0.090 [−0.226, +0.097], whose fit is
  closed-form.
* **The pooled interval, at two decimals:** [−0.14, −0.02].
* **`R1_mlp_clf`'s and `R1_gbm_clf`'s point estimates, only as ranges.** They change even at two decimals: −0.09 or
  −0.10, and −0.12 or −0.13.
* **`gate_gbm`, `R1_mlp_reg`, `R1_gbm_reg`, and `R1_gbm_clf`'s bounds:** identical in every run on the reference
  platform, which is the only platform with exact values for them. Pooled figures of the same model classes have
  moved by 1.1e-3 (gradient boosting, on the second platform) and 1.3e-3 (MLP, on the reference platform). Either
  exceeds any three-decimal margin, so three decimals is not guaranteed for these on another platform.
* **The means and win counts in section 3** are not stored in `causal_threshold.csv`, so `--verify` does not cover
  them; they shift by about one cell × signal pair between runs.

The registered rule reads "streaming holds" when the lower bound is above −0.05 and "streaming costs" when the
interval lies entirely below −0.05. Here the interval straddles −0.05, so the reading is **inconclusive**: the
data rule out neither a negligible cost nor a substantial one. 382 of 1,000 draws left at least one pooled cell out
under the prize filter.

## 3. Where the loss comes from: the cap, not the threshold

At 20%, over the 12 pooled cells and the six learned signals:

| policy | mean nDG | difference to official |
|---|---|---|
| official top-k | +0.205 | — |
| **A**, frozen threshold | **+0.217** | **+0.012** |
| **B**, frozen threshold and causal cap | **+0.105** | **−0.100** |

* A beats the official ranking in 41 of 72 cell × signal pairs, B in 21.
* A frozen threshold alone costs nothing on average. The causal running cap is what removes about half of the
  achieved decision value.
* The cap binds because escalations are not spread evenly in time: a threshold that is correct on average still
  fires in bursts, and floor(1 + k·t) refuses the later frames of a burst.

**Per cell, B − official at 20%, averaged over the six learned signals:**

| cell | difference |
|---|---|
| nuPlan PDM-Closed scalar_J | −0.254 |
| nuPlan PDM-Closed safety | −0.206 |
| nuScenes oracle brake | −0.183 |
| KITTI oracle traj | −0.180 |
| nuScenes mono brake | −0.114 |
| KITTI oracle brake | −0.092 |
| nuScenes mono plan_ade | −0.070 |
| nuScenes mono plan_fde | −0.070 |
| KITTI mono brake | −0.058 |
| nuScenes oracle plan_fde | −0.055 |
| nuPlan IDM scalar_J | −0.026 |
| KITTI mono traj | +0.018 |
| nuScenes oracle plan_ade | +0.061 |

## 4. Refitting is not the explanation

At 20%, over all 14 cells and the six learned signals:

| variant | official | A | B | B − official |
|---|---|---|---|---|
| V1, refit on train, threshold on validation | +0.203 | +0.214 | +0.108 | −0.095 |
| V2, official models, cross-fitted threshold | +0.203 | +0.215 | +0.122 | −0.081 |

Keeping the official models and only cross-fitting the threshold recovers 0.014 of the 0.095. The cost is a
property of streaming, not of refitting on fewer units.

## 5. The realised rate misses the target

1,195 of 1,840 rate rows deviate from the target by more than 5 percentage points.

| policy, variant | rows | over 5 pp | median deviation | worst |
|---|---|---|---|---|
| A, V1 | 432 | 292 | 7.4 pp | 29.2 pp |
| A, V2 | 432 | 215 | 5.0 pp | 35.5 pp |
| B, V1 | 432 | 335 | 7.9 pp | 33.3 pp |
| B, V2 | 432 | 353 | 8.5 pp | 31.8 pp |
| random (Bernoulli), either variant | 56 | 0 | 0.2 pp (A), 3.0 pp (B) | 4.3 pp |

* The deviation grows with the target rate: median 5.1, 8.3, 10.2 and 10.7 pp under B at 10, 20, 30 and 50%.
* It is one-sided in both directions depending on the cell: a threshold frozen on validation scores fires far too
  often on some test units (nuScenes mono brake, R1_mlp_clf at 50%: 85.5% realised) and far too rarely on others
  (nuPlan IDM safety, gate_ridge at 50%: 16.7%).
* Only `random`, which needs no calibration, hits its target.
* This is the practical cost the benchmark's top-k protocol hides: a deployed allocator does not know the test
  distribution, and a rate promised by calibration is not the rate delivered.

## 6. Significance against random changes in both directions

Counting rows (cell × signal × rate, V1) whose paired lower bound over random is above zero:

| policy | rows beating random | rows whose verdict changes vs official |
|---|---|---|
| official top-k | 71 | — |
| A | 106 | 55 |
| B | 81 | 64 |

Streaming does not simply weaken every allocator. It promotes some (KITTI oracle brake gates at 10%, several
nuScenes R1 rows) and demotes others (nuScenes mono plan_ade R1_mlp_clf at 30 and 50%, nuScenes mono brake
R1_gbm_reg at 30 and 50%).

## 7. Measured budgets

Costs and overheads from `93_budget_allocation.py`; nuPlan uses the KITTI profile, as in the official budget table.
The target rate is the escalation share the budget still allows after the allocator's own overhead, and policy B is
run at that rate.

| budget | mean feasible rate | mean nDG of policy B | uniform full fidelity |
|---|---|---|---|
| 20% ms | 0.076 | +0.026 | infeasible in every cell: a full pass costs more than the budget |
| 50% ms | 0.259 | +0.110 | feasible in all 14 cells |

* At 20% ms both gates and both GBM routers have zero feasible rate: their per-call overhead consumes the budget,
  so they escalate nothing and gain nothing. The MLP routers and cheap-side criticality are the only allocators
  that still act.
* At 50% ms, **uniform full fidelity is the stronger baseline in 11 of 14 cells.** Comparing loss reduction as a
  share of the all-cheap loss, running FULL everywhere reaches 51.8% on KITTI oracle traj against 30.6% for the
  best allocator, and 26.5% against 16.1% on KITTI oracle brake. The allocator wins only on nuPlan IDM scalar_J
  (3.09% against 3.05%), nuScenes oracle brake (8.27% against 8.21%) and nuScenes oracle plan_fde (1.42% against
  1.30%).
* Note on denominators: the uniform-full row's own nDG is 1.0 by construction, because its reference prize is the
  oracle at a 100% rate. The comparison above therefore uses loss reduction against the all-cheap loss, which is
  comparable across rows.

## 8. The 20% table, every cell and signal (V1)

nDG against the official oracle prize at 20%. "A" is the frozen threshold, "B" adds the causal cap.

| cell | signal | official | A frozen | B streaming | B − official |
|---|---|---|---|---|---|
| KITTI mono brake | R1_gbm_clf | +0.220 | +0.307 | +0.084 | -0.136 |
| KITTI mono brake | R1_gbm_reg | +0.203 | +0.291 | +0.134 | -0.070 |
| KITTI mono brake | R1_mlp_clf | +0.168 | +0.169 | +0.099 | -0.069 |
| KITTI mono brake | R1_mlp_reg | +0.174 | +0.290 | +0.157 | -0.017 |
| KITTI mono brake | R2_cnn_clf | +0.063 | +nan | +nan | +nan |
| KITTI mono brake | criticality_cheap | +0.056 | -0.037 | +0.001 | -0.055 |
| KITTI mono brake | gate_gbm | +0.113 | +0.200 | +0.095 | -0.018 |
| KITTI mono brake | gate_ridge | +0.149 | +0.197 | +0.113 | -0.036 |
| KITTI mono brake | uncertainty | +0.060 | -0.020 | -0.008 | -0.067 |
| KITTI mono traj | R1_gbm_clf | +0.115 | +0.251 | +0.156 | +0.041 |
| KITTI mono traj | R1_gbm_reg | +0.283 | +0.315 | +0.213 | -0.070 |
| KITTI mono traj | R1_mlp_clf | +0.337 | +0.217 | +0.305 | -0.031 |
| KITTI mono traj | R1_mlp_reg | +0.213 | +0.364 | +0.293 | +0.079 |
| KITTI mono traj | R2_cnn_clf | +0.115 | +nan | +nan | +nan |
| KITTI mono traj | criticality_cheap | +0.199 | +0.344 | +0.101 | -0.099 |
| KITTI mono traj | gate_gbm | +0.259 | +0.294 | +0.208 | -0.051 |
| KITTI mono traj | gate_ridge | +0.125 | +0.314 | +0.266 | +0.140 |
| KITTI mono traj | uncertainty | +0.160 | +0.340 | +0.195 | +0.036 |
| KITTI oracle brake | R1_gbm_clf | +0.349 | +0.563 | +0.189 | -0.160 |
| KITTI oracle brake | R1_gbm_reg | +0.273 | +0.368 | +0.154 | -0.119 |
| KITTI oracle brake | R1_mlp_clf | +0.294 | +0.336 | +0.140 | -0.154 |
| KITTI oracle brake | R1_mlp_reg | +0.239 | +0.376 | +0.191 | -0.047 |
| KITTI oracle brake | R2_cnn_clf | +0.067 | +nan | +nan | +nan |
| KITTI oracle brake | criticality_cheap | -0.014 | +0.052 | -0.002 | +0.012 |
| KITTI oracle brake | gate_gbm | +0.201 | +0.366 | +0.209 | +0.007 |
| KITTI oracle brake | gate_ridge | +0.234 | +0.289 | +0.153 | -0.080 |
| KITTI oracle brake | uncertainty | +0.049 | +0.066 | +0.002 | -0.046 |
| KITTI oracle traj | R1_gbm_clf | +0.486 | +0.606 | +0.229 | -0.257 |
| KITTI oracle traj | R1_gbm_reg | +0.485 | +0.650 | +0.278 | -0.207 |
| KITTI oracle traj | R1_mlp_clf | +0.464 | +0.554 | +0.209 | -0.255 |
| KITTI oracle traj | R1_mlp_reg | +0.409 | +0.477 | +0.196 | -0.213 |
| KITTI oracle traj | R2_cnn_clf | +0.115 | +nan | +nan | +nan |
| KITTI oracle traj | criticality_cheap | +0.022 | +0.043 | +0.035 | +0.012 |
| KITTI oracle traj | gate_gbm | +0.575 | +0.798 | +0.396 | -0.179 |
| KITTI oracle traj | gate_ridge | +0.456 | +0.869 | +0.485 | +0.029 |
| KITTI oracle traj | uncertainty | +0.037 | +0.057 | +0.052 | +0.015 |
| nuPlan idm scalar_J | R1_gbm_clf | +0.059 | +0.064 | +0.000 | -0.059 |
| nuPlan idm scalar_J | R1_gbm_reg | +0.000 | +0.000 | +0.000 | +0.000 |
| nuPlan idm scalar_J | R1_mlp_clf | +0.938 | +0.911 | +0.903 | -0.034 |
| nuPlan idm scalar_J | R1_mlp_reg | +0.031 | +0.026 | +0.026 | -0.004 |
| nuPlan idm scalar_J | criticality_cheap | +0.008 | +0.004 | +0.004 | -0.004 |
| nuPlan idm scalar_J | gate_gbm | +0.000 | +0.026 | +0.000 | +0.000 |
| nuPlan idm scalar_J | gate_ridge | +0.056 | +0.056 | +0.000 | -0.056 |
| nuPlan pdm_closed safety | R1_gbm_clf | +0.069 | -0.067 | -0.067 | -0.136 |
| nuPlan pdm_closed safety | R1_gbm_reg | +0.072 | -0.001 | +0.069 | -0.003 |
| nuPlan pdm_closed safety | R1_mlp_clf | -0.067 | +0.069 | +0.069 | +0.136 |
| nuPlan pdm_closed safety | R1_mlp_reg | +0.143 | +0.210 | +0.276 | +0.133 |
| nuPlan pdm_closed safety | criticality_cheap | +0.070 | +0.000 | +0.000 | -0.070 |
| nuPlan pdm_closed safety | gate_gbm | +0.992 | +0.701 | +0.348 | -0.644 |
| nuPlan pdm_closed safety | gate_ridge | +0.992 | +0.919 | +0.274 | -0.718 |
| nuPlan pdm_closed scalar_J | R1_gbm_clf | +0.072 | -0.063 | -0.064 | -0.137 |
| nuPlan pdm_closed scalar_J | R1_gbm_reg | +0.065 | -0.018 | +0.054 | -0.010 |
| nuPlan pdm_closed scalar_J | R1_mlp_clf | +0.278 | +0.066 | +0.066 | -0.212 |
| nuPlan pdm_closed scalar_J | R1_mlp_reg | +0.141 | +0.207 | +0.270 | +0.129 |
| nuPlan pdm_closed scalar_J | criticality_cheap | +0.066 | +0.001 | +0.001 | -0.066 |
| nuPlan pdm_closed scalar_J | gate_gbm | +0.989 | +0.901 | +0.413 | -0.577 |
| nuPlan pdm_closed scalar_J | gate_ridge | +0.988 | +0.911 | +0.269 | -0.719 |
| nuScenes mono brake | R1_gbm_clf | +0.253 | +0.322 | +0.179 | -0.074 |
| nuScenes mono brake | R1_gbm_reg | +0.271 | +0.048 | -0.055 | -0.326 |
| nuScenes mono brake | R1_mlp_clf | +0.257 | +0.143 | -0.038 | -0.296 |
| nuScenes mono brake | R1_mlp_reg | +0.081 | +0.253 | +0.249 | +0.168 |
| nuScenes mono brake | R2_cnn_clf | -0.118 | +nan | +nan | +nan |
| nuScenes mono brake | criticality_cheap | +0.317 | +0.116 | +0.122 | -0.194 |
| nuScenes mono brake | gate_gbm | +0.274 | +0.136 | +0.179 | -0.096 |
| nuScenes mono brake | gate_ridge | +0.251 | +0.162 | +0.188 | -0.063 |
| nuScenes mono brake | uncertainty | -0.055 | -0.083 | -0.024 | +0.031 |
| nuScenes mono plan_ade | R1_gbm_clf | +0.121 | +0.069 | +0.034 | -0.087 |
| nuScenes mono plan_ade | R1_gbm_reg | +0.137 | -0.001 | -0.002 | -0.139 |
| nuScenes mono plan_ade | R1_mlp_clf | +0.135 | +0.144 | +0.140 | +0.005 |
| nuScenes mono plan_ade | R1_mlp_reg | +0.136 | +0.108 | +0.023 | -0.113 |
| nuScenes mono plan_ade | R2_cnn_clf | +0.140 | +nan | +nan | +nan |
| nuScenes mono plan_ade | criticality_cheap | -0.028 | -0.024 | -0.024 | +0.004 |
| nuScenes mono plan_ade | gate_gbm | +0.089 | +0.027 | +0.046 | -0.043 |
| nuScenes mono plan_ade | gate_ridge | +0.022 | +0.001 | -0.019 | -0.041 |
| nuScenes mono plan_ade | uncertainty | -0.048 | -0.035 | -0.007 | +0.041 |
| nuScenes mono plan_fde | R1_gbm_clf | +0.018 | +0.136 | +0.063 | +0.046 |
| nuScenes mono plan_fde | R1_gbm_reg | +0.034 | -0.080 | -0.062 | -0.097 |
| nuScenes mono plan_fde | R1_mlp_clf | +0.114 | -0.077 | -0.080 | -0.194 |
| nuScenes mono plan_fde | R1_mlp_reg | +0.166 | +0.022 | -0.004 | -0.169 |
| nuScenes mono plan_fde | R2_cnn_clf | +0.128 | +nan | +nan | +nan |
| nuScenes mono plan_fde | criticality_cheap | +0.039 | +0.009 | +0.010 | -0.029 |
| nuScenes mono plan_fde | gate_gbm | -0.040 | -0.104 | -0.090 | -0.051 |
| nuScenes mono plan_fde | gate_ridge | -0.064 | -0.030 | -0.019 | +0.046 |
| nuScenes mono plan_fde | uncertainty | -0.097 | -0.101 | -0.081 | +0.016 |
| nuScenes oracle brake | R1_gbm_clf | +0.355 | -0.155 | -0.173 | -0.528 |
| nuScenes oracle brake | R1_gbm_reg | +0.317 | +0.301 | -0.028 | -0.345 |
| nuScenes oracle brake | R1_mlp_clf | +0.131 | +0.279 | +0.058 | -0.073 |
| nuScenes oracle brake | R1_mlp_reg | +0.186 | +0.243 | +0.064 | -0.122 |
| nuScenes oracle brake | R2_cnn_clf | -0.124 | +nan | +nan | +nan |
| nuScenes oracle brake | criticality_cheap | +0.342 | +0.200 | +0.153 | -0.189 |
| nuScenes oracle brake | gate_gbm | -0.060 | +0.074 | -0.009 | +0.052 |
| nuScenes oracle brake | gate_ridge | +0.202 | +0.485 | +0.119 | -0.083 |
| nuScenes oracle brake | uncertainty | -0.136 | -0.201 | -0.059 | +0.077 |
| nuScenes oracle plan_ade | R1_gbm_clf | -0.064 | +0.040 | +0.033 | +0.097 |
| nuScenes oracle plan_ade | R1_gbm_reg | -0.110 | -0.047 | -0.049 | +0.061 |
| nuScenes oracle plan_ade | R1_mlp_clf | +0.046 | +0.048 | +0.109 | +0.063 |
| nuScenes oracle plan_ade | R1_mlp_reg | -0.060 | +0.007 | +0.020 | +0.080 |
| nuScenes oracle plan_ade | R2_cnn_clf | -0.020 | +nan | +nan | +nan |
| nuScenes oracle plan_ade | criticality_cheap | -0.187 | -0.043 | +0.008 | +0.194 |
| nuScenes oracle plan_ade | gate_gbm | -0.135 | -0.150 | -0.110 | +0.025 |
| nuScenes oracle plan_ade | gate_ridge | -0.084 | +0.036 | -0.047 | +0.037 |
| nuScenes oracle plan_ade | uncertainty | -0.006 | +0.009 | -0.040 | -0.034 |
| nuScenes oracle plan_fde | R1_gbm_clf | +0.202 | +0.015 | +0.003 | -0.199 |
| nuScenes oracle plan_fde | R1_gbm_reg | -0.049 | -0.102 | -0.100 | -0.050 |
| nuScenes oracle plan_fde | R1_mlp_clf | +0.182 | +0.084 | +0.076 | -0.106 |
| nuScenes oracle plan_fde | R1_mlp_reg | +0.107 | +0.028 | +0.037 | -0.070 |
| nuScenes oracle plan_fde | R2_cnn_clf | +0.114 | +nan | +nan | +nan |
| nuScenes oracle plan_fde | criticality_cheap | -0.172 | -0.039 | -0.007 | +0.166 |
| nuScenes oracle plan_fde | gate_gbm | -0.213 | -0.125 | -0.137 | +0.076 |
| nuScenes oracle plan_fde | gate_ridge | -0.013 | -0.035 | +0.004 | +0.017 |
| nuScenes oracle plan_fde | uncertainty | +0.005 | +0.024 | -0.009 | -0.013 |

## 9. Caveats

* The pooled interval straddles the registered −0.05 line, so the primary question is not settled by this run.
* The two PDM-Closed cells dominate the pooled mean and are dropped from about a third of the bootstrap draws,
  as in the earlier allocation tracks.
* `R2_cnn_clf` could not be calibrated at all from cached scores, so the pixel router is absent from every
  streaming row.
* nuPlan IDM safety keeps Task 7's undefined rule (4 affected test states); its realised rates are reported but its
  nDG is not.
* The stream order inside a nuPlan log follows the scenario start time and then the iteration. Scenario windows
  inside one log overlap by 5.2–9.1 s, so this ordering is a convention, not a physical replay.

## 10. What this changes for the paper

* **Report the benchmark's numbers as hindsight top-k, and say so.** A frozen threshold reproduces them on average
  (+0.012 at 20%), but the deployable version of the same allocator, with a causal cap, loses about half of the
  decision value in the mean (−0.100).
* **Report the realised rate, not just the target.** Two thirds of the rows miss the intended escalation rate by
  more than 5 percentage points, in both directions.
* **At a tight latency budget the ranking barely matters.** At 20% ms only the MLP routers and cheap-side
  criticality can act at all; at 50% ms, running FULL on every frame beats the best allocator in 11 of 14 cells.
