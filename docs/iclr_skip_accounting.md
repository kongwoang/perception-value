# The pixel router under skipping cost accounting

**What this is.** The benchmark charges allocators as a cascade: every input pays the allocator and CHEAP, and an
escalated input also runs FULL. A skipping router instead decides before CHEAP runs and, on an escalated input,
runs FULL *instead of* CHEAP. This adds that accounting as a second variant for the pixel router (R2), next to the
shipped cascade. It does not replace the cascade.

**Provenance.**
* Pre-registration: the pre-registration record (not part of this release), Task 16 Part A, committed before the
  script was written or run. One recorded deviation, below.
* Code: `scripts/128_skip_accounting.py`. Output: `results/final/skip_accounting.csv`.
* Nothing retrained; R2's cached scores are used as they are. No official result file changed. CPU only, 1,000
  bootstrap draws.

## 1. The two accountings

| design | mean cost per input | share f an allocator can escalate within budget B |
|---|---|---|
| cascade (shipped) | Cs + Cc + f·Cf | max((B − Cc − Cs) / Cf, 0) |
| **skipping** | Cs + (1 − f)·Cc + f·Cf = Cs + Cc + f·(Cf − Cc) | clip((B − Cc − Cs) / (Cf − Cc), 0, 1) |

Cs is the router, Cc the CHEAP pass and Cf the FULL pass.

* **Budget.** B = Cc + q·Cf at q = 10, 20, 30 and 50%, the benchmark's own definition, so both designs spend the
  same budget.
* **Decision values are the same in both designs.** Escalation replaces CHEAP's output with FULL's either way, so
  V = J(CHEAP) − J(FULL) is unchanged. Skipping changes how many inputs R2 can afford to escalate, not which ones
  it picks.
* **Constants** are those of the shipped budget track. Cc and Cf are the median end-to-end latency and GPU-rail
  energy per frame. Cs is R2's resize, host-to-device copy and engine call, and its energy is Cs × the CPU+GPU
  rails over idle. Energy therefore mixes rails, exactly as in the cascade.
* **Image decoding is charged in neither design.** The profiler decodes before its clock starts, and R2's cost
  excludes decoding. Under skipping, an escalated input still needs a decoded image for R2 and FULL.

**Eligibility.** Only R2 can skip: it reads the raw camera frame, which exists before CHEAP runs. The
detection-list routers (`R1_*`) read CHEAP's detection list, and the gates read features of CHEAP's detections.
Neither can run before CHEAP, so both are structurally ineligible and are not evaluated here.

## 2. Validation

The script's cascade branch reproduces all **80** shipped R2 budget rows of
`results/final/benchmark_budget_routers.csv`. The escalated share is **exactly equal in 80 of 80**, and the point
`eta` is **within 1e-9 in 80 of 80**. That includes 0.0% / 0.0% at the 20% ms budget and 0.0% / 23.9% at the 50% ms
budget (nuScenes / KITTI).

**Deviation.** The first smoke run stopped at this gate, with shares equal in 56 of 80. The differences were at
most 9.7e-17: pandas' default CSV float parser is not round-trip exact. Reading the same file with
`float_precision="round_trip"` makes all 80 exactly equal. The gate still requires exact equality.

## 3. Shares from the measured constants

| track | unit | Cc | Cf | Cs | cascade, 10 / 20 / 30 / 50% | **skipping**, 10 / 20 / 30 / 50% |
|---|---|---|---|---|---|---|
| nuScenes | ms | 12.710 | 19.351 | 9.766 | 0 / 0 / 0 / 0 | **0 / 0 / 0 / 0** |
| KITTI | ms | 13.179 | 18.469 | 4.817 | 0 / 0 / 3.9 / 23.9% | **0 / 0 / 13.7 / 83.5%** |
| nuScenes | mJ | 23.706 | 66.661 | 15.792 | 0 / 0 / 6.3 / 26.3% | **0 / 0 / 9.8 / 40.8%** |
| KITTI | mJ | 13.858 | 33.050 | 7.789 | 0 / 0 / 6.4 / 26.4% | **0 / 0 / 11.1 / 45.5%** |

* **nuScenes, latency.** Skipping buys nothing. Cs + Cc = 22.476 ms already exceeds even the 50% budget of
  22.386 ms, and skipping saves CHEAP only on inputs R2 escalates.
* **KITTI, latency.** At 50% the share rises from 23.9% to **83.5%**. Once CHEAP is skipped, an escalation costs
  only the 5.29 ms gap between FULL and CHEAP instead of FULL's full 18.47 ms.
* **Energy.** The constants support skipping on both datasets, so the energy budgets are evaluated too. In all,
  28 (cell, unit, budget) rows have a positive skipping share: KITTI ms, KITTI mJ and nuScenes mJ, each at 30% and
  50%.

## 4. Does R2 beat random under skipping?

**Protocol.**
* Test split of the frozen benchmark split, the benchmark's cells, and R2's cached scores; coverage 100% in every
  cell.
* k = floor(f·n): R2's gain uses the exact tie expectation; random escalates the same share.
* The denominator is the oracle escalating the same k (section 5).
* Paired cluster bootstrap over test units: 1,000 draws, **every draw kept** with no prize filter, and no draw had
  a non-positive oracle.
* R2 **beats** random when the 2.5th percentile of the paired gain difference is above 0, and **loses** when the
  97.5th percentile is below 0.

**Result: R2 beats random in 0 of 28 rows and loses to random in 2.** Its point nDG is below random's in 18 of 28.

| cell | unit | budget | skip share | nDG R2 | nDG random | R2 − random [95% CI] | gain / all-cheap loss, R2 vs random | verdict |
|---|---|---|---|---|---|---|---|---|
| nuScenes oracle braking | mJ | 30% | 9.8% | −0.123 | +0.040 | −0.163 [−0.697, +0.049] | −2.44% vs +0.80% | ties |
| nuScenes oracle braking | mJ | 50% | 40.8% | −0.172 | +0.168 | **−0.340 [−1.171, −0.025]** | −3.41% vs +3.34% | **loses** |
| nuScenes oracle planner ADE | mJ | 30% | 9.8% | +0.079 | +0.010 | +0.069 [−0.047, +0.221] | +0.20% vs +0.03% | ties |
| nuScenes oracle planner ADE | mJ | 50% | 40.8% | +0.048 | +0.035 | +0.013 [−0.130, +0.236] | +0.14% vs +0.11% | ties |
| nuScenes oracle planner FDE | mJ | 30% | 9.8% | +0.052 | +0.027 | +0.025 [−0.085, +0.160] | +0.25% vs +0.13% | ties |
| nuScenes oracle planner FDE | mJ | 50% | 40.8% | +0.107 | +0.112 | −0.005 [−0.196, +0.291] | +0.51% vs +0.53% | ties |
| nuScenes mono braking | mJ | 30% | 9.8% | +0.004 | +0.034 | −0.031 [−0.159, +0.123] | +0.09% vs +0.88% | ties |
| nuScenes mono braking | mJ | 50% | 40.8% | +0.023 | +0.144 | −0.121 [−0.490, +0.130] | +0.60% vs +3.69% | ties |
| nuScenes mono planner ADE | mJ | 30% | 9.8% | +0.087 | +0.051 | +0.037 [−0.016, +0.140] | +0.44% vs +0.25% | ties |
| nuScenes mono planner ADE | mJ | 50% | 40.8% | +0.199 | +0.186 | +0.013 [−0.149, +0.120] | +1.14% vs +1.07% | ties |
| nuScenes mono planner FDE | mJ | 30% | 9.8% | +0.109 | +0.035 | +0.074 [−0.068, +0.165] | +0.83% vs +0.27% | ties |
| nuScenes mono planner FDE | mJ | 50% | 40.8% | +0.114 | +0.142 | −0.028 [−0.311, +0.127] | +0.90% vs +1.12% | ties |
| KITTI oracle braking | ms | 30% | 13.7% | +0.040 | +0.090 | −0.050 [−0.121, +0.036] | +1.60% vs +3.62% | ties |
| KITTI oracle braking | ms | 50% | 83.5% | +0.540 | +0.548 | −0.009 [−0.163, +0.093] | +21.77% vs +22.12% | ties |
| KITTI oracle braking | mJ | 30% | 11.1% | +0.031 | +0.073 | −0.042 [−0.118, +0.030] | +1.25% vs +2.93% | ties |
| KITTI oracle braking | mJ | 50% | 45.5% | +0.219 | +0.299 | −0.080 [−0.211, +0.063] | +8.83% vs +12.06% | ties |
| KITTI oracle Planner B | ms | 30% | 13.7% | +0.064 | +0.136 | −0.072 [−0.135, +0.015] | +3.34% vs +7.07% | ties |
| KITTI oracle Planner B | ms | 50% | 83.5% | +0.695 | +0.831 | −0.136 [−0.361, +0.108] | +36.13% vs +43.22% | ties |
| KITTI oracle Planner B | mJ | 30% | 11.1% | +0.063 | +0.110 | −0.048 [−0.110, +0.036] | +3.25% vs +5.72% | ties |
| KITTI oracle Planner B | mJ | 50% | 45.5% | +0.200 | +0.453 | **−0.253 [−0.395, −0.061]** | +10.41% vs +23.56% | **loses** |
| KITTI mono braking | ms | 30% | 13.7% | +0.025 | +0.049 | −0.024 [−0.123, +0.079] | +0.70% vs +1.39% | ties |
| KITTI mono braking | ms | 50% | 83.5% | +0.313 | +0.291 | +0.022 [−0.089, +0.087] | +9.10% vs +8.47% | ties |
| KITTI mono braking | mJ | 30% | 11.1% | +0.016 | +0.042 | −0.026 [−0.108, +0.058] | +0.42% vs +1.12% | ties |
| KITTI mono braking | mJ | 50% | 45.5% | +0.197 | +0.158 | +0.039 [−0.170, +0.138] | +5.76% vs +4.62% | ties |
| KITTI mono Planner B | ms | 30% | 13.7% | +0.096 | +0.091 | +0.005 [−0.094, +0.275] | +2.35% vs +2.24% | ties |
| KITTI mono Planner B | ms | 50% | 83.5% | +0.492 | +0.556 | −0.064 [−0.215, +0.022] | +12.09% vs +13.67% | ties |
| KITTI mono Planner B | mJ | 30% | 11.1% | +0.095 | +0.074 | +0.021 [−0.089, +0.276] | +2.33% vs +1.81% | ties |
| KITTI mono Planner B | mJ | 50% | 45.5% | +0.238 | +0.303 | −0.065 [−0.186, +0.040] | +5.84% vs +7.45% | ties |

By unit: latency, 8 rows, 0 wins and 0 losses; energy, 20 rows, 0 wins and 2 losses. The interval shown is on the
nDG difference; the verdict comes from the paired gain difference, which has the same sign on every draw here.

**Reading.**
* **The result is as expected: R2 tracks random, and in two energy rows it is significantly worse.** Skipping
  lets R2 escalate far more inputs, 83.5% instead of 23.9% on KITTI at the 50% latency budget. But it cannot
  improve R2's choice of which inputs to escalate, and that choice was not better than random under the cascade.
* **High nDG at large shares measures the share, not R2.** At 83.5%, R2 reaches 0.31–0.70 nDG on KITTI, but random
  escalating the same share reaches 0.29–0.83. What counts is the paired difference, and it never excludes zero.
* **As a share of the all-cheap loss.** At the 83.5% share, R2 recovers 21.8% of it on KITTI oracle braking and
  36.1% on KITTI oracle Planner B; random recovers 22.1% and 43.2%.

## 5. Protocol check: the denominator at large shares

**Denominator used in all 28 rows:** the oracle escalating the **same k** at the realised share (the
budget-constrained oracle), not the capacity oracle Σ max(V, 0). The two coincide exactly when
#positive V ≤ k ≤ #non-negative V. They differ in two regimes:

| regime | rows | same-k oracle / capacity |
|---|---|---|
| at capacity: #positive ≤ k ≤ #non-negative | 21 | 1.000 |
| share too small to reach every positive input: k < #positive | 6 | 0.863–0.997 |
| **share above the non-negative share: k > #non-negative** | **1** | **0.996** |

**Share of non-negative decision values** in each affected cell, test split:

| cell | non-negative V | positive V |
|---|---|---|
| nuScenes oracle braking | 97.0% | 3.5% |
| nuScenes oracle planner ADE | 79.5% | 20.5% |
| nuScenes oracle planner FDE | 93.2% | 9.2% |
| nuScenes mono braking | 94.0% | 6.6% |
| nuScenes mono planner ADE | 70.1% | 26.2% |
| nuScenes mono planner FDE | 88.3% | 12.0% |
| KITTI oracle braking | 88.5% | 13.6% |
| KITTI oracle Planner B | 98.8% | 5.9% |
| KITTI mono braking | **79.0%** | 21.7% |
| KITTI mono Planner B | 94.6% | 8.1% |

**The escalated share exceeds the non-negative share in exactly one row: KITTI mono braking at the 50% latency
budget, 83.5% against 79.0%.** There, filling the budget forces some inputs where FULL is worse, so even the oracle
falls short of capacity (0.996 of it), and nDG uses that same-k oracle. At the same 50% budget the other three
KITTI cells stay within their non-negative share.

The shipped cascade normalises by the oracle at the nominal budget share q. That convention is not used here,
because the skipping share can exceed q.

## 6. Caveats

* **Cost only.** This changes the accounting, not the router. R2's scores come from the model trained under the
  cascade, and a router trained for a skipping design might choose differently. That was not tested, since
  nothing is retrained.
* **Decoding is uncharged** in both designs (section 1). A deployment that must decode once for R2 and FULL would
  add the same cost to every input in both designs, which lowers both shares.
* **Energy mixes rails**, GPU-rail energy for the detector passes and CPU+GPU over idle for R2, as the shipped
  cascade does.
* **Test units are few**, 24 nuScenes scenes and 6 KITTI sequences, so most intervals are wide. With 28 one-sided
  tests at 2.5%, about 0.7 wins and 0.7 losses are expected by chance. The 2 losses are modestly above that, and
  0 wins is consistent with no effect.
