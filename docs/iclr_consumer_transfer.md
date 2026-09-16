# Consumer transfer: how much of an allocator's value survives a change of consumer

**What this is.** Entry (A, B) is the allocator trained for consumer A, evaluated against consumer B's decision
value on B's frozen test split, with the benchmark's exact tie expectation. The diagonal is the official held-out
result.

**Provenance.**
* Pre-registration: `RESEARCH_LOG.md`, Task 13 Part A (commit `3a656a5`), committed before the script ran.
* Code: `scripts/126_consumer_transfer.py`. Output: `results/final/consumer_transfer.csv` (845 rows).
* No official result file was changed. CPU only, cached scores.

**Signals.** The four cached R1 routers and the two gates. R1 scores come from the cached router runs, with the
identifier columns detected from each file; gates are refit with the official hyperparameters on the train ∪ val
units of the consumer they are trained for, which is the official protocol and keeps every evaluated test unit
outside the fitting set.

**Check.** The diagonal reproduces the official held-out nDG to 3 decimals in **312 of 312** comparisons.

## 1. The matrices at 20% (mean nDG over the six signals)

Rows are the consumer the allocator was trained for, columns the consumer it is evaluated against.

**KITTI oracle**

| trained for | brake | traj |
|---|---|---|
| brake | **0.265** | 0.468 |
| traj | 0.292 | **0.479** |

**KITTI mono**

| trained for | brake | traj |
|---|---|---|
| brake | **0.171** | 0.184 |
| traj | 0.174 | **0.222** |

**nuScenes oracle**

| trained for | brake | plan_ade | plan_fde |
|---|---|---|---|
| brake | **0.188** | −0.022 | 0.030 |
| plan_ade | 0.400 | **−0.068** | 0.019 |
| plan_fde | 0.167 | −0.018 | **0.036** |

**nuScenes mono**

| trained for | brake | plan_ade | plan_fde |
|---|---|---|---|
| brake | **0.231** | 0.067 | 0.047 |
| plan_ade | 0.104 | **0.107** | 0.100 |
| plan_fde | 0.155 | 0.099 | **0.038** |

**nuPlan real perception**

| trained for | IDM scalar_J | PDM-Closed safety | PDM-Closed scalar_J |
|---|---|---|---|
| IDM scalar_J | **0.181** | −0.057 | −0.055 |
| PDM-Closed safety | 0.790 | **0.367** | 0.364 |
| PDM-Closed scalar_J | 0.471 | 0.426 | **0.422** |

## 2. Transfer regret

nDG(A→B) − nDG(B→B), over the 528 off-diagonal entries of all quotas:

| subset | median | IQR | min | max | within 0.05 of the diagonal |
|---|---|---|---|---|---|
| all | −0.002 | [−0.082, +0.063] | −1.062 | +0.969 | 38.3% |
| core (KITTI, nuScenes) | −0.003 | — | — | — | 39.8% of 384 |
| nuPlan real perception | −0.002 | — | — | — | 34.0% of 144 |

**On the core tracks the consumer barely matters.** Median regret per group at 20% is −0.002 (KITTI mono), +0.009
(KITTI oracle), −0.036 (nuScenes mono) and +0.009 (nuScenes oracle). Training the allocator against a braking
controller or against a trajectory planner produces nearly the same ranking.

**On nuPlan the median hides a bimodal split.** The largest regrets at 20%:

| trained for | evaluated on | signal | nDG | diagonal | regret |
|---|---|---|---|---|---|
| IDM scalar_J | PDM-Closed safety | gate_ridge / gate_gbm | 0.000 | 0.992 | **−0.992** |
| IDM scalar_J | PDM-Closed scalar_J | gate_gbm / gate_ridge | −0.000 | 0.989 | **−0.990** |
| PDM-Closed scalar_J | IDM scalar_J | R1_mlp_clf | 0.014 | 0.938 | −0.924 |
| PDM-Closed safety | IDM scalar_J | gate_gbm | 0.940 | 0.000 | **+0.940** |
| PDM-Closed safety | IDM scalar_J | R1_gbm_clf | 0.992 | 0.059 | **+0.933** |

An allocator trained for IDM is worthless on PDM-Closed, while allocators trained for PDM-Closed are far better on
IDM than IDM's own allocator. The direction of transfer matters more than the fact of transferring.

## 3. Beating random

Entries whose paired lower bound over random is above zero, every bootstrap draw kept:

| quota | diagonal | off-diagonal |
|---|---|---|
| 10% | 10 of 78 | 14 of 132 |
| 20% | 9 of 78 | 18 of 132 |
| 30% | 11 of 78 | 18 of 132 |
| 50% | 16 of 78 | 23 of 132 |

**Off-diagonal entries beat random at least as often as the diagonal does**, in both absolute count and rate. Of
the 18 off-diagonal wins at 20%, 13 are KITTI (traj→brake and brake→traj), 2 are nuScenes, 1 is nuPlan
(PDM-Closed safety → PDM-Closed scalar_J, gate_ridge, 0.988) and none involve IDM as the training consumer.

Two consequences for the paper:
1. **The "planner-conditional value" claim is about the decision values, not about the allocators.** V differs
   between consumers, but the rankings the allocators produce do not have to.
2. **An allocator trained for the wrong consumer is often as good as the right one**, which weakens the case for
   fitting one allocator per consumer on these tracks.

## 4. Two hygiene notes

* **Frame coverage.** Cached R1 scores exist only for the frames of the consumer they were trained on. The
  nuScenes plan cells hold 2,655 of the 3,376 brake frames, so the 64 plan→brake R1 entries cover 78.7% of the
  evaluation frames; the rest are scored below the threshold and never escalate. 6 of those 64 entries beat random.
  Every other entry has full coverage.
* **The 25% prize filter again only adds wins.** It changes 66 of 840 verdicts, all in the same direction, 64 of
  them on nuPlan, where the prize sits in one log. This repeats the Task 12 finding on a different table.
