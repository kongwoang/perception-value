# Does a perception-centric evaluation objective pick a different allocator?

**What this is.** The benchmark ranks allocation signals by the decision value they realise at a budget. A
perception-first author would rank the same signals by the perception gain they realise instead. This holds the
method pool, the selection protocol, the exact tie expectation and the cluster bootstrap fixed, and changes only
the *evaluation objective*, to ask which method each objective selects.

This is not Task 9. Task 9 asked which **training target** produces a better model and returned a pre-registered
null; that null stands. Here nothing is retrained: the same cached scores are scored against a different ruler.

**Provenance.**
* Pre-registration: the pre-registration record (not part of this release), Task 13 Part D, committed before the
  script ran.
* Code: `scripts/127_objective_swap.py`. Output: `results/final/objective_swap.csv`.
* No official result file was changed. CPU only, cached scores, 1,000 bootstrap draws.

**Objectives.**

| name | per-input value | normaliser |
|---|---|---|
| `E_dec` | decision value V = J(CHEAP) − J(FULL) | decision-value oracle prize at the same budget |
| `E_perc_dE` | `dE_E1_fn_only`, the reduction in missed objects | perception-gain oracle at the same budget |
| `E_perc_risk` | `dE_E6_risk_weighted`, criticality-weighted | perception-gain oracle at the same budget |

**Method pool, in three tiers.** The headline is tiers 1 and 3; tier 2 is reported but flagged, because those
methods are trained on V and are therefore advantaged under `E_dec` by construction.

| tier | methods |
|---|---|
| 1, target-free | `random`, `uncertainty`, `criticality_cheap`, `trivial_ego_speed` |
| 2, V-trained (advantaged under `E_dec`) | `gate_ridge`, `gate_gbm`, four `R1_*` routers, `R2_cnn_clf` |
| 3, perception diagnostics | `dE_exact`, `dE_E1_fn_only`, `dE_E6_risk_weighted`, `PKL`, `TIP` |

**Sanity check.** The `E_dec` column reproduces the official held-out nDG in **688 of 688 comparable rows**, to a
maximum absolute difference of 1.1e-16. A further 40 rows have no official counterpart: they are the nuPlan IDM
safety cell, which the benchmark declares undefined (4 affected test states, below the minimum of 10). That cell
is excluded from every summary below; including it does not change any conclusion.

In the table, `section=score` rows carry `ndg_defined` per objective, and every `section=compare` row carries
`e_dec_defined` with `n_affected_dec`. The two differ on purpose: regret is measured in decision value, so the
nuPlan IDM safety cell is flagged `e_dec_defined=False` on all 16 of its compare rows, while its `E_perc` score
rows remain defined — the perception gain is non-zero on many more inputs than the decision value is.

## The two objectives disagree about which method to use

Headline pool, 13 defined cells × 4 budgets = 52 comparisons per variant.

| | `E_perc_dE` | `E_perc_risk` |
|---|---|---|
| argmax differs | 42 / 52 | 40 / 52 |
| Kendall tau between the rankings, median | +0.171 | 0.000 |
| selection regret, median nDG | −0.104 | −0.122 |
| selection regret, median share of the all-cheap loss | −0.008 | −0.007 |
| the `E_perc` winner is worse than random on decision value | 16 / 52 | 22 / 52 |
| regret intervals excluding zero | 4 / 52 | 0 / 52 |

**What the objectives pick.** Under `E_perc` the winner is a perception diagnostic in **104 of 104** headline
comparisons — `dE_E6_risk_weighted` 52, `dE_exact` 40, `dE_E1_fn_only` 12. Under `E_dec` the winner is split
evenly between the target-free tier and the diagnostics, 52 and 52; `trivial_ego_speed` alone wins 34 of them.
Part of this is definitional and must not be read as a finding: under `E_perc` with value G, the signal equal to
G scores 1.0 by construction.

By track (headline, defined cells):

| track | objective | n | argmax differs | median regret nDG | worse than random |
|---|---|---|---|---|---|
| nuScenes | `E_perc_dE` | 24 | 17 | −0.041 | 5 |
| nuScenes | `E_perc_risk` | 24 | 15 | −0.074 | 8 |
| KITTI | `E_perc_dE` | 16 | 14 | −0.146 | 0 |
| KITTI | `E_perc_risk` | 16 | 13 | −0.285 | 8 |
| nuPlan | `E_perc_dE` | 12 | 11 | −0.231 | 11 |
| nuPlan | `E_perc_risk` | 12 | 12 | −0.169 | 6 |

**With the V-trained tier added** (104 comparisons): the argmax differs in 87, the median regret grows to −0.229
nDG, and the `E_perc` winner is worse than random on decision value in 38. `E_dec` then picks a V-trained method
in 44 of 104 — the advantage that tier is flagged for.

## Reading

**The choice of objective changes the chosen method in most cells, and the perception-chosen method is
frequently worse than random on the decision.** The two rankings are close to unrelated: median Kendall tau is
+0.171 for the missed-object gain and 0.000 for the risk-weighted one. Selecting by perception gain costs a
median of about 0.10–0.12 nDG, and in 16–22 of 52 cells the selected method does not even beat random.

**The honest qualification is the interval, not the point.** With 24, 6 and 9 test units per track, the paired
bootstrap is wide: only 4 of 52 regrets exclude zero under `E_perc_dE`, and none under `E_perc_risk`. The
direction is consistent across tracks, budgets and both perception variants, and the sign of the median never
flips; but on this data the per-cell regret is mostly not separable from zero. What is separable is the
disagreement itself — which method gets chosen — not the size of the loss it causes.

**What this supports.** It is evidence that a perception-centric evaluation is not a proxy for a decision-centric
one: the two objectives select different allocators, and the perception-selected allocator carries no reliable
decision benefit. It is not evidence that perception metrics are worthless, and nothing here revisits Task 9's
null on training targets.
