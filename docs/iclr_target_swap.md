# Target swap: the same allocators trained on perception gain instead of decision value

**Question.** Does the allocation advantage come from the decision-value objective, or from the architecture?

**Method.** Each learned allocator is scored against V = J(CHEAP) − J(FULL) twice:
* with its official V-target scores;
* retrained by the same code on a perception-gain label G, with the same inputs, hyperparameters, seeds and
  training units.

**Provenance.**
* Pre-registration: `RESEARCH_LOG.md`, Task 9 (commit `33f96a4`), committed before any G-target model was trained.
* Code: `scripts/122_target_swap.py`; the audit in section 9 is `scripts/129_target_swap_audit.py`.
* Outputs: `results/final/benchmark_target_swap.csv` (1,106 rows) and `benchmark_target_swap_summary.json`.
* No official result file was changed.

**Architectures.**

| family | variants |
|---|---|
| gates (engineered features; 18 on nuPlan real perception) | `gate_ridge`, `gate_gbm` |
| detection-list routers (R1) | `R1_mlp_reg`, `R1_mlp_clf`, `R1_gbm_reg`, `R1_gbm_clf` |

Regression models are fit on G itself; classifiers on 1[G > 0].

**Perception gain G.** Error of CHEAP minus error of FULL, so it is oriented like V.

| G | core (KITTI, nuScenes) | nuPlan real perception |
|---|---|---|
| **primary (deciding)** | `dE_E5_combined` (FN, FP and localisation error) | change in missed eligible in-camera tracks plus false-positive agents |
| secondary | `dE_exact` (FN count) | change in missed tracks |
| secondary | `dE_E6_risk_weighted` | `dE_E6_risk_weighted` |

G is a training label only; it never enters the inputs.

**Cells.**
* The 14 official held-out cells: 10 core plus 4 nuPlan real-perception cells.
* IDM safety is undefined (4 affected test states).

**Statistic.**
* nDG against V on the frozen test split, with the exact tie expectation.
* 1,000 bootstrap draws. Units are resampled once per dataset per draw and shared by every cell, target and
  architecture, so all differences are paired.

## 1. Checks

| check | result |
|---|---|
| (a) reproduces the official nDG | **434 of 434** values (every architecture × cell × quota, plus the 20% ms budget) to 3 decimals; maximum absolute difference 1e-16 |
| inputs and labels | official V, frame keys and order equal; no G column among the inputs; the official leakage guards ran unchanged |
| nuPlan G | false-positive counts equal `n_tracks − n_tracks_nofp` on all states; the missed-track change equals 119's `dE_E1_fn_only` |
| bootstrap drops at 20% | a draw is dropped when its prize falls below a quarter of the full-sample prize. Both PDM-Closed cells 319 draws each, KITTI oracle traj 85, KITTI mono traj 29, every other cell ≤ 7. In 382 of 1,000 draws at least one pooled cell was left out |

## 2. Pre-registered reading: **architecture-driven**

**Pooled test.** Primary G, 20% quota, 12 cells (10 core plus the two PDM-Closed cells). The statistic is the mean
paired difference Δ = nDG(V-target) − nDG(G-target).

| architecture | mean Δ | 95% CI | cells with V > G | cells where V wins significantly | cells where G wins significantly |
|---|---|---|---|---|---|
| gate_ridge | +0.118 | [−0.080, +0.275] | 9/12 | 2 | 0 |
| gate_gbm | +0.106 | [−0.058, +0.217] | 7/12 | 3 | 0 |
| R1_mlp_reg | +0.040 | [−0.083, +0.141] | 6/12 | 0 | 0 |
| R1_mlp_clf | +0.117 | [−0.037, +0.197] | 9/12 | 2 | 0 |
| R1_gbm_reg | −0.069 | [−0.186, +0.131] | 6/12 | 0 | 2 |
| R1_gbm_clf | +0.013 | [−0.070, +0.164] | 7/12 | 1 | 2 |

**Why this reading.**
* Every pooled CI includes 0, which is the registered condition for "architecture-driven".
* Both secondary G variants give the same reading. Their means lie between −0.057 and +0.157, and every CI includes 0.

**What the point estimates say.**
* Five of six architectures lean towards the V-target: both gates and R1_mlp_clf by about 0.12, R1_mlp_reg by 0.04
  and R1_gbm_clf by 0.01.
* Only R1_gbm_reg leans towards the G-target, by 0.07.
* None of these leans is significant.

## 3. The 20% table (primary G)

**How to read it.**
* Each entry is nDG(V-target) / nDG(G-target) / Δ.
* **▲** marks a Δ CI entirely above 0 (V-target better); **▼** a Δ CI entirely below 0 (G-target better).

| cell | gate_ridge | gate_gbm | R1_mlp_reg | R1_mlp_clf | R1_gbm_reg | R1_gbm_clf |
|---|---|---|---|---|---|---|
| nuScenes oracle · brake | 0.20 / 0.05 / +0.15 | −0.06 / 0.06 / −0.12 | 0.19 / 0.40 / −0.21 | 0.13 / 0.29 / −0.16 | 0.32 / 0.33 / −0.01 | 0.35 / 0.07 / +0.28 |
| nuScenes oracle · plan_ade | −0.08 / 0.16 / −0.25 | −0.13 / −0.03 / −0.10 | −0.06 / −0.02 / −0.04 | 0.05 / −0.07 / +0.12 | −0.11 / −0.02 / −0.09 | −0.06 / −0.07 / +0.01 |
| nuScenes oracle · plan_fde | −0.01 / 0.27 / −0.28 | −0.21 / 0.06 / −0.27 | 0.11 / 0.00 / +0.10 | 0.18 / 0.03 / +0.15 | −0.05 / −0.06 / +0.01 | 0.20 / −0.07 / +0.27 |
| nuScenes mono · brake | 0.25 / 0.09 / +0.16 | 0.27 / 0.13 / +0.15 | 0.08 / 0.18 / −0.10 | 0.26 / 0.36 / −0.10 | 0.27 / 0.24 / +0.03 | 0.25 / −0.07 / +0.32 |
| nuScenes mono · plan_ade | 0.02 / −0.00 / +0.02 | 0.09 / 0.09 / −0.00 | 0.14 / 0.05 / +0.09 | 0.14 / 0.04 / +0.10 | 0.14 / 0.06 / +0.08 | 0.12 / 0.14 / −0.02 |
| nuScenes mono · plan_fde | −0.06 / −0.10 / +0.03 | −0.04 / −0.03 / −0.01 | 0.17 / 0.00 / +0.16 | 0.11 / −0.03 / +0.15 | 0.03 / 0.07 / −0.04 | 0.02 / 0.14 / −0.13 |
| KITTI oracle · brake | 0.23 / 0.14 / +0.09 | 0.20 / 0.09 / +0.11 | 0.24 / 0.03 / +0.21 | 0.29 / 0.12 / +0.18 | 0.27 / 0.16 / +0.12 | 0.35 / 0.19 / +0.16 **▲** |
| KITTI oracle · traj | 0.46 / 0.26 / +0.20 | 0.58 / 0.29 / +0.28 **▲** | 0.41 / 0.17 / +0.24 | 0.46 / 0.07 / +0.39 **▲** | 0.49 / 0.25 / +0.23 | 0.49 / 0.30 / +0.19 |
| KITTI mono · brake | 0.15 / 0.07 / +0.08 | 0.11 / 0.02 / +0.10 | 0.17 / 0.02 / +0.16 | 0.17 / 0.05 / +0.12 | 0.20 / 0.05 / +0.15 | 0.22 / 0.14 / +0.08 |
| KITTI mono · traj | 0.13 / 0.34 / −0.22 | 0.26 / 0.25 / +0.00 | 0.21 / 0.21 / −0.00 | 0.34 / 0.06 / +0.27 | 0.28 / 0.34 / −0.06 | 0.11 / 0.30 / −0.18 |
| nuPlan PDM-Closed · safety | 0.99 / 0.29 / +0.71 **▲** | 0.99 / 0.43 / +0.57 **▲** | 0.14 / 0.22 / −0.07 | −0.07 / 0.01 / −0.07 | 0.07 / 0.70 / −0.63 **▼** | 0.07 / 0.49 / −0.42 **▼** |
| nuPlan PDM-Closed · scalar_J | 0.99 / 0.29 / +0.70 **▲** | 0.99 / 0.42 / +0.56 **▲** | 0.14 / 0.20 / −0.06 | 0.28 / 0.01 / +0.27 **▲** | 0.06 / 0.70 / −0.63 **▼** | 0.07 / 0.48 / −0.40 **▼** |
| nuPlan IDM · safety | undefined | undefined | undefined | undefined | undefined | undefined |
| nuPlan IDM · scalar_J (not pooled) | 0.06 / 0.00 / +0.05 | 0.00 / −0.07 / +0.07 | 0.03 / 0.00 / +0.03 **▲** | 0.94 / 0.03 / +0.91 **▲** | 0.00 / 0.00 / +0.00 | 0.06 / −0.08 / +0.14 **▲** |

**Mean Δ at 20% by dataset.** Exploratory: point estimates, not pre-registered.

| group | cells | gate_ridge | gate_gbm | R1_mlp_reg | R1_mlp_clf | R1_gbm_reg | R1_gbm_clf |
|---|---|---|---|---|---|---|---|
| KITTI | 4 | +0.041 | +0.123 | +0.151 | +0.241 | +0.110 | +0.061 |
| nuScenes | 6 | −0.027 | −0.060 | +0.001 | +0.041 | −0.002 | +0.121 |
| nuPlan PDM-Closed | 2 | +0.704 | +0.566 | −0.067 | +0.098 | −0.631 | −0.410 |
| nuPlan IDM (scalar_J) | 1 | +0.053 | +0.072 | +0.030 | +0.911 | +0.000 | +0.135 |

## 4. Where the target matters

**The G-target beats the V-target significantly.**

| setting | rows |
|---|---|
| primary G, 20% | R1_gbm_reg and R1_gbm_clf on both PDM-Closed cells: Δ −0.63 and −0.40 to −0.42. The same four rows hold at every quota |
| primary G, other quotas | KITTI mono traj with gate_ridge, at 30% (Δ −0.47) and 50% (−0.49) |
| secondary G | nuScenes oracle plan_ade with R1_gbm_reg (`dE_exact`, 20–50%). Several PDM-Closed rows for the R1 GBMs and for R1_mlp_clf (`dE_E6_risk_weighted`). A few IDM scalar_J rows, mostly with nDG below 0.01 |

**The V-target beats the G-target significantly.** At 20% with primary G, 8 of the 72 pooled cell × architecture
pairs, against 4 for G:

| cells | architectures |
|---|---|
| both PDM-Closed cells | gate_ridge, gate_gbm; R1_mlp_clf on scalar_J |
| KITTI oracle traj | gate_gbm, R1_mlp_clf |
| KITTI oracle brake | R1_gbm_clf |

Outside the pool, IDM scalar_J also favours the V-target for R1_mlp_reg, R1_mlp_clf (0.94 against 0.03) and
R1_gbm_clf.

**nuScenes.** No cell differs significantly at 20%. Both targets give low, noisy nDG there.

## 5. How similar the two labels are

**Sign agreement.** P(sign G = sign V | both non-zero) on the train split, primary G.

| cells | agreement (pairs) |
|---|---|
| KITTI oracle brake / traj | 0.83 (855) / 0.85 (440) |
| KITTI mono brake / traj | 0.69 (1,384) / 0.62 (762) |
| nuScenes oracle brake / plan_ade / plan_fde | 0.58 (169) / 0.54 (651) / 0.55 (294) |
| nuScenes mono brake / plan_ade / plan_fde | 0.54 (265) / 0.49 (749) / 0.53 (332) |
| PDM-Closed safety / scalar_J | 0.73 (11) / 0.68 (28) |
| IDM scalar_J | 0.40 (10) |

**What the agreement suggests.**
* On nuScenes the two labels agree in sign about as often as chance, yet the two targets still end up with similar,
  low nDG. That suggests neither label is learnable enough from the CHEAP inputs there to separate them.
* On KITTI oracle the labels agree most, and the V-target still leads for all six architectures in both KITTI oracle
  cells. Three of those twelve leads are significant.

## 6. Absolute reduction, comparison with random, budget

**Mean absolute loss reduction at 20% over the 12 pooled cells** (nDG × oracle reduction):

| architecture | V-target | G-target |
|---|---|---|
| gate_ridge | 10.4% | 4.6% |
| gate_gbm | 10.5% | 5.2% |
| R1_mlp_reg | 5.0% | 3.7% |
| R1_mlp_clf | 5.7% | 2.2% |
| R1_gbm_reg | 5.7% | 7.9% |
| R1_gbm_clf | 5.7% | 6.0% |

**Rows beating random at 20%** (paired lower bound > 0, 13 defined cells):

| architecture | V-target | G-target |
|---|---|---|
| gate_ridge | 2 | 1 |
| gate_gbm | 3 | 2 |
| R1_mlp_reg | 1 | 0 |
| R1_mlp_clf | 3 | 0 |
| R1_gbm_reg | 2 | 3 |
| R1_gbm_clf | 2 | 3 |

**20% latency budget.**
* The budget is the same for both targets; only the ranking differs.
* Gates and GBM routers escalate no frames on core cells: their per-call overhead exceeds the budget. There V and G
  are identical by construction.
* Where frames are escalated, the V-target beats the G-target significantly in 10 rows and the G-target wins 1 row
  (IDM scalar_J batched gate, Δ −0.001).

| escalated fraction | gate_ridge | gate_gbm | gate_gbm_batched | R1_mlp_* | R1_gbm_* |
|---|---|---|---|---|---|
| nuScenes | 0 | 0 | 1.7% | 16.6% | 0 |
| KITTI | 0 | 0 | 0.8% | 16.5% | 0 |
| nuPlan | 3.3% | 0 | 5.0% | 17.2% | 0 |

## 7. Caveats

* **The pooled mean is dominated by the two PDM-Closed cells.**
  * Their differences are an order of magnitude larger than on core cells and of opposite sign for gates and GBM
    routers.
  * Their prize sits in one log and one scenario (Task 7), and they are dropped from 319 of 1,000 draws.
* **Sparse V labels may explain the G-target's win for the PDM-Closed GBM routers.**
  * On the fitting states (train ∪ val), V is non-zero on 1.7% (safety) and 5.3% (scalar_J), and G on 74.1%
    (`fit_share_V_nonzero`, and `fit_share_G_pos` + `fit_share_G_neg`, in the audit's `a2_G_labels.csv`).
  * V and G are both non-zero on only 11 (safety) and 28 (scalar_J) training states (`sign_agree_train_n` in
    `benchmark_target_swap_summary.json`).
  * The G-trained GBMs may simply learn from denser supervision. This does not show that perception gain is the
    better objective.
* **IDM scalar_J rests on 10 affected test states** and is excluded from the pooled test, as registered.
* **"At least one R1 variant" is a lenient criterion.** It did not matter here: no pooled CI excludes 0 for any
  variant.
* **The core G labels are ground-truth error metrics.** The G-target models learn to predict them from CHEAP inputs,
  exactly as the V-target models learn V.

## 8. What this changes for the paper

* **Do not claim that the decision-value objective, rather than the architecture, drives the allocation
  advantage.** Under the registered test that claim is not supported for any architecture, with any of the three
  perception-gain labels.
* **What the data do support:**
  * The effect of the objective is cell- and architecture-specific, and it goes both ways.
  * V-trained gates win decisively on PDM-Closed, which rests on one scenario.
  * V-trained allocators lean ahead on KITTI, significantly for three architecture × cell pairs on KITTI oracle.
  * G-trained GBM routers win on PDM-Closed.
  * Averaged over the pooled cells, the V-trained gates recover about twice the absolute loss reduction of their
    G-trained counterparts: 10.4–10.5% against 4.6–5.2%, point estimates.
* **How to state it.** Per cell and per architecture, with the pooled null result reported alongside.

## 9. Audit after the results: no computation error found

Requested after the results were read. Code: `scripts/129_target_swap_audit.py`; outputs in
`results/raw/*_target_swap_audit/`.

| check | result |
|---|---|
| G-target code path with V as label | reproduces the official V-target scores in 84/84 architecture × cell rows (1 differs by 1e-16; nDG identical). The G models differ from the official ones in the label alone |
| degenerate G-target scores | none constant; no tie share above 0.5 at the 20% cut; all 252 recomputed nDG values equal 122's |
| per-cell bootstrap | CI width against the official per-cell CIs: median ratio 1.00 (0.58–1.09); 13 vs 12 rows beat random at 20% |
| pooled bootstrap | re-run with the same seed reproduces the pooled CIs exactly |
| G labels | equal the official `dE_*` diagnostic columns. On nuScenes, `dE_exact` is positive on average (+0.85), so the sign is right; `dE_E5_combined` is negative on 49% of frames, because at 640 nuScenes carries more FP and localisation error |

**Why the result is null** (exploratory, not the registered test):
* **G says little about V.** Spearman(G, V) on the fitting units is at most 0.22. Ranked against V at 20%, the G label
  itself reaches −0.08 to 0.19 on nuScenes, 0.16–0.66 on KITTI and 0.30 on PDM-Closed.
* **The official V-target allocators beat random in only 12 of 78 rows at 20%**, none on nuScenes. There is little
  V-target advantage to lose.
* **On the 10 core cells the target makes no difference to the gates:** mean Δ +0.0002 and +0.0136
  (`A5.exploratory.<arch>.core10_point` in the audit record, equal to the mean of `diff` over those cells in the table). The R1 classifiers
  lean towards V by +0.12 (mlp_clf) and +0.10 (gbm_clf), and every CI includes 0.
* **The two PDM-Closed cells carry the pooled signal in both directions:** gates +0.70 and +0.57, GBM routers −0.63
  and −0.41, all significant. They are dropped from 319 draws, and the pooled mean then averages 10 instead of 12
  cells.
* **Restricting to the 618 draws with all 12 cells present puts both gate CIs above 0.** That subset is conditional
  on resampling PDM-Closed's dominant log, so it is not an unbiased interval and does not change the reading.

**Two registered design choices a reader may question:**
1. **Units are resampled once per dataset and shared by that dataset's cells.** Independent per-cell resampling
   would ignore that the cells share scenes and sequences, and would narrow the pooled CI.
2. **The pooled mean averages the cells that survive the 25%-prize rule in each draw.**
