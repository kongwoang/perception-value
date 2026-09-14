# Phase 0G, Track B — published external planners on nuPlan

> **Note (2026-09-15).** IDM numbers in this document predate the IDM route fix and are superseded. In 58% of nuPlan states the pipeline gave IDM a route it could not start from. Corrected values and the before/after comparison are in `docs/iclr_idm_route_fix.md`. PDM-Closed, nuScenes and KITTI numbers are unaffected.

*Branch `exp/phase0g-planner-conditionality`. Pre-registered 2026-09-12 (RESEARCH_LOG.md), with the
deviations listed in §8 recorded before the results they affect. Numbers from
`results/final/phase0g_external_planner_{summary,transfer}.csv`, `phase0g_external_2x2.csv`.*

## 1. Question

Every downstream system in Phase 0F and Track A was either written by us or near-circular with the
metric under test. Track B asks whether the findings survive **planners we did not design**:
PDM-Closed (tuPlan Garage) and IDMPlanner (nuPlan devkit), at their released configurations. It also
serves the same-objective test: two published planners scored by the **same** cost on the **same**
states, so any disagreement cannot be attributed to a difference in objective.

## 2. Setup validation

**Code and data.** nuplan-devkit `e9241677`, tuPlan Garage `b51d5d04`, nuPlan v1.1 mini (8.0 GB db,
maps v1.0). CPU-only Python 3.9 environment; torch 2.3.1 via pip; two libgomp copies preloaded for
the aarch64 static-TLS defect. Planner parameters were checked field by field against
`idm_planner.yaml` and `pdm_closed_planner.yaml` and match exactly. Full compatibility log in
`third_party/PROVENANCE.md`.

**B2 — untouched official simulation.** `closed_loop_nonreactive_agents`, 4 scenarios: IDMPlanner 4/4
and PDM-Closed 4/4 succeeded on the same scenario tokens with official metrics written. IDMPlanner's
"could not find valid path to the target roadblock" warning also appears in this untouched run, so it
is the released planner's own behaviour on mini, not a harness fault.

**Counterfactual design (B5).** For each of 1,440 logged states per planner (60 scenarios × 24
states), a **fresh planner instance** is initialised for each of three branches — reference tracks,
cheap-detector tracks, full-detector tracks — so no branch can leak state into another (PDM-Closed is
stateful across steps). Each proposed trajectory is scored against the **real** tracked objects over
0.5–4.0 s. All 8,640 branches computed a trajectory.

**B6 — wiring.** Eight unit tests of the intervention pass (reference untouched, identical modes
identical, outcomes invariant to list order, FULL reports more but still loses some, range effect,
objects outside the camera never dropped, field of view follows heading, outcomes vary across
iterations). In the simulator, a `drop_all` probe that removes every object in the camera changed the
trajectory on 3 of 6 states, so the filtered observation reaches the planner; the unchanged states are
ones whose lead agent lies outside the camera cone.

## 3. The perception intervention

nuPlan camera data for mini is ~450 GB against 138 GB free, so YOLOv8s could not run in the simulator.
The detector's behaviour was **measured and transported** instead: a conditional model (detect at 320;
rescue what 320 missed; lose what 320 found) fitted on 46,469 real per-object outcomes over 21 KITTI
sequences.

| validation (every third sequence held out) | model | observed |
|---|---|---|
| Brier, cheap | 0.144 | base rate 0.251 |
| Brier, full (implied) | 0.158 | base rate 0.192 |
| P(detected at 640 \| detected at 320) | 0.968 | 0.969 |
| monotone violation (found at 320, lost at 640) | 0.0150 | 0.0146 |
| recall by class (vehicle / pedestrian / bicycle) | 0.481 / 0.481 / 0.161 | 0.491 / 0.481 / 0.159 |
| worst distance band (30–40 m), cheap recall | 0.248 | 0.352 |

The loss channel is what lets escalation hurt, so it is fitted rather than assumed away. Classes the
fit never saw (cones, barriers, signs) are never dropped — their dummy column was all-zero at fit
time, which would otherwise have silently given them a car's miss profile. Only objects inside a 60°,
80 m forward cone are affected; in the wiring probe about 17% of tracked objects fell inside it.

## 4. Sign-varying value on external planners (B7)

| planner | cost | V > 0 | V = 0 | V < 0 | **P(V<0 \| V≠0)** |
|---|---|---|---|---|---|
| IDMPlanner | collision | 14 | 1,421 | 5 | 0.263 |
| PDM-Closed | collision | 27 | 1,405 | 8 | 0.229 |
| IDMPlanner | safety (collision + clearance) | 24 | 1,405 | 11 | 0.314 |
| PDM-Closed | safety | 44 | 1,370 | 26 | 0.371 |
| IDMPlanner | scalar J | 57 | 1,344 | 39 | **0.406** |
| PDM-Closed | scalar J | 122 | 1,265 | 53 | **0.303** |

**Sign variation replicates on both published planners.** The signal is sparse: 1.3–12.2% of states
respond, depending on planner and cost.

## 5. Selective allocation versus uniform full fidelity (B7)

| planner | cost | all-FULL cuts | oracle@20 cuts | selective extra share |
|---|---|---|---|---|
| IDMPlanner | safety | 3.9% | 6.1% | 0.364 |
| PDM-Closed | safety | 14.7% | 21.1% | 0.303 |
| IDMPlanner | scalar J | 2.0% | 3.2% | 0.371 |
| PDM-Closed | scalar J | 12.6% | 17.9% | 0.297 |

*Selective extra share* = (oracle@20 − all-FULL) / oracle@20: the part of the oracle's achievable
improvement that selectivity adds beyond running the expensive mode everywhere. **Selective allocation
beats uniform full fidelity on both planners**, by 30–37% of the achievable benefit.

## 6. PDM-Closed versus IDMPlanner — the same-objective test (B8)

| cost | responsive (IDM / PDM) | both respond | strict pairs | disagreement | γ | best-compatible @20 (IDM→PDM / PDM→IDM) |
|---|---|---|---|---|---|---|
| collision | 1.3% / 2.4% | 15 | 21,270 | 0.002 | **+0.995** [+0.99, +1.00] | 1.000 / 1.000 |
| safety | 2.4% / 4.9% | 21 | 30,620 | 0.109 | **+0.781** [+0.48, +0.98] | 1.000 / 0.997 |
| scalar J | 6.7% / 12.2% | 72 | 103,524 | 0.326 | +0.348 [+0.13, +0.59] | 0.918 / 0.982 |

Sign agreement on the states both planners respond on: collision **15/15**, safety **19/21 = 0.905**
[0.73, 1.00], scalar J 49/72 = 0.681 [0.55, 0.81]; chance is 0.5.

**The two planners agree.** Comparing braking controller and planner on nuScenes, which differ in
objective: pairwise disagreement 0.525, γ −0.050.

## 7. Separating objective from planner

B8 alone cannot say whether agreement comes from the shared objective or the shared IDM core. Costs
were recorded per component, which gives the full 2×2 on the same states:

| contrast | pair | both respond | γ [95% CI] |
|---|---|---|---|
| same planner, different objective | IDM safety vs IDM progress deviation | 35 | **+0.701** [+0.44, +0.89] |
| same planner, different objective | PDM safety vs PDM progress deviation | 70 | +0.434 [+0.19, +0.68] |
| different planner, same objective | IDM vs PDM, safety | 21 | +0.782 [+0.44, +0.98] |
| different planner, same objective | IDM vs PDM, progress deviation | 72 | +0.342 [+0.09, +0.59] |
| both differ | IDM safety vs PDM progress deviation | 29 | +0.559 [+0.23, +0.84] |
| both differ | PDM safety vs IDM progress deviation | 33 | +0.365 [−0.00, +0.70] |

**Every cell agrees positively, including one planner scored under two objectives.** On nuPlan neither
the planner nor the objective produces the disagreement seen on nuScenes. The likeliest mechanism is a
shared factor: V is non-zero almost only where the intervention removed a relevant nearby object, and
there every cost on either planner moves the same way.

## 8. Falsifiers and deviations

| falsifier | outcome |
|---|---|
| **B-F1** sign variation disappears (both planners < 0.05) | **does not fire** — 0.23–0.41 |
| **B-F2** selectivity adds < 10% for both planners | **does not fire** — 30–37% |
| **B-F3** ranking transfers (overlap@20 ≥ 0.80 and both cross-η ≥ 0.80) | **does not fire by its letter** (overlap@20 = 0.22), **but its intent is met** on the safety cost: the letter fails only because quota overlap is not tie-robust when targets respond on 2–5% of states, and the tie-robust measures show agreement (γ +0.78, best-compatible ≈ 1.0 both ways) |
| **B-F4** one published score solves both planners | not assessed — PKL and TIP were not forced through an invalid interface |

Deviations, each recorded before the result it affects:
- **B4**: a measured, transported detector model instead of running YOLOv8s on nuPlan cameras.
- **Official metrics**: trajectories are scored by an external cost (geometric collision, clearance,
  deviation from the log), not by nuPlan's metric suite; replacing it is pending a decision.
- **Open-loop at the state level**: PDM-Closed is closed-loop-capable, but each state is evaluated as a
  counterfactual on a logged state, not as a rollout.
- **Scalar weights** (10 / 1.5 / 0.1, Planner B's) were chosen after a 112-state pilot and before the
  1,440-state run; per-component results are reported so nothing hinges on them.
- The strict-pair counts in §6 use exact ties; after a later change treating differences below 1e-9 as
  ties, the safety count is 30,606 rather than 30,620 (§7 already uses it). No reported value changes
  beyond the third decimal; §6 will be regenerated.

## 9. What Track B supports

1. **Sign-varying value replicates externally** on two published planners.
2. **Selective allocation beats uniform full fidelity externally**, by 30–37% of the achievable benefit.
3. **Downstream-conditionality does not replicate.** On nuPlan, two planners — and one planner under
   two objectives — agree about which states deserve compute.

The test is weak at detecting disagreement: the two planners share an IDM core (PDM-Closed generates
its proposals with `BatchIDMPolicy`), the intervention removes whole objects rather than perturbing
boxes, confidences and classes as the real detector does, only about 17% of objects fall in the camera
cone, and the effective sample is 15–72 states. Those limit how far the negative result generalises —
but it is the only external test available, and it does not support claim 3.
