# Phase 0G — final verdict: planner-conditionality falsification

*Written 2026-09-13 22:00, in the order the Phase 0G specification requires. Supporting documents:
`iclr_phase0g_planner_d.md` (Track A), `iclr_phase0g_external_planners.md` (Track B),
`iclr_corrected_results.md` (every post-review number). FDE robustness and the deployable gate are
add-ons outside this order and are reported in `iclr_corrected_results.md` when they land.*

## 1. Pre-registered question

> The marginal downstream value of additional perception compute is conditional on the downstream
> decision maker, and an allocation score that works for one planner need not preserve the
> valuable-input ranking for another.

The test is whether the identity of the decision maker changes the **ordering** of
V_i^q = J_q(π_q(z_i^cheap), s_i) − J_q(π_q(z_i^full), s_i), not merely its scale.

## 2. Frozen before results

RESEARCH_LOG.md, 2026-09-12 16:45, before any Phase 0G allocation number: the scene split (667 / 98 /
85, rule-generated, asserted disjoint); Planner D architecture, target, loss, both variants and their
corruption values; the viability gate (validation ADE ≥ 10% below constant velocity); the primary
real-trajectory ADE cost; quotas; the 400-draw scene bootstrap; all eight falsifiers; and the Track B
deviation for the perception intervention. Two amendments were recorded before any training run
(excluding frames whose horizon leaves the scene; capping training scenes by measured throughput),
and one architecture change was made under rule A7 using validation ADE only.

## 3. Planner D viability

**Failed on all six runs — D-F4 fires.**

| variant | validation ADE (3 seeds) | constant velocity | improvement | pass |
|---|---|---|---|---|
| D-GT | 2.004 ± 0.050 m | 2.002 m | −0.1% | 0/3 |
| D-Aug | 1.984 ± 0.015 m | 2.002 m | +0.9% | 0/3 |

The entire BEV scene contributes about 1% beyond the ego's own velocity, reproducing the known result
that nuScenes open-loop planning is ego-status dominated. Three implementation defects were caught
before this conclusion was accepted: a mirrored coordinate frame, a withheld ego-velocity input, and a
central-difference velocity that read the pose half a second into the future and had inflated the
baseline by 27%.

## 4. Planner D primary ADE result

Not interpreted, per D-F4. Recorded as a property of the setting: quadrupling detector compute moves
Planner D's trajectory error by about 0.0003 m on average, with 47.6–53.6% of affected frames made
worse — absent and symmetric decision value.

## 5. C-versus-D ranking transfer

**Unassessable.** Comparing rankings against a decision value that is identically near zero carries
no information, so D-F1 and D-F2 can neither pass nor fail.

What Track A did establish instead: scoring PKL's own planner (Planner C) against the real
trajectory rather than against its own ground-truth-conditioned output lowers PKL's η@20 from
**0.872 to 0.395** (oracle geometry; 0.420 under mono) — most of the headline was shared functional
form. PKL's released planner is itself 47% worse than constant velocity as a point predictor.

## 6. D-GT versus D-Aug robustness

Both variants fail viability by the same margin and have the same near-zero, sign-symmetric decision
value, so the D-GT failure is not an artefact of the train/test perception gap.

## 7. External PDM-Closed / IDMPlanner setup validation

Untouched official simulation passed for both (4/4 scenarios each, same tokens, official metrics
written). Planner parameters match the released configs exactly. 8,640 counterfactual branches
(1,440 states × 3 branches × 2 planners) all computed a trajectory, each on a fresh planner instance.
Intervention wiring proven: removing every object in the camera changes the trajectory on 3 of 6
probe states; eight unit tests of the intervention pass. The intervention is a detector model fitted
to 46,469 real YOLOv8s outcomes, validated on held-out sequences (Brier 0.144 vs base rate 0.251).

## 8. External sign-varying result

**Replicates.** P(V < 0 | V ≠ 0) = **0.406** (IDMPlanner) and **0.303** (PDM-Closed) on the scalar
cost; 0.314 and 0.371 on safety alone.

## 9. External oracle@20 versus all-FULL

**Selective allocation beats uniform full fidelity on both.** PDM-Closed, safety cost: oracle@20 cuts
21.1% against 14.7% for all-FULL; selectivity adds **30–37%** of the achievable benefit across both
planners and both scalar costs.

## 10. PDM-Closed versus IDMPlanner ranking transfer

**The rankings transfer.** Safety cost: pairwise disagreement 0.109, γ **+0.78** [+0.48, +0.98],
best-compatible transfer 1.000 / 0.997, same sign on 19 of the 21 states both respond on. The 2×2
separating planner from objective agrees in every cell, including one planner under two objectives
(γ +0.70). For contrast, braking controller against PKL's planner on nuScenes: disagreement 0.525,
γ −0.050.

## 11. All falsifier outcomes

| falsifier | outcome |
|---|---|
| D-F1 PKL/TIP transfer to an independent learned planner | unassessable (D invalid) |
| D-F2 planner identity does not change oracle allocation (C vs D) | unassessable (D invalid) |
| D-F3 sign variation disappears on Planner D | does not fire (47.6–53.6% harmful), reported as a property of the setting under D-F4 |
| **D-F4** Planner D fails constant velocity | **fires, 6/6** |
| B-F1 sign variation disappears externally | does not fire (0.30 / 0.41) |
| B-F2 selective allocation adds < 10% externally | does not fire (30–37%) |
| **B-F3** external planner conditionality disappears | **does not fire by its letter** (overlap@20 = 0.22); **its intent is met** on the safety cost, where the tie-robust measures show agreement — the letter fails only because quota overlap is not tie-robust at 2–5% response rates |
| B-F4 one published score solves both external planners | not assessed |

## 12. Claims that survive

1. **The frame-level value of additional perception compute is sign-varying**: 45–52% of affected
   frames are made worse on nuScenes across two geometry variants, and 30–41% on two published
   nuPlan planners.
2. **Selective allocation beats uniform full fidelity** on every downstream system measured, including
   both external planners.
3. **Between the braking controller and PKL's own planner on nuScenes, which frames deserve compute
   is unrelated** (pairwise disagreement 0.525, γ −0.050 over 464,238 strictly ranked pairs), and the
   ranking of allocation signals reverses between them — reported as a finding about those two
   dissimilar systems.
4. **No published planning-aware score approaches the decision oracle** on any valid target tested
   (maximum PKL η@20 = 0.420).
5. **nuScenes open-loop waypoint prediction is a poor target for adaptive-perception evaluation**:
   a learned planner there extracts about 1% from the scene, and its decision value is absent.

## 13. Claims that must be removed

- **"The value of perception compute is conditional on the downstream decision maker"** as a general
  property. It holds between two dissimilar systems on nuScenes and does not replicate on nuPlan,
  where neither planner identity nor objective produces disagreement.
- The "objective-conditionality" reframing proposed during analysis: refuted by one planner agreeing
  with itself across two objectives.
- Any claim that PKL's advantage on its own planner (0.872) reflects allocation skill.
- Any claim resting on Planner D's allocation numbers.
- "Planning-aware metrics fail at compute allocation": PKL reaches 0.395–0.420 on its own planner.
- Any closed-loop claim: every evaluation is open-loop at the state level.

## 14. Final verdict

# WEAK GO

The specification's WEAK GO clause — *"external effect present but planner transfer remains high"* —
describes the outcome exactly: the sign-varying effect and the value of selective allocation replicate
on two published planners, while the external planners' valuable-state rankings transfer almost
perfectly. It is not STRONG GO or GO: Track A produced no valid controlled evidence and Track B
contradicts the conditionality claim. It is not NO-GO: no existing score approaches the decision oracle
anywhere, and two of the three original claims gained external replication.

**The paper that follows is not the one pre-registered.** Its leading claims become 1 and 2 of §12,
supported in-domain and externally; downstream-conditionality is demoted to a scoped observation
reported together with the nuPlan evidence against it. The external test is weak at detecting
disagreement — shared IDM core, whole-object removal, about 17% of objects in the camera cone, 15–72
effective states — and that weakness is itself a limitation to state, not a reason to discount the
result.
