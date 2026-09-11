"""A deterministic longitudinal safety controller, and a decision cost scored against GT.

The point of this module is to turn *perception* into an *action*, so that the value of
extra perception compute can be measured where it actually matters. Three rules keep the
measurement honest:

  * one code path. `required_decel` is called identically on CHEAP geometry, FULL geometry
    and ground-truth geometry. Nothing is tuned per mode.
  * the cost is not the Phase-0 metric renamed. It scores the *action* against the true
    scene: under-braking is a safety shortfall, over-braking wastes progress and comfort.
    `sum(criticality x detection error)` never appears.
  * ground truth enters only in the evaluator. The controller sees monocular estimates and
    ego speed (which a real vehicle reads off its own CAN bus, not off the camera).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

KEEP, DECELERATE, HARD_BRAKE = 0, 1, 2
ACTION_NAMES = ("KEEP", "DECELERATE", "HARD_BRAKE")


@dataclass(frozen=True)
class PlannerParams:
    corridor_half_w: float = 1.2     # ego half-width plus a small lateral margin [m]
    standoff: float = 2.0            # distance we want to still have when stopped [m]
    reaction_t: float = 0.4          # actuation delay folded into the stopping distance [s]
    a_decel: float = 2.5             # deceleration commanded by DECELERATE [m/s^2]
    a_hard: float = 6.0              # deceleration commanded by HARD_BRAKE [m/s^2]
    thr_decel: float = 1.0           # a_req above which DECELERATE is chosen
    thr_hard: float = 3.5            # a_req above which HARD_BRAKE is chosen
    max_range: float = 80.0
    a_cap: float = 9.0               # physical limit; a_req is clipped here
    use_ttc: bool = True             # also consider closing speed, not just ego speed


@dataclass(frozen=True)
class CostParams:
    lam_risk: float = 1.0            # weight on under-braking (safety shortfall)
    lam_brake: float = 0.12          # weight on over-braking (comfort / progress)
    lam_jerk: float = 0.02           # weight on changing the command between frames
    collision_a: float = 8.0         # shortfall beyond this counts as an outright failure
    lam_collision: float = 6.0


def in_corridor(lat_min, lat_max, half_w: float) -> np.ndarray:
    return (np.asarray(lat_min) < half_w) & (np.asarray(lat_max) > -half_w)


def required_decel(dist, lat_min, lat_max, ttc, v_ego: float,
                   p: PlannerParams) -> tuple[float, np.ndarray]:
    """Deceleration needed to stop behind the most constraining in-corridor obstacle.

    Returns the scalar requirement and the per-obstacle requirement, so the caller can
    see *which* object drove the decision.
    """
    dist = np.asarray(dist, dtype=np.float64)
    if dist.size == 0:
        return 0.0, np.zeros(0)
    sel = in_corridor(lat_min, lat_max, p.corridor_half_w) & (dist <= p.max_range)
    a_obj = np.zeros(dist.shape)
    if not sel.any() or v_ego <= 0.1:
        return 0.0, a_obj

    # usable stopping distance after reaction time, never negative
    gap = np.maximum(dist[sel] - p.standoff - v_ego * p.reaction_t, 0.05)
    a_stop = v_ego ** 2 / (2.0 * gap)

    if p.use_ttc:
        # a closing obstacle needs the *relative* speed bled off, not the ego speed
        t = np.clip(np.asarray(ttc, dtype=np.float64)[sel], 0.1, 1e3)
        v_rel = np.clip(dist[sel] / t, 0.0, 60.0)
        gap_rel = np.maximum(dist[sel] - p.standoff, 0.05)
        a_close = v_rel ** 2 / (2.0 * gap_rel)
        a_stop = np.maximum(a_stop, a_close)

    a_obj[sel] = np.clip(a_stop, 0.0, p.a_cap)
    return float(a_obj.max()), a_obj


def discrete_action(a_req: float, p: PlannerParams) -> int:
    if a_req >= p.thr_hard:
        return HARD_BRAKE
    if a_req >= p.thr_decel:
        return DECELERATE
    return KEEP


def commanded_decel(action: int, p: PlannerParams) -> float:
    return (0.0, p.a_decel, p.a_hard)[action]


def decision_cost(action: int, a_req_true: float, prev_action: int | None,
                  p: PlannerParams, c: CostParams) -> dict:
    """Cost of taking `action` when the true scene demanded `a_req_true`.

    Asymmetric on purpose: failing to brake enough is a safety event, braking more than
    needed only costs progress and comfort. A symmetric cost would make the problem a
    regression on a_req and would not behave like a driving decision.
    """
    a_cmd = commanded_decel(action, p)
    shortfall = max(0.0, a_req_true - a_cmd)
    excess = max(0.0, a_cmd - a_req_true)
    collision = float(shortfall >= c.collision_a - 1e-9)
    jerk = 0.0 if prev_action is None else abs(a_cmd - commanded_decel(prev_action, p))
    J = (c.lam_risk * shortfall ** 2
         + c.lam_brake * excess ** 2
         + c.lam_jerk * jerk
         + c.lam_collision * collision)
    return {"J": J, "shortfall": shortfall, "excess": excess,
            "collision": collision, "jerk": jerk, "a_cmd": a_cmd}


def action_costs(a_req_true: float, prev_action: int | None,
                 p: PlannerParams, c: CostParams) -> np.ndarray:
    return np.array([decision_cost(a, a_req_true, prev_action, p, c)["J"] for a in range(3)])


def decision_margin(a_req_perceived: float, p: PlannerParams) -> float:
    """How much the perceived requirement would have to move to flip the command.

    The controller is a threshold rule, so its margin is the distance to the nearest
    threshold that bounds the current action — not a cost gap. (An earlier version used
    the best/second-best cost gap, which is only the right margin for a cost-minimising
    controller and was nearly constant here.)

    Deployable: computed from what the controller already has. A small margin means a
    small perception correction could change the action, which is the decision-sensitivity
    notion Phase 0C tests, and is not a restatement of uncertainty or criticality.
    """
    a = float(a_req_perceived)
    if a >= p.thr_hard:
        return a - p.thr_hard
    if a >= p.thr_decel:
        return min(a - p.thr_decel, p.thr_hard - a)
    return p.thr_decel - a


def boundary_distance(a_req: float, p: PlannerParams) -> float:
    """Unsigned distance to the nearest action threshold, whichever side it lies on."""
    return float(min(abs(a_req - p.thr_decel), abs(a_req - p.thr_hard)))


def cost_argmin_action(a_req_perceived: float, prev_action: int | None,
                       p: PlannerParams, c: CostParams) -> int:
    """Alternative controller: pick the action minimising cost under the perceived state.

    Kept as a robustness variant so the findings do not depend on the threshold rule.
    """
    return int(np.argmin(action_costs(a_req_perceived, prev_action, p, c)))


# Controller variants used as negative controls (section 18 of the plan).
PLANNERS = {
    "default": PlannerParams(),
    # deliberately insensitive: thresholds far apart, long standoff, no TTC term
    "insensitive": PlannerParams(thr_decel=3.0, thr_hard=7.5, standoff=0.5, use_ttc=False),
    "aggressive": PlannerParams(thr_decel=0.6, thr_hard=2.2),
    "wide_corridor": PlannerParams(corridor_half_w=2.0),
    "narrow_corridor": PlannerParams(corridor_half_w=0.9),
    "no_ttc": PlannerParams(use_ttc=False),
}

COSTS = {
    "default": CostParams(),
    "safety_heavy": CostParams(lam_brake=0.04, lam_collision=12.0),
    "comfort_heavy": CostParams(lam_brake=0.40, lam_collision=3.0),
    "no_collision_term": CostParams(lam_collision=0.0),
    "symmetric": CostParams(lam_brake=1.0, lam_collision=0.0),
}
