"""Planner B — an independent downstream decision maker, for the Phase 0F cross-planner check.

Phase 0E's standing weakness is that every decision cost in this project comes from one
hand-written controller: a threshold rule on required deceleration, scored by comparing the
commanded deceleration to the required one.  A reviewer can fairly answer the whole result
with "PKL was designed for a different planner; of course it does not predict your own cost
function".  Planner B exists to remove that answer, so it is deliberately different in all
three places where the objection bites:

  * **action space** — a coupled longitudinal/lateral trajectory drawn from a candidate set,
    not three discrete braking levels.
  * **mechanism** — receding-horizon rollout with argmin over candidates, not a threshold
    on a scalar requirement.  Nothing in it computes `required_decel`.
  * **cost** — the plan chosen under perceived geometry is *re-simulated against the true
    obstacle set* and scored by what happens in space-time: geometric overlap, worst
    clearance, progress actually made.  Planner A instead compares two accelerations.  There
    is no shortfall term here and no acceleration comparison anywhere.

The two planners share only what they must to be comparable at all: the same per-object
geometry (`long_near`, `lat_min`, `lat_max`, `ttc`), the same ego speed, and the same
one-code-path rule — `plan` is called identically on cheap, full and ground-truth geometry,
and ground truth enters only in the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# candidate trajectory set: longitudinal acceleration [m/s^2] x target lateral offset [m]
A_LON = np.array([0.0, -1.0, -2.0, -3.0, -4.5, -6.0, -8.0])
D_LAT = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])


@dataclass(frozen=True)
class PlannerBParams:
    horizon: float = 3.0             # rollout length [s]
    dt: float = 0.25                 # rollout step [s]
    tau_lat: float = 1.0             # first-order lag reaching the lateral target [s]
    ego_half_w: float = 0.9          # ego half-width for the overlap test [m]
    front_margin: float = 1.0        # longitudinal margin ahead of the ego origin [m]
    v_ref_floor: float = 2.0         # below this speed, progress is not rewarded [m/s]
    max_range: float = 80.0
    obstacle_closes: bool = True     # propagate obstacles at their estimated range rate


@dataclass(frozen=True)
class CostBParams:
    w_collision: float = 10.0        # geometric overlap during the rollout
    discount: float = 0.9            # per-step discount on collision (sooner is worse)
    w_clearance: float = 1.5         # squared shortfall below the desired clearance
    clear_req: float = 1.0           # desired worst-case lateral clearance [m]
    w_progress: float = 1.0          # normalised squared progress shortfall
    w_accel: float = 0.01            # |a_lon|^2 integrated over the horizon; the hardest
                                     # stop must stay well below the cost of a collision
    w_lateral: float = 0.50          # squared lateral excursion; a full 3 m swerve is a
                                     # real manoeuvre but still cheaper than crashing
    w_switch: float = 0.05           # change of plan between consecutive frames


def _rollout(v_ego: float, p: PlannerBParams):
    """Ego longitudinal distance, speed and lateral offset per (candidate, step)."""
    n = int(round(p.horizon / p.dt))
    t = (np.arange(1, n + 1) * p.dt)[None, :]                       # (1, n)
    a = A_LON[:, None]                                              # (7, 1)
    with np.errstate(divide="ignore"):
        t_stop = np.where(a < 0, np.maximum(v_ego, 0.0) / np.where(a < 0, -a, 1.0), np.inf)
    te = np.minimum(t, t_stop)                                      # motion freezes at stop
    s = np.maximum(v_ego, 0.0) * te + 0.5 * a * te ** 2             # (7, n)
    v = np.maximum(np.maximum(v_ego, 0.0) + a * te, 0.0)
    y = D_LAT[:, None] * (1.0 - np.exp(-t / p.tau_lat))             # (5, n)
    return s, v, y, t[0]


def _simulate(z, lo, hi, ttc, v_ego: float, p: PlannerBParams):
    """Per-candidate collision steps and worst lateral clearance against one obstacle set.

    Returns (collision cost weightable per step, min clearance) with candidates flattened
    in the order (a_lon, d_lat) = np.ndindex(len(A_LON), len(D_LAT)).
    """
    s, _, y, t = _rollout(v_ego, p)
    n = s.shape[1]
    na, nd = len(A_LON), len(D_LAT)

    z = np.asarray(z, float)
    keep = np.isfinite(z) & (z <= p.max_range)
    z, lo, hi = z[keep], np.asarray(lo, float)[keep], np.asarray(hi, float)[keep]
    ttc = np.asarray(ttc, float)[keep]
    if z.size == 0:
        return (np.zeros((na, nd, n), bool), np.full((na, nd), np.inf))

    # Obstacle motion must be expressed in the world, not relative to an ego that keeps its
    # current speed: `ttc` encodes the *relative* closing rate, so propagating the obstacle
    # by it would make braking unable to open the gap.  range_rate = v_obs - v_ego, hence
    # v_obs = v_ego + range_rate, and the gap is z_o + v_obs*t - s_ego(t).
    if p.obstacle_closes:
        rate = np.zeros_like(z)
        good = np.isfinite(ttc) & (ttc > 1e-3)
        rate[good] = -z[good] / ttc[good]                           # negative = closing
        v_obs = np.maximum(v_ego + rate, 0.0)                       # (N,) no reversing
    else:
        # "static" means stationary in the world, so the gap closes at the ego's own speed.
        # Leaving the *relative* rate at zero instead would make obstacles travel with the
        # ego, hold the gap constant, and remove every collision -- which is what the first
        # version of this branch did, and it showed up as exactly 0.0% plan changes.
        v_obs = np.zeros_like(z)
    zt = z[None, :] + v_obs[None, :] * t[:, None]                   # (n, N) world position

    gap = zt[None, None, :, :] - s[:, None, :, None]                # (na,1,n,N)
    reach = gap <= p.front_margin                                   # ego has closed the gap
    lat_gap = np.minimum(hi[None, None, :] - (y[:, :, None] - p.ego_half_w),
                         (y[:, :, None] + p.ego_half_w) - lo[None, None, :])  # (nd,n,N)
    overlap = lat_gap > 0.0
    hit = reach & overlap[None, :, :, :]                            # (na,nd,n,N)

    coll = hit.any(axis=3)                                          # (na,nd,n)
    # clearance only counts where the ego actually reaches the obstacle longitudinally
    relevant = np.broadcast_to(reach, hit.shape)
    gaps = np.where(relevant, np.broadcast_to(-lat_gap[None], hit.shape), np.inf)
    clear = gaps.reshape(na, nd, -1).min(axis=2)
    return coll, clear


def _candidate_costs(z, lo, hi, ttc, v_ego: float, prev: int | None,
                     p: PlannerBParams, c: CostBParams) -> np.ndarray:
    coll, clear = _simulate(z, lo, hi, ttc, v_ego, p)
    n = coll.shape[2]
    # discount the *earliest* collision rather than summing over steps: once the ego is
    # inside an obstacle every later step also overlaps, so a sum would grow with how long
    # the rollout keeps running and swamp every other term.
    disc = c.discount ** np.arange(n)
    J = c.w_collision * (coll * disc[None, None, :]).max(axis=2)

    # clearance shortfall is bounded: once the ego is well inside an obstacle the collision
    # term already fires, and an unbounded depth-of-overlap penalty would dominate it
    cl = np.where(np.isfinite(clear), clear, c.clear_req)
    short = np.clip(c.clear_req - cl, 0.0, 2.0 * c.clear_req)
    J = J + c.w_clearance * short ** 2

    s, _, _, _ = _rollout(v_ego, p)
    v_ref = max(v_ego, p.v_ref_floor)
    ref = v_ref * p.horizon
    J = J + c.w_progress * ((ref - s[:, -1]) / ref)[:, None] ** 2
    J = J + c.w_accel * (A_LON ** 2 * p.horizon)[:, None]
    J = J + c.w_lateral * (D_LAT ** 2)[None, :]
    if prev is not None:
        pa, pd_ = divmod(int(prev), len(D_LAT))
        J = J + c.w_switch * ((A_LON[:, None] - A_LON[pa]) ** 2 / 4.0
                              + (D_LAT[None, :] - D_LAT[pd_]) ** 2)
    return J


def plan(z, lo, hi, ttc, v_ego: float, prev: int | None = None,
         p: PlannerBParams | None = None, c: CostBParams | None = None) -> int:
    """Chosen candidate index, from perceived geometry only.  Never sees ground truth."""
    p, c = p or PlannerBParams(), c or CostBParams()
    J = _candidate_costs(z, lo, hi, ttc, v_ego, prev, p, c)
    return int(np.argmin(J))


def executed_cost(cand: int, z_true, lo_true, hi_true, ttc_true, v_ego: float,
                  prev: int | None = None, p: PlannerBParams | None = None,
                  c: CostBParams | None = None) -> dict:
    """What the chosen plan actually costs when run against the true obstacle set.

    This is the evaluator, and it is where ground truth enters.  It is not the planner's
    own objective re-used: the collision term is the *occurrence* of overlap in the true
    scene rather than a discounted sum, because what matters downstream is whether the
    manoeuvre hit something, not how the planner scored it.
    """
    p, c = p or PlannerBParams(), c or CostBParams()
    a_i, d_i = divmod(int(cand), len(D_LAT))
    coll, clear = _simulate(z_true, lo_true, hi_true, ttc_true, v_ego, p)
    hit = bool(coll[a_i, d_i].any())
    cl = float(clear[a_i, d_i])
    short = float(np.clip(c.clear_req - (cl if np.isfinite(cl) else c.clear_req),
                          0.0, 2.0 * c.clear_req))

    s, _, _, _ = _rollout(v_ego, p)
    v_ref = max(v_ego, p.v_ref_floor)
    ref = v_ref * p.horizon
    progress = ((ref - s[a_i, -1]) / ref) ** 2

    J = (c.w_collision * hit + c.w_clearance * short ** 2 + c.w_progress * progress
         + c.w_accel * A_LON[a_i] ** 2 * p.horizon + c.w_lateral * D_LAT[d_i] ** 2)
    if prev is not None:
        pa, pd_ = divmod(int(prev), len(D_LAT))
        J += c.w_switch * ((A_LON[a_i] - A_LON[pa]) ** 2 / 4.0
                           + (D_LAT[d_i] - D_LAT[pd_]) ** 2)
    return {"J": float(J), "collision": float(hit), "min_clearance": cl,
            "progress_shortfall": float(progress),
            "a_lon": float(A_LON[a_i]), "d_lat": float(D_LAT[d_i])}


def action_name(cand: int) -> str:
    a_i, d_i = divmod(int(cand), len(D_LAT))
    return f"a{A_LON[a_i]:+.1f}/y{D_LAT[d_i]:+.1f}"


# sensitivity variants, so the cross-planner conclusion does not rest on one weighting
PARAMS_B = {
    "default": PlannerBParams(),
    "short_horizon": PlannerBParams(horizon=2.0),
    "long_horizon": PlannerBParams(horizon=4.0),
    "narrow_ego": PlannerBParams(ego_half_w=0.75),
    "static_obstacles": PlannerBParams(obstacle_closes=False),
}
COSTS_B = {
    "default": CostBParams(),
    "safety_heavy": CostBParams(w_collision=20.0, w_progress=0.5),
    "progress_heavy": CostBParams(w_collision=5.0, w_progress=2.0),
    "no_clearance": CostBParams(w_clearance=0.0),
}
