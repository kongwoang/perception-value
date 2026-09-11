"""Temporal replay evaluation: a stateful controller run along whole sequences.

**This is replay, not closed loop.** The ego trajectory is the logged one, so an action
taken at frame t does not change what the camera sees at t+1. What it does capture, and
per-frame scoring cannot, is the *cost of a decision history*: hysteresis, commitment,
switching, and repeated or sustained errors. Anything stronger needs a simulator.

The controller state is deliberately small and interpretable: the previous command, a
braking latch with hysteresis, and a lateral commitment counter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import planner as P


@dataclass(frozen=True)
class TemporalParams:
    brake_hysteresis: float = 0.5   # a_req must fall this far below the threshold to release
    lat_commit_frames: int = 3      # hold a lateral manoeuvre this many frames
    lam_switch: float = 0.25        # per change of longitudinal command
    lam_lat_switch: float = 0.40    # per change of corridor
    lam_sustained: float = 0.60     # per frame of a run of >=3 consecutive under-brakes
    lam_repeat_brake: float = 0.30  # per unnecessary brake beyond the first in a run
    sustained_run: int = 3


def replay_longitudinal(a_req_seq, a_gt_seq, pp: P.PlannerParams, cp: P.CostParams,
                        tp: TemporalParams) -> dict:
    """Run the braking controller along a sequence with hysteresis, and score the history."""
    n = len(a_req_seq)
    acts = np.zeros(n, dtype=int)
    prev = P.KEEP
    for t in range(n):
        a = float(a_req_seq[t])
        # latch: having committed to a level, require a margin before stepping down
        raw = P.discrete_action(a, pp)
        if raw < prev:
            thr = pp.thr_hard if prev == P.HARD_BRAKE else pp.thr_decel
            if a > thr - tp.brake_hysteresis:
                raw = prev
        acts[t] = raw
        prev = raw

    per = [P.decision_cost(int(acts[t]), float(a_gt_seq[t]),
                           int(acts[t - 1]) if t else None, pp, cp) for t in range(n)]
    J = float(sum(p["J"] for p in per))
    switches = float(np.sum(acts[1:] != acts[:-1]))

    shortfall = np.array([p["shortfall"] for p in per])
    under = shortfall > 0.5
    sustained = _run_frames(under, tp.sustained_run)
    unnecessary = np.array([p["excess"] for p in per]) > 1.0
    repeats = max(0.0, _run_total(unnecessary) - _run_count(unnecessary))

    J_seq = (J + tp.lam_switch * switches + tp.lam_sustained * sustained
             + tp.lam_repeat_brake * repeats)
    return {"J_seq": J_seq, "J_frames": J, "switches": switches,
            "sustained_under": sustained, "repeat_brakes": repeats,
            "collisions": float(sum(p["collision"] for p in per)), "actions": acts}


def replay_lateral(geom_seq, v_seq, gt_seq, lp: P.LateralParams, lc: P.LateralCostParams,
                   tp: TemporalParams) -> dict:
    """Run the corridor controller along a sequence with a commitment counter."""
    n = len(v_seq)
    acts, offs = np.zeros(n, dtype=int), np.zeros(n)
    commit = 0
    prev_a, prev_o = P.LAT_KEEP, 0.0
    for t in range(n):
        z, lo, hi = geom_seq[t]
        a, o = P.lateral_action(z, lo, hi, float(v_seq[t]), lp)
        if commit > 0 and prev_a in (P.LAT_LEFT, P.LAT_RIGHT):
            a, o = prev_a, prev_o          # hold the manoeuvre
            commit -= 1
        elif a in (P.LAT_LEFT, P.LAT_RIGHT):
            commit = tp.lat_commit_frames
        acts[t], offs[t] = a, o
        prev_a, prev_o = a, o

    per = [P.lateral_cost(int(acts[t]), float(offs[t]), *gt_seq[t], float(v_seq[t]),
                          int(acts[t - 1]) if t else None, lp, lc) for t in range(n)]
    J = float(sum(p["J"] for p in per))
    switches = float(np.sum(acts[1:] != acts[:-1]))
    unsafe = np.array([p["collision"] for p in per]) > 0
    J_seq = J + tp.lam_lat_switch * switches + tp.lam_sustained * _run_frames(unsafe, tp.sustained_run)
    return {"J_seq": J_seq, "J_frames": J, "switches": switches,
            "collisions": float(unsafe.sum()), "actions": acts}


def _runs(mask: np.ndarray):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _run_frames(mask, min_len: int) -> float:
    """Frames belonging to a run of at least `min_len` consecutive True."""
    return float(sum(b - a for a, b in _runs(mask) if b - a >= min_len))


def _run_count(mask) -> float:
    return float(len(_runs(mask)))


def _run_total(mask) -> float:
    return float(np.sum(mask))
