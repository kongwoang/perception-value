"""Several frame-level definitions of "perception improvement".

The central claim must not depend on one arbitrary perception loss, so this computes a
family of them per frame. Dataset-level AP is deliberately absent: it is not defined for a
single frame, and using it per frame would be a different quantity wearing its name.

Every loss is a *cost* (lower is better), so dE_k = E_k(CHEAP) - E_k(FULL) is positive
when FULL is better, matching the sign convention used throughout.
"""
from __future__ import annotations

import numpy as np

from .risk import RiskConfig, _ioa, match


def frame_losses(gt_xyxy, gt_cls, gt_crit, det, cfg: RiskConfig,
                 dontcare=None, op_conf: float | None = None) -> dict:
    """All perception losses for one frame under one mode, at that mode's operating threshold."""
    keep = det["conf"] >= (cfg.op_conf if op_conf is None else op_conf)
    dx, dc = det["xyxy"][keep].astype(np.float64), det["conf"][keep]
    dcls = det["coarse"][keep]

    best_iou, matched = match(gt_xyxy, gt_cls, dx, dc, dcls, cfg)
    hit = best_iou >= cfg.iou_thr
    fn = float((~hit).sum())

    unmatched = ~matched
    if dontcare is not None and len(dontcare) and len(dx):
        unmatched &= _ioa(dx, dontcare).max(axis=1) < cfg.dontcare_ioa
    fp = float(unmatched.sum())

    # localisation error over matched objects only, so it is not a disguised miss count
    loc = float(np.sum(1.0 - best_iou[hit])) if hit.any() else 0.0
    # class error: a matched object whose coarse class disagrees with GT
    cls_err = 0.0
    if len(dx) and len(gt_xyxy):
        iou_full = _pair_iou(dx, gt_xyxy)
        for gi in np.flatnonzero(hit):
            di = int(np.argmax(iou_full[:, gi]))
            cls_err += float(dcls[di] != gt_cls[gi])

    crit_fn = float(np.sum(gt_crit[~hit])) if len(gt_crit) else 0.0
    return {"fn": fn, "fp": fp, "loc": loc, "cls": cls_err, "crit_fn": crit_fn,
            "n_gt": float(len(gt_xyxy)), "n_det": float(len(dx)),
            "mean_iou_matched": float(best_iou[hit].mean()) if hit.any() else 0.0,
            "conf_sum": float(dc.sum())}


def _pair_iou(a, b):
    from .mono import box_iou
    return box_iou(a, b)


# Named perception losses, each a weighted combination of the primitives above.
# E6 exists only to compare against Phase 0; it is not the headline metric.
METRICS = {
    "E1_fn_only":      {"fn": 1.0},
    "E2_fn_fp":        {"fn": 1.0, "fp": 1.0},
    "E3_class_aware":  {"fn": 1.0, "cls": 1.0},
    "E4_localization": {"fn": 1.0, "loc": 1.0},
    "E5_combined":     {"fn": 1.0, "fp": 0.5, "loc": 1.0, "cls": 0.5},
    "E5_fp_heavy":     {"fn": 1.0, "fp": 2.0, "loc": 1.0, "cls": 0.5},
    "E5_loc_heavy":    {"fn": 1.0, "fp": 0.5, "loc": 3.0, "cls": 0.5},
    "E6_risk_weighted": {"crit_fn": 1.0},
}


def combine(prim: dict, weights: dict) -> float:
    return float(sum(w * prim.get(k, 0.0) for k, w in weights.items()))
