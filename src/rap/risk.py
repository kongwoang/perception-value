"""Risk-weighted perception error, and the two notions of "value of compute".

    Risk(t, m)        = sum_i criticality(i,t) * perception_error(i,t,m)  [+ FP term]
    Value_task(t)     = Risk(t, CHEAP) - Risk(t, FULL)
    Value_visual(t)   = Error_std(t, CHEAP) - Error_std(t, FULL)

Error_std uses the identical error function with every criticality set to 1, so the
two differ *only* by the downstream weighting — which is exactly the quantity under
test.

This module is an oracle: it consumes ground truth freely. Nothing here may be
reached from the feature path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mono import box_iou


@dataclass(frozen=True)
class RiskConfig:
    iou_thr: float = 0.5
    op_conf: float = 0.25            # operating confidence threshold (CHEAP)
    op_conf_full: float | None = None  # if set, FULL uses its own threshold
    error: str = "miss"              # miss | soft_iou
    fp_lambda: float = 0.0           # weight on risk-weighted false positives
    class_aware: bool = False        # class-agnostic matching by default
    min_gt_height: float = 10.0      # GT boxes below this are unmeasurable
    dontcare_ioa: float = 0.5        # detection is ignored if this much of it is DontCare

    def thr(self, role: str) -> float:
        """Operating threshold of one fidelity: CHEAP uses op_conf, FULL op_conf_full when set.

        Every place that filters detections goes through here, so a per-mode operating point
        reaches detection filtering, the monocular lift, oracle-geometry matching, the controllers,
        the planners and the perception losses alike, and op_conf_full=None is the shared-threshold
        pipeline exactly.
        """
        if role not in ("cheap", "full"):
            raise ValueError(role)
        return self.op_conf_full if (role == "full" and self.op_conf_full is not None) else self.op_conf


def match(gt_xyxy, gt_cls, det_xyxy, det_conf, det_cls, cfg: RiskConfig):
    """Greedy confidence-ordered matching. Returns (best_iou_per_gt, matched_det_mask)."""
    n_gt, n_det = len(gt_xyxy), len(det_xyxy)
    best_iou = np.zeros(n_gt)
    det_matched = np.zeros(n_det, dtype=bool)
    if n_gt == 0 or n_det == 0:
        return best_iou, det_matched

    iou = box_iou(det_xyxy, gt_xyxy)
    if cfg.class_aware:
        iou = np.where(det_cls[:, None] == gt_cls[None, :], iou, 0.0)

    gt_taken = np.zeros(n_gt, dtype=bool)
    for di in np.argsort(-det_conf):
        row = np.where(gt_taken, -1.0, iou[di])
        gi = int(np.argmax(row))
        if row[gi] >= cfg.iou_thr:
            gt_taken[gi] = True
            det_matched[di] = True
            best_iou[gi] = row[gi]
    # un-matched GT still record their best overlap, for the soft error variant
    for gi in np.flatnonzero(~gt_taken):
        best_iou[gi] = iou[:, gi].max() if n_det else 0.0
    return best_iou, det_matched


def per_object_error(best_iou: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    if cfg.error == "miss":
        return (best_iou < cfg.iou_thr).astype(np.float64)
    if cfg.error == "soft_iou":
        return np.clip(1.0 - best_iou, 0.0, 1.0)
    raise ValueError(cfg.error)


def frame_risk(gt: dict, det: dict, crit: np.ndarray, cfg: RiskConfig,
               dontcare: np.ndarray | None = None,
               fp_crit: np.ndarray | None = None,
               op_conf: float | None = None) -> dict:
    """Risk of one frame under one perception mode.

    `gt` supplies xyxy/cls for evaluable objects, `crit` their criticality.
    `fp_crit` is the criticality *estimated from the detection itself* (there is no
    GT for a false positive), used only when cfg.fp_lambda > 0.
    """
    keep = det["conf"] >= (cfg.op_conf if op_conf is None else op_conf)
    det_xyxy, det_conf = det["xyxy"][keep], det["conf"][keep]
    det_cls = det["coarse"][keep]

    best_iou, det_matched = match(gt["xyxy"], gt["cls"], det_xyxy, det_conf, det_cls, cfg)
    err = per_object_error(best_iou, cfg)

    risk = float(np.sum(crit * err))
    err_std = float(np.sum(err))

    fp_risk = 0.0
    n_fp = 0
    if len(det_xyxy):
        unmatched = ~det_matched
        if dontcare is not None and len(dontcare):
            inter = _ioa(det_xyxy, dontcare)
            unmatched &= inter.max(axis=1) < cfg.dontcare_ioa
        n_fp = int(unmatched.sum())
        if cfg.fp_lambda > 0 and fp_crit is not None and n_fp:
            fc = fp_crit[keep][unmatched]
            fp_risk = float(cfg.fp_lambda * np.sum(fc * det_conf[unmatched]))

    return {
        "risk": risk + fp_risk, "risk_miss": risk, "risk_fp": fp_risk,
        "err_std": err_std + cfg.fp_lambda * n_fp, "err_std_miss": err_std,
        "n_gt": len(err), "n_miss": float(err.sum()), "n_det": int(keep.sum()), "n_fp": n_fp,
        "crit_sum": float(crit.sum()),
    }


def _ioa(det_xyxy: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Intersection over *detection* area — KITTI's DontCare ignore rule."""
    if len(det_xyxy) == 0 or len(other) == 0:
        return np.zeros((len(det_xyxy), max(len(other), 1)))
    lt = np.maximum(det_xyxy[:, None, :2], other[None, :, :2])
    rb = np.minimum(det_xyxy[:, None, 2:], other[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = np.clip(det_xyxy[:, 2] - det_xyxy[:, 0], 0, None) * \
           np.clip(det_xyxy[:, 3] - det_xyxy[:, 1], 0, None)
    return inter / np.maximum(area[:, None], 1e-9)
