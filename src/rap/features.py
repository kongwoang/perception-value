"""Frame-level features a *deployable* gate could compute.

Every feature is derived from three things and nothing else:
  * the CHEAP detections of the current frame,
  * the CHEAP detections and downscaled image of the *previous* frame,
  * the camera calibration.

The FULL prediction, the ground truth and any future frame are out of reach by
construction. `assert_no_leakage` enforces this against the registry rather than
trusting the caller, because the entire result of Phase 0 is worthless if a
criticality feature quietly carries oracle information.

Groups map onto the experimental arms:
    conf     -> A   confidence
    unc      -> B   visual uncertainty      (A + entropy/margin/NMS ambiguity)
    complex  -> C   scene complexity
    crit     -> D   downstream criticality, estimated monocularly
"""
from __future__ import annotations

import numpy as np

from .geometry import CriticalityModel
from .mono import box_iou

LEGAL_SOURCES = {"cheap_det", "cheap_image", "cheap_prev", "calib"}

# arm -> feature groups it may use
ARMS = {
    "A_conf": ("conf",),
    "B_uncertainty": ("conf", "unc"),
    "C_complexity": ("complex",),
    "D_criticality": ("crit",),
    "E_unc_crit": ("conf", "unc", "crit"),
    "F_all_visual": ("conf", "unc", "complex"),
    "G_all": ("conf", "unc", "complex", "crit"),
}

# Value_task is, roughly, stakes x failure-propensity: a frame is worth more compute
# when more criticality is present AND cheap perception is likely to fail on it.
# `feat_crit_sum` alone carries the stakes. This arm exists so the report can say
# whether criticality *structure* (corridor, TTC, risk-weighted uncertainty) adds
# anything beyond that single scalar.
STAKES_COLS = ("feat_crit_sum",)
ARM_EXTRA_COLS = {"H_visual_plus_stakes": STAKES_COLS}
ARMS["H_visual_plus_stakes"] = ("conf", "unc", "complex")

_REGISTRY: dict[str, tuple[str, str]] = {}


def _reg(name: str, group: str, source: str) -> str:
    assert source in LEGAL_SOURCES, source
    _REGISTRY[name] = (group, source)
    return name


def registry() -> dict[str, tuple[str, str]]:
    return dict(_REGISTRY)


def assert_no_leakage(columns) -> None:
    """Every column offered to a predictor must be a registered, legal feature."""
    unknown = [c for c in columns if c not in _REGISTRY]
    if unknown:
        raise ValueError(f"unregistered feature columns (possible leakage): {unknown}")
    bad = [c for c in columns if _REGISTRY[c][1] not in LEGAL_SOURCES]
    if bad:
        raise ValueError(f"features with illegal provenance: {bad}")


def columns_for(arm: str) -> list[str]:
    groups = ARMS[arm]
    cols = [c for c, (g, _) in _REGISTRY.items() if g in groups]
    cols += [c for c in ARM_EXTRA_COLS.get(arm, ()) if c not in cols]
    return cols


# ----------------------------------------------------------------------------- helpers

def _safe(fn, arr, default=0.0):
    return float(fn(arr)) if len(arr) else float(default)


def _q(arr, q, default=0.0):
    return float(np.percentile(arr, q)) if len(arr) else float(default)


# ----------------------------------------------------------------------------- extractors

def confidence_features(conf: np.ndarray) -> dict:
    f = {
        _reg("feat_conf_max", "conf", "cheap_det"): _safe(np.max, conf),
        _reg("feat_conf_mean", "conf", "cheap_det"): _safe(np.mean, conf),
        _reg("feat_conf_min", "conf", "cheap_det"): _safe(np.min, conf, 1.0),
        _reg("feat_conf_std", "conf", "cheap_det"): _safe(np.std, conf),
        _reg("feat_conf_p25", "conf", "cheap_det"): _q(conf, 25),
        _reg("feat_conf_median", "conf", "cheap_det"): _q(conf, 50),
        _reg("feat_conf_inv_sum", "conf", "cheap_det"): float(np.sum(1.0 - conf)),
        _reg("feat_conf_n_low", "conf", "cheap_det"): float(np.sum(conf < 0.4)),
        _reg("feat_conf_frac_low", "conf", "cheap_det"): _safe(lambda a: np.mean(a < 0.4), conf),
        _reg("feat_conf_none", "conf", "cheap_det"): float(len(conf) == 0),
    }
    return f


def uncertainty_features(det: dict, keep: np.ndarray, op_conf: float) -> dict:
    ent, marg, bent = det["entropy"][keep], det["margin"][keep], det["binent"][keep]
    below = det["conf"][~keep]
    n_post = max(int(keep.sum()), 1)
    return {
        _reg("feat_ent_mean", "unc", "cheap_det"): _safe(np.mean, ent),
        _reg("feat_ent_max", "unc", "cheap_det"): _safe(np.max, ent),
        _reg("feat_ent_sum", "unc", "cheap_det"): float(np.sum(ent)),
        _reg("feat_margin_mean", "unc", "cheap_det"): _safe(np.mean, marg, 1.0),
        _reg("feat_margin_min", "unc", "cheap_det"): _safe(np.min, marg, 1.0),
        _reg("feat_binent_mean", "unc", "cheap_det"): _safe(np.mean, bent),
        _reg("feat_binent_max", "unc", "cheap_det"): _safe(np.max, bent),
        _reg("feat_binent_sum", "unc", "cheap_det"): float(np.sum(bent)),
        # how much the raw detector output had to be pruned to produce this frame
        _reg("feat_nms_ratio", "unc", "cheap_det"): float(det["n_cand"]) / n_post,
        _reg("feat_n_cand", "unc", "cheap_det"): float(det["n_cand"]),
        _reg("feat_n_cand_raw", "unc", "cheap_det"): float(det["n_cand_raw"]),
        # detections that nearly fired: the frame sits on a decision knife-edge
        _reg("feat_n_borderline", "unc", "cheap_det"): float(len(below)),
        _reg("feat_borderline_max", "unc", "cheap_det"): _safe(np.max, below),
        _reg("feat_borderline_gap", "unc", "cheap_det"): op_conf - _safe(np.max, below),
    }


def complexity_features(det: dict, keep: np.ndarray, img_stats: dict,
                        img_area: float) -> dict:
    xyxy = det["xyxy"][keep].astype(np.float64)
    w = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) if len(xyxy) else np.zeros(0)
    h = np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None) if len(xyxy) else np.zeros(0)
    area = w * h
    if len(xyxy) > 1:
        iou = box_iou(xyxy, xyxy)
        np.fill_diagonal(iou, 0.0)
        overlap_pairs = float((iou > 0.3).sum() / 2)
        overlap_max = float(iou.max())
    else:
        overlap_pairs, overlap_max = 0.0, 0.0
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 if len(xyxy) else np.zeros(0)
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 if len(xyxy) else np.zeros(0)
    f = {
        _reg("feat_n_det", "complex", "cheap_det"): float(len(xyxy)),
        _reg("feat_area_frac_sum", "complex", "cheap_det"): float(area.sum() / img_area),
        _reg("feat_area_frac_mean", "complex", "cheap_det"): _safe(np.mean, area / img_area),
        _reg("feat_area_frac_max", "complex", "cheap_det"): _safe(np.max, area / img_area),
        _reg("feat_area_frac_std", "complex", "cheap_det"): _safe(np.std, area / img_area),
        _reg("feat_h_min_px", "complex", "cheap_det"): _safe(np.min, h, 0.0),
        _reg("feat_h_median_px", "complex", "cheap_det"): _q(h, 50),
        _reg("feat_n_small", "complex", "cheap_det"): float(np.sum(h < 40)),
        _reg("feat_overlap_pairs", "complex", "cheap_det"): overlap_pairs,
        _reg("feat_overlap_max", "complex", "cheap_det"): overlap_max,
        _reg("feat_spread_x", "complex", "cheap_det"): _safe(np.std, cx),
        _reg("feat_spread_y", "complex", "cheap_det"): _safe(np.std, cy),
    }
    for k, v in img_stats.items():
        src = "cheap_prev" if k.startswith("motion") else "cheap_image"
        f[_reg(f"feat_{k}", "complex", src)] = float(v)
    return f


def criticality_features(det: dict, keep: np.ndarray, geo: dict,
                         model: CriticalityModel) -> dict:
    """Frame aggregates of per-detection *estimated* criticality."""
    conf = det["conf"][keep]
    bent = det["binent"][keep]
    z = geo["z"][keep]
    lat_min, lat_max, ttc = geo["lat_min"][keep], geo["lat_max"][keep], geo["ttc"][keep]
    box_h = geo["box_h"][keep]
    crit = model(z, lat_min, lat_max, ttc) if len(z) else np.zeros(0)

    hw = model.corridor_half_w
    in_corr = (lat_min < hw) & (lat_max > -hw) if len(z) else np.zeros(0, bool)

    # below-threshold detections: criticality of what the cheap model almost saw
    bkeep = ~keep
    zb = geo["z"][bkeep]
    crit_b = (model(zb, geo["lat_min"][bkeep], geo["lat_max"][bkeep], geo["ttc"][bkeep])
              if len(zb) else np.zeros(0))

    csum = float(crit.sum())
    return {
        _reg("feat_crit_sum", "crit", "cheap_det"): csum,
        _reg("feat_crit_max", "crit", "cheap_det"): _safe(np.max, crit),
        _reg("feat_crit_mean", "crit", "cheap_det"): _safe(np.mean, crit),
        _reg("feat_crit_p90", "crit", "cheap_det"): _q(crit, 90),
        _reg("feat_n_in_corridor", "crit", "cheap_det"): float(in_corr.sum()),
        _reg("feat_crit_sum_corridor", "crit", "cheap_det"): float(crit[in_corr].sum()) if len(z) else 0.0,
        _reg("feat_z_min", "crit", "cheap_det"): _safe(np.min, z, 200.0),
        _reg("feat_z_min_corridor", "crit", "cheap_det"): _safe(np.min, z[in_corr], 200.0) if len(z) else 200.0,
        _reg("feat_z_p25", "crit", "cheap_det"): _q(z, 25, 200.0),
        _reg("feat_lat_abs_min", "crit", "cheap_det"): (
            float(np.min(np.maximum(0.0, np.maximum(-lat_max, lat_min)))) if len(z) else 50.0),
        _reg("feat_ttc_min", "crit", "cheap_prev"): float(np.min(np.minimum(ttc, 1e3))) if len(z) else 1e3,
        _reg("feat_n_ttc_lt4", "crit", "cheap_prev"): float(np.sum(ttc < 4.0)),
        _reg("feat_n_ttc_lt2", "crit", "cheap_prev"): float(np.sum(ttc < 2.0)),
        # risk-weighted uncertainty: uncertainty that lands where it matters
        _reg("feat_riskw_unc_sum", "crit", "cheap_det"): float(np.sum(crit * (1.0 - conf))),
        _reg("feat_riskw_unc_mean", "crit", "cheap_det"): float(
            np.sum(crit * (1.0 - conf)) / csum) if csum > 1e-6 else 0.0,
        _reg("feat_riskw_ent_sum", "crit", "cheap_det"): float(np.sum(crit * bent)),
        # critical *and* small: exactly what more resolution would help with
        _reg("feat_crit_small_sum", "crit", "cheap_det"): float(crit[box_h < 40].sum()) if len(z) else 0.0,
        _reg("feat_crit_borderline_sum", "crit", "cheap_det"): float(crit_b.sum()),
        _reg("feat_crit_borderline_max", "crit", "cheap_det"): _safe(np.max, crit_b),
    }


def frame_features(det: dict, geo: dict, img_stats: dict, img_area: float,
                   model: CriticalityModel, op_conf: float) -> dict:
    keep = det["conf"] >= op_conf
    f = {}
    f.update(confidence_features(det["conf"][keep]))
    f.update(uncertainty_features(det, keep, op_conf))
    f.update(complexity_features(det, keep, img_stats, img_area))
    f.update(criticality_features(det, keep, geo, model))
    return f


def image_stats(small_gray: np.ndarray, prev_small_gray: np.ndarray | None) -> dict:
    """Statistics of the downscaled frame the cheap network already consumes."""
    import cv2
    g = small_gray.astype(np.float32)
    lap = cv2.Laplacian(small_gray, cv2.CV_32F)
    sx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    stats = {
        "img_bright_mean": g.mean(), "img_bright_std": g.std(),
        "img_dark_frac": float((g < 40).mean()), "img_bright_frac": float((g > 215).mean()),
        "img_lap_var": float(lap.var()), "img_edge_density": float((mag > 60).mean()),
        "img_contrast_p95_p5": float(np.percentile(g, 95) - np.percentile(g, 5)),
    }
    if prev_small_gray is None:
        stats.update({"motion_mean": 0.0, "motion_p95": 0.0, "motion_first": 1.0})
    else:
        d = np.abs(g - prev_small_gray.astype(np.float32))
        stats.update({"motion_mean": float(d.mean()),
                      "motion_p95": float(np.percentile(d, 95)), "motion_first": 0.0})
    return stats


def _prime_registry() -> None:
    """Populate the registry at import time.

    Provenance is declared next to each computation, which keeps the two from
    drifting apart — but that means the registry would otherwise only exist after
    features had been computed, and the leakage guard has to work on a table loaded
    from disk. Running the extractors once over an empty frame fixes both.
    """
    import numpy as _np
    empty_f = _np.zeros(0, _np.float32)
    det = {"xyxy": _np.zeros((0, 4), _np.float32), "conf": empty_f,
           "coarse": _np.array([], dtype="U8"), "entropy": empty_f,
           "margin": empty_f, "binent": empty_f,
           "n_cand": 0.0, "n_cand_raw": 0.0, "n_post": 0.0}
    geo = {k: empty_f for k in ("z", "z_ground", "z_height", "lat_min", "lat_max",
                                "ttc", "box_h")}
    stats = {"img_bright_mean": 0.0, "img_bright_std": 0.0, "img_dark_frac": 0.0,
             "img_bright_frac": 0.0, "img_lap_var": 0.0, "img_edge_density": 0.0,
             "img_contrast_p95_p5": 0.0, "motion_mean": 0.0, "motion_p95": 0.0,
             "motion_first": 1.0}
    frame_features(det, geo, stats, 1.0, CriticalityModel("_prime"), 0.25)


_prime_registry()
