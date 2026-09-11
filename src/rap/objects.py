"""Object-level (candidate-anchored) table: does extra compute fix *this* object?

Phase 0 scored frames. Phase 0B needs per-object labels, because the hypothesis under
test decomposes a frame's value into

    Value(t) = sum_i  criticality(i) . P(cheap fails on i) . P(full recovers i | fails)

and the two probabilities are different questions that a frame-level regressor cannot
separate.

The rows are anchored on **CHEAP candidates**, not on ground-truth objects, because that
is what a deployed gate actually has. Candidates are kept down to the detector's cache
floor (0.10), well below the operating threshold: a measurement on this data shows that
most ground-truth objects extra compute recovers appear in CHEAP output only as
sub-threshold candidates, so an anchor restricted to accepted detections would be blind
to them.

Ground truth objects with no overlapping candidate at all are invisible to this anchor.
That is a real ceiling, it is measured (`anchor_coverage`), and it is reported rather
than hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as G
from . import kitti
from .cache import DetCache
from .mono import box_iou
from .risk import RiskConfig, match

# Candidate features are the deployable ones; provenance is checked the same way frame
# features are, against this registry.
#
# The registry is declared here rather than filled in as a side effect of building a
# table: the leakage guard has to work on a table loaded from disk, and a registry that
# only exists after the first build silently passes on an empty column list.
OBJ_FEATURES: dict[str, str] = {
    # detector uncertainty about this candidate
    "o_conf": "unc", "o_logit": "unc", "o_entropy": "unc", "o_margin": "unc",
    "o_binent": "unc", "o_below_thr": "unc", "o_conf_gap_to_thr": "unc",
    # apparent size and image position -- what resolution actually buys
    "o_h_px": "size", "o_w_px": "size", "o_area_frac": "size", "o_log_h": "size",
    "o_aspect": "size", "o_cx_frac": "size", "o_cy_frac": "size",
    "o_bottom_below_horizon": "size", "o_touches_edge": "size",
    # monocular ego geometry -> estimated downstream criticality
    "o_z": "crit", "o_z_ground": "crit", "o_z_height": "crit", "o_lat_abs": "crit",
    "o_ttc": "crit", "o_ttc_inv": "crit", "o_c_hat": "crit",
    "o_is_person": "crit", "o_is_cyclist": "crit",
    # frame and neighbourhood context
    "o_n_overlap": "ctx", "o_max_overlap": "ctx", "o_n_cand_frame": "ctx",
    "o_conf_rank": "ctx", "o_frame_conf_mean": "ctx", "o_frame_conf_max": "ctx",
    "o_motion_mean": "ctx", "o_img_bright_mean": "ctx", "o_img_lap_var": "ctx",
    "o_prev_h": "ctx",
}


def _f(name: str, group: str) -> str:
    """Assert, rather than register: the declaration above is the single source."""
    assert OBJ_FEATURES.get(name) == group, f"{name!r} not declared in OBJ_FEATURES as {group!r}"
    return name


def assert_no_object_leakage(cols) -> None:
    unknown = [c for c in cols if c not in OBJ_FEATURES]
    if unknown:
        raise ValueError(f"unregistered object-level columns (possible leakage): {unknown}")


def object_columns(groups=None) -> list[str]:
    return [c for c, g in OBJ_FEATURES.items() if groups is None or g in groups]


def _assign_gt_to_candidates(gt_xyxy: np.ndarray, cand_xyxy: np.ndarray,
                             cand_conf: np.ndarray, min_iou: float):
    """One candidate per GT object, at most one GT per candidate.

    Without an exclusive assignment several candidates would carry the same object's
    value and the per-frame sum would double-count it.
    """
    n_gt, n_c = len(gt_xyxy), len(cand_xyxy)
    gt_of_cand = np.full(n_c, -1, dtype=int)
    cand_of_gt = np.full(n_gt, -1, dtype=int)
    if n_gt == 0 or n_c == 0:
        return gt_of_cand, cand_of_gt, np.zeros(n_c)
    iou = box_iou(cand_xyxy, gt_xyxy)
    best_iou_per_cand = iou.max(axis=1) if n_gt else np.zeros(n_c)
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for ci, gi in order:
        if iou[ci, gi] < min_iou:
            break
        if gt_of_cand[ci] == -1 and cand_of_gt[gi] == -1:
            gt_of_cand[ci] = gi
            cand_of_gt[gi] = ci
    return gt_of_cand, cand_of_gt, best_iou_per_cand


def build_sequence(seq: str, cheap: DetCache, full: DetCache,
                   model: G.CriticalityModel, cfg: RiskConfig,
                   geom: np.ndarray | None = None,
                   anchor_iou: float = 0.3) -> pd.DataFrame:
    geom = G.sequence_geometry(seq) if geom is None else geom
    crit_all = G.criticality_for(geom, model)
    calib = kitti.load_calib(seq)
    rows = []

    for i, frame in enumerate(cheap.frames):
        frame = int(frame)
        j = full.index[frame]
        sel = geom["frame"] == frame
        g, crit = geom[sel], crit_all[sel]
        keep_gt = (g["y2"] - g["y1"]) >= cfg.min_gt_height
        g, crit = g[keep_gt], crit[keep_gt]
        gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
              if len(g) else np.zeros((0, 4)))

        dc, df = cheap.det(i), full.det(j)
        geo = cheap.geo(i)
        sc = cheap.scalars(i)

        # -- oracle side: was this GT object handled correctly by each mode?
        kc = dc["conf"] >= cfg.op_conf
        kf = df["conf"] >= cfg.op_conf
        cls = np.array(["v"] * len(gt))
        bc, _ = match(gt, cls, dc["xyxy"][kc].astype(float), dc["conf"][kc], dc["coarse"][kc], cfg)
        bf, _ = match(gt, cls, df["xyxy"][kf].astype(float), df["conf"][kf], df["coarse"][kf], cfg)
        gt_cheap_ok = bc >= cfg.iou_thr
        gt_full_ok = bf >= cfg.iou_thr

        # -- deployable side: candidates, including sub-threshold ones
        cand = dc["xyxy"].astype(float)
        n = len(cand)
        if n == 0:
            continue
        gt_of_cand, cand_of_gt, best_iou = _assign_gt_to_candidates(
            gt, cand, dc["conf"], anchor_iou)

        w = np.clip(cand[:, 2] - cand[:, 0], 1e-3, None)
        h = np.clip(cand[:, 3] - cand[:, 1], 1e-3, None)
        cx = (cand[:, 0] + cand[:, 2]) / 2
        cy_ = (cand[:, 1] + cand[:, 3]) / 2
        img_w = sc.get("img_w", 1242.0)
        img_h = sc.get("img_h", 375.0)
        c_hat = (model(geo["z"], geo["lat_min"], geo["lat_max"], geo["ttc"])
                 if n else np.zeros(0))

        # crowding: how contested is this candidate's neighbourhood
        if n > 1:
            iou_self = box_iou(cand, cand)
            np.fill_diagonal(iou_self, 0.0)
            n_overlap = (iou_self > 0.2).sum(1).astype(float)
            max_overlap = iou_self.max(1)
        else:
            n_overlap = np.zeros(n)
            max_overlap = np.zeros(n)

        conf = dc["conf"].astype(float)
        for k in range(n):
            gi = gt_of_cand[k]
            has_gt = gi >= 0
            if has_gt:
                cheap_ok = bool(gt_cheap_ok[gi])
                full_ok = bool(gt_full_ok[gi])
                c_gt = float(crit[gi])
                gain = float(int(not cheap_ok) - int(not full_ok))   # err_cheap - err_full
                dist_gt = float(g["long_near"][gi])
                h_gt = float(g["y2"][gi] - g["y1"][gi])
            else:
                cheap_ok, full_ok, c_gt, gain, dist_gt, h_gt = True, True, 0.0, 0.0, np.nan, np.nan

            rows.append({
                "seq": seq, "frame": frame, "cand": k,
                # ---- labels (oracle) ----
                "has_gt": has_gt,
                "cheap_fail": bool(has_gt and not cheap_ok),
                "full_ok": bool(has_gt and full_ok),
                "full_recovers": bool(has_gt and (not cheap_ok) and full_ok),
                "gain": gain,
                "crit_gt": c_gt,
                "value": c_gt * gain,
                "gt_dist": dist_gt, "gt_h_px": h_gt,
                # ---- deployable features ----
                _f("o_conf", "unc"): float(conf[k]),
                _f("o_logit", "unc"): float(np.log(np.clip(conf[k], 1e-6, 1 - 1e-6) /
                                                   (1 - np.clip(conf[k], 1e-6, 1 - 1e-6)))),
                _f("o_entropy", "unc"): float(dc["entropy"][k]),
                _f("o_margin", "unc"): float(dc["margin"][k]),
                _f("o_binent", "unc"): float(dc["binent"][k]),
                _f("o_below_thr", "unc"): float(conf[k] < cfg.op_conf),
                _f("o_conf_gap_to_thr", "unc"): float(cfg.op_conf - conf[k]),
                _f("o_h_px", "size"): float(h[k]),
                _f("o_w_px", "size"): float(w[k]),
                _f("o_area_frac", "size"): float(w[k] * h[k] / (img_w * img_h)),
                _f("o_log_h", "size"): float(np.log(h[k])),
                _f("o_aspect", "size"): float(w[k] / h[k]),
                _f("o_cx_frac", "size"): float(cx[k] / img_w),
                _f("o_cy_frac", "size"): float(cy_[k] / img_h),
                _f("o_bottom_below_horizon", "size"): float(cand[k, 3] - calib.cy),
                _f("o_touches_edge", "size"): float(
                    (cand[k, 0] <= 2) or (cand[k, 2] >= img_w - 3)),
                _f("o_z", "crit"): float(geo["z"][k]),
                _f("o_z_ground", "crit"): float(geo["z_ground"][k]),
                _f("o_z_height", "crit"): float(geo["z_height"][k]),
                _f("o_lat_abs", "crit"): float(max(0.0, max(-geo["lat_max"][k], geo["lat_min"][k]))),
                _f("o_ttc", "crit"): float(min(geo["ttc"][k], 1e3)),
                _f("o_ttc_inv", "crit"): float(1.0 / max(min(geo["ttc"][k], 1e3), 0.05)),
                _f("o_c_hat", "crit"): float(c_hat[k]),
                _f("o_is_person", "crit"): float(dc["coarse"][k] == "person"),
                _f("o_is_cyclist", "crit"): float(dc["coarse"][k] == "cyclist"),
                _f("o_n_overlap", "ctx"): float(n_overlap[k]),
                _f("o_max_overlap", "ctx"): float(max_overlap[k]),
                _f("o_n_cand_frame", "ctx"): float(n),
                _f("o_conf_rank", "ctx"): float((conf > conf[k]).sum()),
                _f("o_frame_conf_mean", "ctx"): float(conf.mean()),
                _f("o_frame_conf_max", "ctx"): float(conf.max()),
                _f("o_motion_mean", "ctx"): float(sc.get("motion_mean", 0.0)),
                _f("o_img_bright_mean", "ctx"): float(sc.get("img_bright_mean", 0.0)),
                _f("o_img_lap_var", "ctx"): float(sc.get("img_lap_var", 0.0)),
                _f("o_prev_h", "ctx"): float(geo["box_h"][k]),
            })

        # GT objects with no candidate at all: recorded so the anchor ceiling is measurable
        for gi in np.flatnonzero(cand_of_gt < 0):
            rows.append({
                "seq": seq, "frame": frame, "cand": -1, "has_gt": True,
                "cheap_fail": not bool(gt_cheap_ok[gi]), "full_ok": bool(gt_full_ok[gi]),
                "full_recovers": bool((not gt_cheap_ok[gi]) and gt_full_ok[gi]),
                "gain": float(int(not gt_cheap_ok[gi]) - int(not gt_full_ok[gi])),
                "crit_gt": float(crit[gi]),
                "value": float(crit[gi]) * float(int(not gt_cheap_ok[gi]) - int(not gt_full_ok[gi])),
                "gt_dist": float(g["long_near"][gi]),
                "gt_h_px": float(g["y2"][gi] - g["y1"][gi]),
            })

    df = pd.DataFrame(rows)
    for c in object_columns():
        if c in df:
            df[c] = df[c].astype(float)
    return df


def anchor_coverage(df: pd.DataFrame) -> dict:
    """How much of the recoverable risk an object-anchored model can even see."""
    anchored = df["cand"] >= 0
    rec = df["full_recovers"]
    tot_value = df.loc[df["gain"] > 0, "value"].sum()
    return {
        "n_rows": int(len(df)),
        "n_anchored": int(anchored.sum()),
        "n_unanchored_gt": int((~anchored).sum()),
        "recovered_anchored_frac": float((anchored & rec).sum() / max(rec.sum(), 1)),
        "recovered_risk_anchored_frac": float(
            df.loc[anchored & (df["gain"] > 0), "value"].sum() / max(tot_value, 1e-9)),
        "p_recover_given_fail": float(
            df.loc[df["cheap_fail"], "full_ok"].mean()) if df["cheap_fail"].any() else np.nan,
    }
