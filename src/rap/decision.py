"""Build the per-frame decision table for a given perception pair and geometry source.

Phase 0C measured `dJ` with the deployed monocular range estimator. Phase 0D has to
separate three things that were entangled there:

  * which objects each mode *detected*,
  * how well the monocular estimator *ranged* them,
  * whether the ego was moving fast enough for a braking decision to exist at all.

`range_source` controls the second one. "mono" is what a deployed system has; "oracle"
substitutes true geometry for detections that match a GT object, leaving only the
detection difference; "noisy_gt" puts calibrated noise back on top of oracle geometry so
the comparison is not between a noisy estimator and a perfect one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import geometry as G
from . import kitti
from . import planner as P
from .cache import DetCache
from .mono import box_iou
from .risk import RiskConfig


class KittiAdapter:
    """Dataset-specific pieces the decision builder needs; nuScenes supplies its own."""

    name = "kitti"

    @staticmethod
    def geometry(seq):
        return G.sequence_geometry(seq)

    @staticmethod
    def speeds(seq):
        return kitti.load_oxts(seq)[:, 8]


def _apply_range_source(bx, geo_k, g, source: str, rng, sigma: float):
    """Return (z, lat_min, lat_max, ttc) for the kept detections under one geometry source."""
    z = geo_k["z"].copy()
    lo = geo_k["lat_min"].copy()
    hi = geo_k["lat_max"].copy()
    tt = geo_k["ttc"].copy()
    if source == "mono" or len(bx) == 0 or len(g) == 0:
        return z, lo, hi, tt
    gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
    iou = box_iou(bx, gt)
    best = iou.argmax(1)
    matched = iou.max(1) >= 0.5
    if not matched.any():
        return z, lo, hi, tt
    zt = g["long_near"][best[matched]].astype(float)
    if source == "noisy_gt":
        # multiplicative range noise at the scale the monocular estimator actually shows
        zt = zt * np.clip(1.0 + rng.normal(0.0, sigma, zt.shape), 0.2, 3.0)
        scale = zt / np.maximum(g["long_near"][best[matched]].astype(float), 1e-6)
        lo[matched] = g["lat_min"][best[matched]] * scale
        hi[matched] = g["lat_max"][best[matched]] * scale
    else:
        lo[matched] = g["lat_min"][best[matched]]
        hi[matched] = g["lat_max"][best[matched]]
    z[matched] = zt
    tt[matched] = g["ttc"][best[matched]]
    return z, lo, hi, tt


def mono_range_sigma(det_dir, cheap_mode, seqs, cfg: RiskConfig,
                     adapter=KittiAdapter) -> float:
    """Relative range error of the deployed monocular estimator, for noise calibration."""
    errs = []
    for s in seqs:
        geom = adapter.geometry(s)
        c = DetCache(det_dir / cheap_mode / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            if not len(g):
                continue
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.op_conf
            bx = d["xyxy"][k].astype(float)
            if not len(bx):
                continue
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            iou = box_iou(bx, gt)
            m = iou.max(1) >= 0.5
            if m.any():
                zt = g["long_near"][iou.argmax(1)[m]].astype(float)
                errs.append((geo["z"][k][m] - zt) / np.maximum(zt, 1e-6))
    e = np.concatenate(errs) if errs else np.zeros(1)
    return float(np.std(np.clip(e, -2, 2)))


def build(det_dir, cheap_mode: str, full_mode: str, seqs, cfg: RiskConfig,
          pp: P.PlannerParams, cp: P.CostParams, crit_model,
          range_source: str = "mono", sigma: float = 0.0, seed: int = 0,
          lp: P.LateralParams | None = None,
          lc: P.LateralCostParams | None = None,
          adapter=KittiAdapter) -> pd.DataFrame:
    """One row per frame. Both downstream tasks are scored from the same perception."""
    lp = lp or P.LateralParams()
    lc = lc or P.LateralCostParams()
    rng = np.random.default_rng(seed)
    rows = []
    for s in seqs:
        geom = adapter.geometry(s)
        crit_all = G.criticality_for(geom, crit_model)
        speeds = adapter.speeds(s)
        c = DetCache(det_dir / cheap_mode / f"{s}.npz")
        f = DetCache(det_dir / full_mode / f"{s}.npz")
        prev_c = prev_f = None
        prev_lc_ = prev_lf_ = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            sel = geom["frame"] == fr
            g, cg = geom[sel], crit_all[sel]
            keep = (g["y2"] - g["y1"]) >= cfg.min_gt_height
            g, cg = g[keep], cg[keep]

            out = {}
            for tag, cache, idx in (("cheap", c, i), ("full", f, j)):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.op_conf
                geo_k = {kk: vv[k] for kk, vv in geo.items()}
                bx = d["xyxy"][k].astype(float)
                z, lo, hi, tt = _apply_range_source(bx, geo_k, g, range_source, rng, sigma)
                a, _ = P.required_decel(z, lo, hi, tt, v, pp)
                lat_a, lat_off = P.lateral_action(z, lo, hi, v, lp)
                out[tag] = {"a": a, "n": int(k.sum()), "conf": d["conf"][k],
                            "ent": d["binent"][k], "lat_a": lat_a, "lat_off": lat_off}
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"],
                                       g["ttc"], v, pp)
            act_c = P.discrete_action(out["cheap"]["a"], pp)
            act_f = P.discrete_action(out["full"]["a"], pp)
            Jc = P.decision_cost(act_c, a_gt, prev_c, pp, cp)
            Jf = P.decision_cost(act_f, a_gt, prev_f, pp, cp)
            prev_c, prev_f = act_c, act_f

            Lc = P.lateral_cost(out["cheap"]["lat_a"], out["cheap"]["lat_off"],
                                g["long_near"], g["lat_min"], g["lat_max"], v, prev_lc_, lp, lc)
            Lf = P.lateral_cost(out["full"]["lat_a"], out["full"]["lat_off"],
                                g["long_near"], g["lat_min"], g["lat_max"], v, prev_lf_, lp, lc)
            lat_gt_a, lat_gt_off = P.lateral_action(g["long_near"], g["lat_min"],
                                                    g["lat_max"], v, lp)
            prev_lc_, prev_lf_ = out["cheap"]["lat_a"], out["full"]["lat_a"]
            conf = out["cheap"]["conf"]
            rows.append({
                "seq": s, "frame": fr, "v_ego": v,
                "a_cheap": out["cheap"]["a"], "a_full": out["full"]["a"], "a_gt": a_gt,
                "act_cheap": act_c, "act_full": act_f,
                "J_cheap": Jc["J"], "J_full": Jf["J"], "dJ": Jc["J"] - Jf["J"],
                "dJ_long": Jc["J"] - Jf["J"],
                "collision_cheap": Jc["collision"], "collision_full": Jf["collision"],
                "same_action": int(act_c == act_f),
                "n_cheap": out["cheap"]["n"], "n_full": out["full"]["n"],
                "empty_cheap": int(out["cheap"]["n"] == 0),
                "lat_act_cheap": out["cheap"]["lat_a"], "lat_act_full": out["full"]["lat_a"],
                "lat_act_gt": lat_gt_a,
                "Jlat_cheap": Lc["J"], "Jlat_full": Lf["J"], "dJ_lat": Lc["J"] - Lf["J"],
                "lat_same_action": int(out["cheap"]["lat_a"] == out["full"]["lat_a"]),
                "lat_collision_cheap": Lc["collision"], "lat_collision_full": Lf["collision"],
                "crit_sum": float(cg.sum()),
                "unc_sum": float(out["cheap"]["ent"].sum()),
                "conf_mean": float(conf.mean()) if len(conf) else 0.0,
                "n_gt": int(len(g)),
            })
    return pd.DataFrame(rows)


def add_perception_gain(df: pd.DataFrame, det_dir, cheap_mode, full_mode, seqs,
                        cfg: RiskConfig, adapter=KittiAdapter) -> pd.DataFrame:
    """dE = detections recovered, matched the same way Phase 0 did."""
    from .risk import match
    rec = []
    for s in seqs:
        geom = adapter.geometry(s)
        c = DetCache(det_dir / cheap_mode / f"{s}.npz")
        f = DetCache(det_dir / full_mode / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                  if len(g) else np.zeros((0, 4)))
            cls = np.array(["v"] * len(gt))
            e = {}
            for tag, cache, idx in (("cheap", c, i), ("full", f, f.index[fr])):
                d = cache.det(idx)
                k = d["conf"] >= cfg.op_conf
                b, _ = match(gt, cls, d["xyxy"][k].astype(float), d["conf"][k],
                             d["coarse"][k], cfg)
                e[tag] = float((b < cfg.iou_thr).sum())
            rec.append({"seq": s, "frame": fr, "dE": e["cheap"] - e["full"]})
    return df.merge(pd.DataFrame(rec), on=["seq", "frame"], how="left", validate="one_to_one")
