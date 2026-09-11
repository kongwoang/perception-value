"""Assemble the per-frame analysis table: oracle targets + deployable features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import geometry as G
from . import kitti
from .cache import DetCache
from .risk import RiskConfig, frame_risk


def _gt_for_frame(geom: np.ndarray, crit: np.ndarray, frame: int, cfg: RiskConfig):
    sel = geom["frame"] == frame
    g = geom[sel]
    c = crit[sel]
    h = g["y2"] - g["y1"]
    ok = h >= cfg.min_gt_height
    g, c = g[ok], c[ok]
    gt = {
        "xyxy": np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], axis=1).astype(np.float64),
        "cls": np.array([kitti.TYPE_TO_COARSE.get(t, "vehicle") for t in g["type"]]),
    }
    return gt, c, g


def _dontcare(seq: str, frame: int) -> np.ndarray:
    lab = kitti.load_labels(seq)
    sel = (lab["frame"] == frame) & np.isin(lab["type"], kitti.IGNORE_TYPES)
    d = lab[sel]
    if len(d) == 0:
        return np.zeros((0, 4))
    return np.stack([d["x1"], d["y1"], d["x2"], d["y2"]], axis=1).astype(np.float64)


def build_sequence(seq: str, cheap: DetCache, full: DetCache,
                   model: G.CriticalityModel, cfg: RiskConfig,
                   geom: np.ndarray | None = None) -> pd.DataFrame:
    geom = G.sequence_geometry(seq) if geom is None else geom
    crit = G.criticality_for(geom, model)
    rows = []
    for i, frame in enumerate(cheap.frames):
        frame = int(frame)
        j = full.index[frame]
        gt, c, graw = _gt_for_frame(geom, crit, frame, cfg)
        dc = _dontcare(seq, frame)

        dc_cheap, dc_full = cheap.det(i), full.det(j)
        geo_cheap, geo_full = cheap.geo(i), full.geo(j)
        fp_crit_cheap = model(geo_cheap["z"], geo_cheap["lat_min"],
                              geo_cheap["lat_max"], geo_cheap["ttc"]) if len(geo_cheap["z"]) else np.zeros(0)
        fp_crit_full = model(geo_full["z"], geo_full["lat_min"],
                             geo_full["lat_max"], geo_full["ttc"]) if len(geo_full["z"]) else np.zeros(0)

        rc = frame_risk(gt, dc_cheap, c, cfg, dc, fp_crit_cheap)
        rf = frame_risk(gt, dc_full, c, cfg, dc, fp_crit_full)

        sc = cheap.scalars(i)
        img_area = sc.get("img_w", 1242.0) * sc.get("img_h", 375.0)
        img_stats = {k: v for k, v in sc.items()
                     if k.startswith(("img_bright", "img_dark", "img_lap", "img_edge",
                                      "img_contrast", "motion"))}
        feats = F.frame_features(dc_cheap, geo_cheap, img_stats, img_area, model, cfg.op_conf)

        rec = {
            "seq": seq, "frame": frame,
            "risk_cheap": rc["risk"], "risk_full": rf["risk"],
            "value_task": rc["risk"] - rf["risk"],
            "err_std_cheap": rc["err_std"], "err_std_full": rf["err_std"],
            "value_visual": rc["err_std"] - rf["err_std"],
            "n_gt": rc["n_gt"], "crit_sum_gt": rc["crit_sum"],
            "n_miss_cheap": rc["n_miss"], "n_miss_full": rf["n_miss"],
            "n_fp_cheap": rc["n_fp"], "n_fp_full": rf["n_fp"],
            "n_det_cheap": rc["n_det"], "n_det_full": rf["n_det"],
            "risk_miss_cheap": rc["risk_miss"], "risk_miss_full": rf["risk_miss"],
        }
        rec.update(feats)
        rows.append(rec)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("feat_")]
    F.assert_no_leakage(cols)
    return cols
