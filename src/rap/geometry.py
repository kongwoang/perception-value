"""Downstream-criticality geometry, computed from ground truth.

This is the *oracle* side of Phase 0: it defines what "mattering" means for a
detected object, using GT 3D geometry that no deployed policy is allowed to see.

Everything is expressed in the IMU/ego frame (x forward, y left, z up), so
"longitudinal", "lateral" and "corridor" have their physical meaning.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import kitti


# ----------------------------------------------------------------------------- per-object geometry

GEOM_FIELDS = [
    ("seq", "U8"), ("frame", "i4"), ("track_id", "i4"), ("type", "U16"),
    ("x1", "f4"), ("y1", "f4"), ("x2", "f4"), ("y2", "f4"),
    ("truncated", "f4"), ("occluded", "i4"),
    ("long_near", "f4"),     # nearest longitudinal distance of the footprint [m]
    ("long_cen", "f4"),      # footprint centre longitudinal distance [m]
    ("lat_cen", "f4"),       # footprint centre lateral offset, +left [m]
    ("lat_min", "f4"),       # footprint lateral extent, low edge
    ("lat_max", "f4"),       # footprint lateral extent, high edge
    ("range_rate", "f4"),    # d(long_near)/dt [m/s], negative = closing
    ("ttc", "f4"),           # long_near / closing rate [s], inf if opening
    ("has_rate", "?"),
]


def sequence_geometry(seq: str, smooth_halfwidth: int = 2) -> np.ndarray:
    """Per-object ego-frame geometry for every evaluable GT object in a sequence."""
    calib = kitti.load_calib(seq)
    lab = kitti.load_labels(seq)
    keep = np.isin(lab["type"], kitti.EVAL_TYPES)
    lab = lab[keep]

    out = np.zeros(len(lab), dtype=GEOM_FIELDS)
    for i, row in enumerate(lab):
        corners = kitti.box3d_corners_cam(row)
        imu = kitti.cam_points_to_imu(corners[:4], calib)  # bottom face is enough
        out[i]["seq"] = seq
        out[i]["frame"] = row["frame"]
        out[i]["track_id"] = row["track_id"]
        out[i]["type"] = row["type"]
        for c in ("x1", "y1", "x2", "y2", "truncated", "occluded"):
            out[i][c] = row[c]
        out[i]["long_near"] = imu[:, 0].min()
        out[i]["long_cen"] = imu[:, 0].mean()
        out[i]["lat_cen"] = imu[:, 1].mean()
        out[i]["lat_min"] = imu[:, 1].min()
        out[i]["lat_max"] = imu[:, 1].max()

    _fill_range_rate(out, smooth_halfwidth)
    return out


def _fill_range_rate(geom: np.ndarray, halfwidth: int) -> None:
    """Range rate per track by local linear fit of long_near against time.

    long_near is measured in the *current* camera frame, so its time derivative is
    already the relative closing rate — exactly what TTC needs, with no ego-motion
    compensation.
    """
    geom["ttc"] = np.inf
    for tid in np.unique(geom["track_id"]):
        if tid < 0:
            continue
        idx = np.flatnonzero(geom["track_id"] == tid)
        order = idx[np.argsort(geom["frame"][idx])]
        f = geom["frame"][order].astype(float) * kitti.FRAME_DT
        d = geom["long_near"][order].astype(float)
        n = len(order)
        for j in range(n):
            lo, hi = max(0, j - halfwidth), min(n, j + halfwidth + 1)
            # only fit over temporally contiguous neighbours
            while lo < j and f[j] - f[lo] > (halfwidth + 0.5) * kitti.FRAME_DT:
                lo += 1
            while hi - 1 > j and f[hi - 1] - f[j] > (halfwidth + 0.5) * kitti.FRAME_DT:
                hi -= 1
            if hi - lo < 2:
                continue
            slope = np.polyfit(f[lo:hi], d[lo:hi], 1)[0]
            k = order[j]
            geom["range_rate"][k] = slope
            geom["has_rate"][k] = True
            closing = -slope
            geom["ttc"][k] = d[j] / closing if closing > 1e-3 else np.inf


# ----------------------------------------------------------------------------- criticality models

@dataclass(frozen=True)
class CriticalityModel:
    """Maps ego-frame geometry to a scalar in [0, 1]: how much this object matters.

    `composite` is the primary model; the others exist so every headline number can
    be re-run under a different definition of "matters" (GO criterion 6).
    """
    name: str
    kind: str = "composite"          # composite | corridor | proximity | ttc | uniform | binary
    corridor_half_w: float = 0.9     # ego half-width [m]
    lat_sigma: float = 1.5           # softness of the corridor falloff [m]
    range_free: float = 5.0          # distance below which proximity term saturates [m]
    range_tau: float = 20.0          # proximity decay constant [m]
    ttc_hot: float = 1.5             # TTC at/below which the TTC term saturates [s]
    ttc_cold: float = 7.0            # TTC above which the TTC term is 0 [s]
    ttc_weight: float = 0.5          # blend between proximity and TTC inside composite
    floor: float = 0.0               # criticality floor for non-critical objects
    max_range: float = 80.0          # beyond this an object is irrelevant

    def corridor_term(self, lat_min, lat_max):
        """1 inside the ego corridor, decaying with lateral clearance."""
        clearance = np.maximum(0.0, np.maximum(-lat_max, lat_min) - self.corridor_half_w)
        return np.exp(-0.5 * (clearance / self.lat_sigma) ** 2)

    def proximity_term(self, long_near):
        return np.exp(-np.maximum(0.0, long_near - self.range_free) / self.range_tau)

    def ttc_term(self, ttc):
        t = np.clip(ttc, 0.0, None)
        return np.clip((self.ttc_cold - t) / (self.ttc_cold - self.ttc_hot), 0.0, 1.0)

    def __call__(self, long_near, lat_min, lat_max, ttc) -> np.ndarray:
        long_near = np.asarray(long_near, dtype=np.float64)
        lat_min = np.asarray(lat_min, dtype=np.float64)
        lat_max = np.asarray(lat_max, dtype=np.float64)
        ttc = np.asarray(ttc, dtype=np.float64)

        if self.kind == "uniform":
            return np.ones_like(long_near)

        corr = self.corridor_term(lat_min, lat_max)
        prox = self.proximity_term(long_near)
        ttcw = self.ttc_term(ttc)

        if self.kind == "corridor":
            c = corr
        elif self.kind == "proximity":
            c = prox
        elif self.kind == "ttc":
            c = ttcw
        elif self.kind == "binary":
            in_corr = (lat_min < self.corridor_half_w) & (lat_max > -self.corridor_half_w)
            urgent = (long_near < 20.0) | (ttc < 4.0)
            c = np.where(in_corr & urgent, 1.0, 0.0)
        else:  # composite
            c = corr * ((1 - self.ttc_weight) * prox + self.ttc_weight * ttcw)

        c = np.where(long_near > self.max_range, 0.0, c)
        c = np.where(long_near < -2.0, 0.0, c)  # behind the ego origin
        return self.floor + (1 - self.floor) * np.clip(c, 0.0, 1.0)


PRIMARY = CriticalityModel("composite")

CRITICALITY_MODELS = {
    m.name: m for m in [
        PRIMARY,
        CriticalityModel("uniform", kind="uniform"),
        CriticalityModel("corridor", kind="corridor"),
        CriticalityModel("proximity", kind="proximity"),
        CriticalityModel("ttc", kind="ttc"),
        CriticalityModel("binary", kind="binary"),
        replace(PRIMARY, name="composite_wide", corridor_half_w=1.8, lat_sigma=2.5),
        replace(PRIMARY, name="composite_narrow", corridor_half_w=0.9, lat_sigma=0.6),
        replace(PRIMARY, name="composite_near", range_tau=10.0),
        replace(PRIMARY, name="composite_far", range_tau=40.0),
        replace(PRIMARY, name="composite_ttcheavy", ttc_weight=0.85),
        replace(PRIMARY, name="composite_proxheavy", ttc_weight=0.15),
    ]
}


def criticality_for(geom: np.ndarray, model: CriticalityModel) -> np.ndarray:
    return model(geom["long_near"], geom["lat_min"], geom["lat_max"], geom["ttc"])
