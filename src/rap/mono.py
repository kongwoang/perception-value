"""Criticality estimated from CHEAP detections alone — the deployable counterpart
to `geometry.py`.

Nothing here may touch ground truth, the FULL prediction, or a future frame. It
uses only: the CHEAP 2D boxes of the current frame, the CHEAP boxes of the
*previous* frame, and the camera calibration (known at build time on a real
vehicle).

Two independent range cues are used because each fails differently:
  * ground-plane cue: z = f_y * h_cam / (v_bottom - c_y), exact for an object
    standing on a flat road, degenerate near the horizon;
  * height-prior cue: z = f_y * H_class / h_box, robust near the horizon but
    wrong for atypically sized objects.
"""
from __future__ import annotations

import numpy as np

from .kitti import Calib, FRAME_DT

# Typical physical heights [m] used as the class-conditional size prior.
HEIGHT_PRIOR = {"vehicle": 1.55, "person": 1.72, "cyclist": 1.72}
CAMERA_HEIGHT = 1.65  # KITTI cam2 height above the road [m]


def range_ground(y2: np.ndarray, calib: Calib, min_px: float = 2.0) -> np.ndarray:
    """Distance from where the box bottom meets the road plane."""
    dv = np.maximum(y2 - calib.cy, min_px)
    return calib.fy * CAMERA_HEIGHT / dv


def range_height(y1: np.ndarray, y2: np.ndarray, coarse: np.ndarray,
                 calib: Calib, min_px: float = 2.0) -> np.ndarray:
    """Distance from the apparent box height and a class height prior."""
    h = np.maximum(y2 - y1, min_px)
    prior = np.array([HEIGHT_PRIOR.get(c, 1.6) for c in coarse], dtype=np.float64)
    return calib.fy * prior / h


def fuse_range(z_ground: np.ndarray, z_height: np.ndarray, y2: np.ndarray,
               calib: Calib) -> np.ndarray:
    """Trust the ground cue when the box bottom is well below the horizon."""
    dv = y2 - calib.cy
    w_ground = np.clip((dv - 5.0) / 25.0, 0.0, 1.0)
    return w_ground * z_ground + (1 - w_ground) * z_height


def lateral_offset(x1: np.ndarray, x2: np.ndarray, z: np.ndarray, calib: Calib):
    """Lateral extent of the object in the ego frame (+left), from the box edges.

    The camera x axis points right, the ego y axis points left, so the sign flips.
    A constant lateral mounting offset of cam2 is absorbed into the principal point,
    which is what a deployed system would calibrate anyway.
    """
    xl = (x1 - calib.cx) * z / calib.fx
    xr = (x2 - calib.cx) * z / calib.fx
    lat_max, lat_min = -xl, -xr           # flip right-handedness
    return lat_min, lat_max


def ttc_from_scale(curr_h: np.ndarray, prev_h: np.ndarray, dt: float = FRAME_DT,
                   max_ttc: float = 1e3) -> np.ndarray:
    """Scale-change TTC: a box growing by a factor s over dt hits in dt/(s-1).

    Purely monocular and calibration-free; this is the classic looming cue.
    """
    ttc = np.full(len(curr_h), np.inf)
    ok = np.isfinite(prev_h) & (prev_h > 1) & (curr_h > 1)
    growth = np.zeros_like(ttc)
    growth[ok] = curr_h[ok] / prev_h[ok]
    expanding = ok & (growth > 1.0 + 1e-3)
    ttc[expanding] = dt / (growth[expanding] - 1.0)
    return np.minimum(ttc, max_ttc)


def associate_prev(curr_xyxy: np.ndarray, prev_xyxy: np.ndarray,
                   iou_thr: float = 0.3) -> np.ndarray:
    """Greedy IoU association of current boxes to the previous frame's boxes.

    Returns, per current box, the previous box height or NaN. Frame-to-frame motion
    at 10 Hz is small enough that plain IoU is adequate; this is the cheapest
    association a real gate could afford.
    """
    out = np.full(len(curr_xyxy), np.nan)
    if len(prev_xyxy) == 0 or len(curr_xyxy) == 0:
        return out
    iou = box_iou(curr_xyxy, prev_xyxy)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        order = np.argsort(-iou[ci])
        for pj in order:
            if iou[ci, pj] < iou_thr:
                break
            if pj not in taken:
                taken.add(int(pj))
                out[ci] = prev_xyxy[pj, 3] - prev_xyxy[pj, 1]
                break
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) -> (n,m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def predicted_geometry(det: dict, prev_det: dict | None, calib: Calib) -> dict:
    """Per-detection estimated ego geometry: range, lateral extent, TTC."""
    xyxy = det["xyxy"].astype(np.float64)
    n = len(xyxy)
    if n == 0:
        z = np.zeros(0)
        return {"z": z, "z_ground": z, "z_height": z, "lat_min": z, "lat_max": z,
                "ttc": z, "box_h": z}
    x1, y1, x2, y2 = xyxy.T
    zg = range_ground(y2, calib)
    zh = range_height(y1, y2, det["coarse"], calib)
    z = np.clip(fuse_range(zg, zh, y2, calib), 0.5, 200.0)
    lat_min, lat_max = lateral_offset(x1, x2, z, calib)
    prev_h = (associate_prev(xyxy, prev_det["xyxy"].astype(np.float64))
              if prev_det is not None else np.full(n, np.nan))
    ttc = ttc_from_scale(y2 - y1, prev_h)
    return {"z": z, "z_ground": np.clip(zg, 0.5, 200.0), "z_height": np.clip(zh, 0.5, 200.0),
            "lat_min": lat_min, "lat_max": lat_max, "ttc": ttc, "box_h": y2 - y1}
