"""Planner D — a learned downstream planner that is independent of PKL.

Phase 0F left the decisive comparison unbalanced.  Planner A and B are hand-written, and
Planner C is PKL's own released planner, which is close to circular with PKL: PKL is a
divergence between predicted and ground-truth-conditioned trajectory heatmaps, and J_C is
the displacement of those same heatmaps' argmax.  Planner D fills the missing quadrant — a
*learned* planner independent of PKL's architecture and objective — so that C-vs-D changes
only the planner while holding dataset, BEV representation, fidelity pair, perception
outputs, geometry variant and scene set fixed.

What is deliberately shared with PKL: the 5-channel BEV raster, built by calling PKL's own
rasteriser.  What is deliberately different: the architecture (a plain conv encoder, no
skips), the output parameterisation (16x2 regressed waypoints, not 16 spatial heatmaps), the
objective (MSE on coordinates, not masked BCE on cells), and the evaluation target (error
against the *real* future ego trajectory, not against the planner's own GT-conditioned
output).

Nothing here reads PKL weights, reuses PKL layers, distils PKL, or sees a PKL/TIP score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# 16 horizons at 0.25 s spacing, matching the released planner's local_ts
HORIZONS = np.arange(0.25, 4.01, 0.25)
assert len(HORIZONS) == 16


class PlannerD(nn.Module):
    """Frozen architecture (Phase 0G pre-registration): 4 strided conv blocks -> GAP -> MLP."""

    def __init__(self, cin: int = 5, nwp: int = 16, width=(32, 64, 128, 256)):
        super().__init__()
        blocks, c = [], cin
        for w in width:
            blocks += [nn.Conv2d(c, w, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, w), w),
                       nn.ReLU(inplace=True)]
            c = w
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(c, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, nwp * 2))
        self.nwp = nwp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.encoder(x)).flatten(1)
        return self.head(h).reshape(-1, self.nwp, 2)


@dataclass(frozen=True)
class Corruption:
    """Fixed synthetic perception corruption for the D-Aug variant.

    Values stipulated in the Phase 0G pre-registration before any allocation result, and not
    fitted to the Phase-0F test detections.  D-Aug exists only to check whether a D-GT
    result is an artefact of the train/test distribution gap.
    """
    p_drop: float = 0.10
    sigma_xy: float = 0.50          # metres
    sigma_size: float = 0.10        # multiplicative
    sigma_yaw_deg: float = 5.0

    def apply(self, lobjs: np.ndarray, lws: np.ndarray, rng) -> tuple:
        """Corrupt ego-frame objects (N,4 as x,y,cos,sin) and their (N,2) length/width."""
        if len(lobjs) == 0:
            return lobjs, lws
        keep = rng.random(len(lobjs)) >= self.p_drop
        o, w = lobjs[keep].copy(), lws[keep].copy()
        if len(o) == 0:
            return np.zeros((0, 4)), np.zeros((0, 2))
        o[:, :2] += rng.normal(0.0, self.sigma_xy, o[:, :2].shape)
        yaw = np.arctan2(o[:, 3], o[:, 2]) + np.deg2rad(
            rng.normal(0.0, self.sigma_yaw_deg, len(o)))
        o[:, 2], o[:, 3] = np.cos(yaw), np.sin(yaw)
        w *= np.clip(1.0 + rng.normal(0.0, self.sigma_size, w.shape), 0.3, 3.0)
        return o, w


def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple:
    """Average and final displacement error, in metres, over (N,16,2) arrays."""
    d = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return d.mean(1), d[:, -1]


def constant_velocity(v_xy: np.ndarray) -> np.ndarray:
    """Baseline: hold the current ego velocity and heading.

    `v_xy` is (N,2) ego-frame velocity in m/s at the current instant; the ego frame has the
    ego at the origin facing +x, so the prediction is simply v * t.
    """
    return np.asarray(v_xy)[:, None, :] * HORIZONS[None, :, None]


# --------------------------------------------------------------------------------------
# Object-channel re-rendering, for the D-Aug variant.
#
# The 5 raster channels are [road, road_divider, lane_divider, objects, ego] and only the
# object channel depends on the boxes, so corrupting perception means re-rendering channel 3
# alone.  PKL's own `get_corners` and the same fillPoly call are used, so a corrupted raster
# is drawn exactly the way an uncorrupted one is.

OBJ_CHANNEL = 3


def _pkl_render():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "third_party" / "pkl"))
    from planning_centric_metrics.planning_kl import get_corners, get_grid
    return get_corners, get_grid


def object_channel(lobjs: np.ndarray, lws: np.ndarray) -> np.ndarray:
    """Redraw the object channel from ego-frame boxes, as PKL draws it."""
    import cv2
    get_corners, get_grid = _pkl_render()
    dx, bx, (nx, ny) = get_grid([-17.0, -38.5, 60.0, 38.5], [0.3, 0.3])
    img = np.zeros((nx, ny), np.float32)
    for box, lw in zip(np.asarray(lobjs, float), np.asarray(lws, float)):
        pts = get_corners(box, lw)
        pts = np.round((pts - bx[:2] + dx[:2] / 2.0) / dx[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(img, [pts], 1.0)
    return img
