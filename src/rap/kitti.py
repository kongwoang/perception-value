"""KITTI tracking loader: labels, calibration, ego motion, and the camera->ego transform.

KITTI conventions used here:
  * label `location` is the centre of the BOTTOM face of the 3D box, in the
    rectified reference-camera frame (x right, y down, z forward).
  * velodyne frame is x forward, y left, z up; the IMU frame shares that layout.
    All downstream geometry is expressed in the IMU frame, which is the one that
    is physically meaningful for "is this object in the ego corridor".
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import KITTI_CALIB, KITTI_IMAGES, KITTI_LABELS, KITTI_OXTS

# KITTI label columns (tracking variant)
LABEL_COLS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2", "h", "w", "l", "loc_x", "loc_y", "loc_z", "ry",
]

# Classes that are traffic participants we hold the detector responsible for.
EVAL_TYPES = ("Car", "Van", "Truck", "Tram", "Pedestrian", "Person", "Person_sitting", "Cyclist")
IGNORE_TYPES = ("DontCare", "Misc")

# Coarse class used when matching is class-aware.
TYPE_TO_COARSE = {
    "Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
    "Pedestrian": "person", "Person": "person", "Person_sitting": "person",
    "Cyclist": "cyclist",
}

FRAME_DT = 0.1  # KITTI tracking is 10 Hz


def sequences() -> list[str]:
    return sorted(p.stem for p in KITTI_LABELS.glob("*.txt"))


@dataclass(frozen=True)
class Calib:
    P2: np.ndarray        # 3x4 projection, rectified cam2
    R_rect: np.ndarray    # 4x4
    Tr_velo_cam: np.ndarray  # 4x4
    Tr_imu_velo: np.ndarray  # 4x4

    @property
    def fx(self) -> float: return float(self.P2[0, 0])

    @property
    def fy(self) -> float: return float(self.P2[1, 1])

    @property
    def cx(self) -> float: return float(self.P2[0, 2])

    @property
    def cy(self) -> float: return float(self.P2[1, 2])

    @functools.cached_property
    def cam_to_imu(self) -> np.ndarray:
        """4x4 taking rectified-camera points to the IMU frame (x fwd, y left, z up)."""
        return np.linalg.inv(self.Tr_imu_velo) @ np.linalg.inv(self.Tr_velo_cam) @ np.linalg.inv(self.R_rect)


def _to44(vals: np.ndarray, rows: int) -> np.ndarray:
    M = np.eye(4)
    M[:rows, : vals.size // rows] = vals.reshape(rows, -1)
    return M


@functools.lru_cache(maxsize=64)
def load_calib(seq: str) -> Calib:
    raw: dict[str, np.ndarray] = {}
    for line in (KITTI_CALIB / f"{seq}.txt").read_text().strip().splitlines():
        if not line.strip():
            continue
        key, _, rest = line.strip().partition(" ")
        raw[key.rstrip(":")] = np.fromstring(rest, sep=" ")
    R = np.eye(4)
    R[:3, :3] = raw["R_rect"].reshape(3, 3)
    return Calib(
        P2=raw["P2"].reshape(3, 4),
        R_rect=R,
        Tr_velo_cam=_to44(raw["Tr_velo_cam"], 3),
        Tr_imu_velo=_to44(raw["Tr_imu_velo"], 3),
    )


@functools.lru_cache(maxsize=64)
def load_labels(seq: str) -> np.ndarray:
    """Structured array of every label row in a sequence (including DontCare)."""
    rows = []
    for line in (KITTI_LABELS / f"{seq}.txt").read_text().strip().splitlines():
        f = line.split()
        rows.append((
            int(f[0]), int(f[1]), f[2], float(f[3]), int(f[4]), float(f[5]),
            *(float(v) for v in f[6:17]),
        ))
    dtype = [("frame", "i4"), ("track_id", "i4"), ("type", "U16"),
             ("truncated", "f4"), ("occluded", "i4"), ("alpha", "f4")] + \
            [(c, "f4") for c in LABEL_COLS[6:]]
    return np.array(rows, dtype=dtype)


@functools.lru_cache(maxsize=64)
def load_oxts(seq: str) -> np.ndarray:
    """(n_frames, 30) oxts rows. Column 8 is forward velocity vf [m/s], 22 is yaw rate wz."""
    return np.loadtxt(KITTI_OXTS / f"{seq}.txt", dtype=np.float64)


def image_path(seq: str, frame: int) -> Path:
    return KITTI_IMAGES / seq / f"{frame:06d}.png"


@functools.lru_cache(maxsize=64)
def frame_ids(seq: str) -> np.ndarray:
    """Frames that exist on disk as images, ascending."""
    return np.array(sorted(int(p.stem) for p in (KITTI_IMAGES / seq).glob("*.png")), dtype=np.int32)


def box3d_corners_cam(row) -> np.ndarray:
    """(8,3) corners of the 3D box in rectified-camera coords. Bottom face first."""
    h, w, l = float(row["h"]), float(row["w"]), float(row["l"])
    ry = float(row["ry"])
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=float)
    z = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts = R @ np.stack([x, y, z])
    pts += np.array([row["loc_x"], row["loc_y"], row["loc_z"]], dtype=float)[:, None]
    return pts.T


def cam_points_to_imu(pts_cam: np.ndarray, calib: Calib) -> np.ndarray:
    """(n,3) rectified-camera points -> (n,3) IMU points (x forward, y left, z up)."""
    homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
    return (calib.cam_to_imu @ homo.T).T[:, :3]
