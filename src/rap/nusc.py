"""Minimal nuScenes reader exposing the same interface as `rap.kitti`.

The official devkit does not import cleanly in this environment and pulls dependencies
that would overwrite the Jetson OpenCV build, so this reads the JSON tables directly. It
covers exactly the slice Phase 0D needs: CAM_FRONT keyframes, 3D boxes in the ego frame,
ego speed, camera calibration, and per-instance range rate for TTC.

nuScenes ego frame is x forward, y left, z up — the same convention `rap.geometry` already
uses for KITTI's IMU frame, so the criticality models and planners carry over unchanged.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import GEOM_FIELDS, _fill_range_rate
from .kitti import Calib

FRAME_DT = 0.5      # nuScenes keyframes are 2 Hz
IMG_W, IMG_H = 1600, 900

# nuScenes categories that correspond to the traffic participants KITTI holds the
# detector responsible for.
EVAL_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
                 "vehicle.construction", "human.pedestrian", "vehicle.motorcycle",
                 "vehicle.bicycle")


class NuScenesDB:
    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini"):
        self.root = Path(dataroot)
        self.version = version
        t = self.root / version
        self._t = {n: {r["token"]: r for r in json.loads((t / f"{n}.json").read_text())}
                   for n in ("scene", "sample", "sample_data", "ego_pose",
                             "calibrated_sensor", "sensor", "instance", "category",
                             "attribute", "visibility")}
        self.annotations = json.loads((t / "sample_annotation.json").read_text())
        self._ann_by_sample: dict[str, list] = {}
        for a in self.annotations:
            self._ann_by_sample.setdefault(a["sample_token"], []).append(a)
        # CAM_FRONT keyframe data rows, indexed by sample
        cam_tokens = {k for k, v in self._t["sensor"].items() if v["channel"] == "CAM_FRONT"}
        self._cam_sensor = cam_tokens
        self._cam_by_sample = {}
        for sd in self._t["sample_data"].values():
            if not sd["is_key_frame"]:
                continue
            cs = self._t["calibrated_sensor"][sd["calibrated_sensor_token"]]
            if cs["sensor_token"] in cam_tokens:
                self._cam_by_sample[sd["sample_token"]] = sd

    # ---- scenes and frames -------------------------------------------------
    @functools.cached_property
    def scenes(self) -> list[str]:
        return sorted(self._t["scene"], key=lambda k: self._t["scene"][k]["name"])

    def scene_name(self, scene: str) -> str:
        return self._t["scene"][scene]["name"]

    @functools.lru_cache(maxsize=256)
    def samples(self, scene: str) -> tuple:
        """Sample tokens of one scene, in temporal order."""
        out, tok = [], self._t["scene"][scene]["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self._t["sample"][tok]["next"]
        return tuple(out)

    def image_path(self, sample_token: str) -> Path:
        return self.root / self._cam_by_sample[sample_token]["filename"]

    # ---- calibration -------------------------------------------------------
    @functools.lru_cache(maxsize=256)
    def calib(self, sample_token: str) -> Calib:
        sd = self._cam_by_sample[sample_token]
        cs = self._t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        K = np.array(cs["camera_intrinsic"], dtype=float)
        P2 = np.zeros((3, 4))
        P2[:3, :3] = K
        return Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))

    def _cam_to_ego(self, sample_token: str):
        sd = self._cam_by_sample[sample_token]
        cs = self._t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        R = Rotation.from_quat(np.roll(cs["rotation"], -1)).as_matrix()   # w,x,y,z -> x,y,z,w
        return R, np.array(cs["translation"], dtype=float)

    def _global_to_ego(self, sample_token: str):
        sd = self._cam_by_sample[sample_token]
        pose = self._t["ego_pose"][sd["ego_pose_token"]]
        R = Rotation.from_quat(np.roll(pose["rotation"], -1)).as_matrix()
        return R, np.array(pose["translation"], dtype=float)

    # ---- ego speed ---------------------------------------------------------
    @functools.lru_cache(maxsize=256)
    def ego_speeds(self, scene: str) -> np.ndarray:
        """Forward speed per keyframe, differentiated from ego poses."""
        toks = self.samples(scene)
        pos, ts = [], []
        for tk in toks:
            _, t = self._global_to_ego(tk)
            pos.append(t)
            ts.append(self._t["sample"][tk]["timestamp"] / 1e6)
        pos, ts = np.array(pos), np.array(ts)
        v = np.zeros(len(toks))
        for i in range(len(toks)):
            a, b = max(0, i - 1), min(len(toks) - 1, i + 1)
            dt = ts[b] - ts[a]
            v[i] = np.linalg.norm(pos[b] - pos[a]) / dt if dt > 1e-6 else 0.0
        return v

    # ---- geometry ----------------------------------------------------------
    def sequence_geometry(self, scene: str, min_pts: int = 1) -> np.ndarray:
        """Same structured array `rap.geometry.sequence_geometry` returns for KITTI."""
        toks = self.samples(scene)
        rows = []
        for fi, tk in enumerate(toks):
            Rg, tg = self._global_to_ego(tk)
            Rc, tc = self._cam_to_ego(tk)
            K = self.calib(tk).P2[:3, :3]
            for a in self._ann_by_sample.get(tk, []):
                cat = self._t["category"][
                    self._t["instance"][a["instance_token"]]["category_token"]]["name"]
                if not cat.startswith(EVAL_PREFIXES):
                    continue
                if a["num_lidar_pts"] + a["num_radar_pts"] < min_pts:
                    continue
                centre = np.array(a["translation"], dtype=float)
                w, l, h = (float(x) for x in a["size"])          # nuScenes order: w, l, h
                Rb = Rotation.from_quat(np.roll(a["rotation"], -1)).as_matrix()
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2],
                                    [-l / 2, -w / 2], [-l / 2, w / 2]])
                # nuScenes `translation` is the CENTRE of the 3D box, unlike KITTI where
                # it is the centre of the bottom face. Treating it as the bottom shifted
                # every projected box up by h/2 (~30 px at 30 m) and left only 0.5% of
                # detections matching a GT box.
                foot = np.stack([np.concatenate([Rb[:2, :2] @ c, [-h / 2]]) + centre
                                 for c in corners])
                foot_ego = (Rg.T @ (foot - tg).T).T                # global -> ego
                # 3D box corners in the camera frame, for the 2D projection
                top = foot.copy()
                top[:, 2] += h
                allc = np.vstack([foot, top])
                cam = (Rc.T @ ((Rg.T @ (allc - tg).T).T - tc).T).T
                # Project only the corners in front of the image plane, then clip.
                # Requiring all eight would silently drop the *closest* objects (their
                # near corners fall behind the camera) — exactly the ones a braking
                # decision depends on; it cut mean criticality from 0.142 to 0.109.
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, IMG_W - 1)
                y1, y2 = np.clip([y1, y2], 0, IMG_H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                rows.append((self.scene_name(scene), fi, a["instance_token"],
                             cat.split(".")[0],
                             x1, y1, x2, y2, 0.0, 0,
                             foot_ego[:, 0].min(), foot_ego[:, 0].mean(),
                             foot_ego[:, 1].mean(), foot_ego[:, 1].min(),
                             foot_ego[:, 1].max(), 0.0, np.inf, False))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        # track ids must be integers for the shared range-rate routine
        inst = {t: i for i, t in enumerate(sorted({r[2] for r in rows}))}
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0], r[1], inst[r[2]]
            out[i]["type"] = r[3]
            for j, c in enumerate(("x1", "y1", "x2", "y2", "truncated", "occluded",
                                   "long_near", "long_cen", "lat_cen", "lat_min",
                                   "lat_max", "range_rate")):
                out[i][c] = r[4 + j]
            out[i]["ttc"] = np.inf
        _fill_range_rate(out, halfwidth=1, dt=FRAME_DT)   # 2 Hz keyframes, tight window
        return out


def make_adapter(db: "NuScenesDB"):
    """Adapter matching `rap.decision.KittiAdapter`, keyed by scene *name*.

    Geometry is cached because the decision builder walks every sequence several times
    (once per range source and planner variant) and reprojecting boxes is the slow part.
    """
    by_name = {db.scene_name(s): s for s in db.scenes}
    geom_cache: dict[str, np.ndarray] = {}
    speed_cache: dict[str, np.ndarray] = {}

    class NuScenesAdapter:
        name = "nuscenes"

        @staticmethod
        def geometry(seq):
            if seq not in geom_cache:
                geom_cache[seq] = db.sequence_geometry(by_name[seq])
            return geom_cache[seq]

        @staticmethod
        def speeds(seq):
            if seq not in speed_cache:
                speed_cache[seq] = db.ego_speeds(by_name[seq])
            return speed_cache[seq]

    return NuScenesAdapter
