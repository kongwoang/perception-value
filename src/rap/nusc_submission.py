"""Convert our 2D monocular detections into nuScenes 3D detection submissions.

PKL and TIP consume the standard nuScenes detection format: oriented 3D boxes in the
global frame. Our detector emits 2D image boxes, so a lift is unavoidable, and the lift is
a confound those metrics were never designed to absorb. Two variants are therefore built
and reported separately:

  "mono"   — fully monocular. Range from the ground-plane/height-prior fusion, lateral
             offset from the box centre, class-prior extent, and **yaw assumed equal to the
             ego heading** because a single 2D box carries no orientation. Deployed-realistic
             but conflates orientation error with detection difference.

  "oracle" — a detection that matches a ground-truth object at IoU >= 0.5 inherits that
             object's exact translation, size and rotation; unmatched detections keep the
             monocular lift. CHEAP and FULL then differ only in *which objects they found*,
             which is the quantity under study, and PKL sees exact orientation.

`oracle` is the primary variant for the planning-aware baselines. Neither variant changes
any scoring definition in the prior work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .mono import box_iou
from .risk import RiskConfig

# our coarse classes -> nuScenes detection names, with class-prior extents (w, l, h) [m]
COARSE_TO_NUSC = {"vehicle": "car", "person": "pedestrian", "cyclist": "bicycle"}
SIZE_PRIOR = {"car": (1.95, 4.62, 1.73), "pedestrian": (0.67, 0.73, 1.77),
              "bicycle": (0.60, 1.70, 1.28), "truck": (2.51, 6.93, 2.84),
              "bus": (2.94, 11.19, 3.44), "motorcycle": (0.77, 2.11, 1.47)}
# nuScenes category prefix -> detection name, for the oracle variant
CAT_TO_NUSC = {"vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus": "bus",
               "vehicle.trailer": "trailer", "vehicle.construction": "construction_vehicle",
               "human.pedestrian": "pedestrian", "vehicle.motorcycle": "motorcycle",
               "vehicle.bicycle": "bicycle"}
DEFAULT_ATTR = {"car": "vehicle.moving", "truck": "vehicle.moving", "bus": "vehicle.moving",
                "trailer": "vehicle.moving", "construction_vehicle": "vehicle.moving",
                "pedestrian": "pedestrian.moving", "motorcycle": "cycle.with_rider",
                "bicycle": "cycle.with_rider"}


def _ego_to_global(pts_ego: np.ndarray, R_ego: np.ndarray, t_ego: np.ndarray) -> np.ndarray:
    return (R_ego @ np.asarray(pts_ego, float).T).T + t_ego


def build_submission(db, adapter, det_cache, seqs, cfg: RiskConfig,
                     variant: str = "oracle") -> dict:
    """One nuScenes detection submission for one perception mode."""
    results = {}
    name_to_scene = {db.scene_name(s): s for s in db.scenes}
    for name in seqs:
        scene = name_to_scene[name]
        toks = db.samples(scene)
        geom = adapter.geometry(name)
        cache = det_cache[name]
        for i, fr in enumerate(cache.frames):
            fr = int(fr)
            if fr >= len(toks):
                continue
            tok = toks[fr]
            R_ego, t_ego = db._global_to_ego(tok)
            yaw_ego = Rotation.from_matrix(R_ego).as_euler("zyx")[0]

            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt2d = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                    if len(g) else np.zeros((0, 4)))

            d, geo = cache.det(i), cache.geo(i)
            keep = d["conf"] >= cfg.op_conf
            bx = d["xyxy"][keep].astype(float)
            boxes = []
            if len(bx):
                matched = np.full(len(bx), -1, int)
                if variant == "oracle" and len(gt2d):
                    iou = box_iou(bx, gt2d)
                    best = iou.argmax(1)
                    ok = iou.max(1) >= cfg.iou_thr
                    matched[ok] = best[ok]
                ann_by_frame = db.annotations_for(scene, fr) if variant == "oracle" else None
                for k in range(len(bx)):
                    if matched[k] >= 0 and ann_by_frame is not None:
                        a = ann_by_frame.get(int(g["track_id"][matched[k]]))
                        if a is not None:
                            cat = a["category"]
                            dname = next((v for p, v in CAT_TO_NUSC.items()
                                          if cat.startswith(p)), "car")
                            boxes.append({
                                "sample_token": tok,
                                "translation": list(map(float, a["translation"])),
                                "size": list(map(float, a["size"])),
                                "rotation": list(map(float, a["rotation"])),
                                "velocity": [0.0, 0.0],
                                "detection_name": dname,
                                "detection_score": float(d["conf"][keep][k]),
                                "attribute_name": DEFAULT_ATTR.get(dname, ""),
                            })
                            continue
                    # monocular lift
                    dname = COARSE_TO_NUSC.get(str(d["coarse"][keep][k]), "car")
                    w, l, h = SIZE_PRIOR[dname]
                    # geo["z"] is range_ground(y2), the distance to the box's ground contact,
                    # i.e. the object's NEAR face.  nuScenes `translation` is the box centre, so
                    # the centre sits half a length further away.  Writing the near-face range
                    # as the centre put every lifted box ~L/2 too close -- 2.3 m for a car and
                    # 5.6 m for a bus, so the error was class-dependent.  The same geo["z"] is
                    # correct where the braking controller consumes it as a gap to the nearest
                    # point, which is why this was easy to miss.
                    x_ego = float(geo["z"][keep][k]) + l / 2.0
                    y_ego = float((geo["lat_min"][keep][k] + geo["lat_max"][keep][k]) / 2)
                    centre = _ego_to_global(np.array([[x_ego, y_ego, h / 2]]), R_ego, t_ego)[0]
                    q = Rotation.from_euler("z", yaw_ego).as_quat()      # x,y,z,w
                    boxes.append({
                        "sample_token": tok,
                        "translation": [float(v) for v in centre],
                        "size": [w, l, h],
                        "rotation": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                        "velocity": [0.0, 0.0],
                        "detection_name": dname,
                        "detection_score": float(d["conf"][keep][k]),
                        "attribute_name": DEFAULT_ATTR.get(dname, ""),
                    })
            results[tok] = boxes
    return {"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                     "use_map": False, "use_external": False},
            "results": results}


def write_submission(sub: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sub))
    return path
