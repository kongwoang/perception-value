#!/usr/bin/env python
"""Task 5 B2-B4: nearest CAM_F0 image per state iteration, matched projection, branch tracks, checks.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44).  For every Track B state and each of its four
history-buffer iterations:

  * the nearest fetched CAM_F0 image by timestamp (|dt| reported, > 50 ms flagged);
  * every tracked object's lidar box projected into that image: global -> ego at the image's ego pose ->
    camera by the inverse extrinsic (the devkit's boxes_lidar_to_img chain), corners clipped at
    z = 0.1 m, normalised coordinates clamped to 1.25x the undistorted image border, then the DB
    distortion (k1, k2, p1, p2, k3) and intrinsic;
  * per matching configuration, Hungarian assignment on IoU between class-compatible detections and
    projected boxes of eligible in-camera tracks (vehicle, pedestrian, bicycle);
  * per branch, the eligible in-camera tracks to remove and the false-positive agents to add (flat
    ground lift from the box bottom centre, class median size from train+val tracks, zero velocity).

Writes the branch specification for 115, a per-object table for the recall checks, the per-state
summary, the projection checks and 20 overlays.  Run with scripts/pynuplan.
"""
from __future__ import annotations

import argparse, json, pickle, sqlite3, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (   # noqa: E402
    NuPlanScenarioBuilder)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter        # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential      # noqa: E402

from rap.cache import DetCache                                                     # noqa: E402
from rap.detector_model import NUPLAN_TO_COARSE, DetectorMissModel                 # noqa: E402
from rap.nuplan_perception import PerceptionFilter                                 # noqa: E402
from rap.paths import CACHE, RESULTS                                              # noqa: E402

DB_DIR = Path("/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini")
IMG_ROOT = Path("/home/kongwoang/datasets/nuplan/sensor_blobs_cam_f0")
WORK = ROOT / "results" / "raw" / "nuplan_task5"
DET = Path(CACHE) / "nuplan_det"
OUTC = Path(CACHE) / "nuplan_real"
OVERLAYS = ROOT / "results" / "final" / "nuplan_real_perception_overlays"
BUFFER = 4
W, H = 1920, 1080
NEAR, CLAMP = 0.1, 1.25
FP_MIN_AHEAD, FP_MAX_RANGE = 0.5, 80.0
DT_FLAG_US = 50_000
ELIGIBLE = ("vehicle", "pedestrian", "bicycle")
COMPAT = {"vehicle": {"vehicle"}, "pedestrian": {"person"}, "bicycle": {"cyclist", "person"}}
DET_TO_CATEGORY = {"vehicle": "vehicle", "person": "pedestrian", "cyclist": "bicycle"}
MODE_CACHE = {"cheap": "ns_cheap_320", "full": "ns_full_640"}
# (mode, threshold key, IoU) -> matching configuration; branches are built from these
COMBOS = {"cheap_025_iou30": ("cheap", "0.25", 0.3), "full_025_iou30": ("full", "0.25", 0.3),
          "full_s1_iou30": ("full", "s1", 0.3), "cheap_025_iou50": ("cheap", "0.25", 0.5),
          "full_025_iou50": ("full", "0.25", 0.5)}
BRANCHES = {"cheap": ("cheap_025_iou30", True), "full": ("full_025_iou30", True),
            "cheap_nofp": ("cheap_025_iou30", False), "full_nofp": ("full_025_iou30", False),
            "full_s1": ("full_s1_iou30", True),
            "cheap_iou50": ("cheap_025_iou50", True), "full_iou50": ("full_025_iou50", True)}
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if int(np.sum(CORNER_SIGNS[i] != CORNER_SIGNS[j])) == 1]


def quat_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_yaw(w, x, y, z):
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Camera:
    """CAM_F0 of one log: extrinsic (camera -> ego), intrinsic, Caltech/OpenCV distortion."""

    def __init__(self, log: str):
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        rows = con.execute("select translation, rotation, intrinsic, distortion, width, height from camera "
                           "where channel = 'CAM_F0'").fetchall()
        con.close()
        assert len(rows) == 1, f"{log}: {len(rows)} CAM_F0 rows"
        t, q, K, D, w, h = rows[0]
        assert (w, h) == (W, H)
        t, q, K, D = (pickle.loads(t), pickle.loads(q), pickle.loads(K), pickle.loads(D))
        self.t = np.array([float(v) for v in t])
        self.R = quat_R(*[float(v) for v in q])
        self.K = np.array(K, float)
        self.D = np.array(D, float)
        border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2],
                           [W / 2, 0], [W / 2, H - 1]], np.float64)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        self.bx, self.by = CLAMP * float(np.abs(und[:, 0]).max()), CLAMP * float(np.abs(und[:, 1]).max())

    def monotone(self):
        k1, k2, _, _, k3 = self.D
        r = np.linspace(0, np.hypot(self.bx, self.by), 2000)
        deriv = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
        return bool((deriv > 0).all()), float(r[-1]), float(deriv.min())

    def distort(self, xn, yn):
        k1, k2, p1, p2, k3 = self.D
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]

    def project_box(self, pc):
        """pc: (8, 3) corners in the camera frame -> clipped (x1, y1, x2, y2) or None."""
        pts = [p for p in pc if p[2] > NEAR]
        for i, j in EDGES:
            zi, zj = pc[i, 2], pc[j, 2]
            if (zi - NEAR) * (zj - NEAR) < 0:
                a = (NEAR - zi) / (zj - zi)
                pts.append(pc[i] + a * (pc[j] - pc[i]))
        if not pts:
            return None
        P = np.asarray(pts)
        u, v = self.distort(np.clip(P[:, 0] / P[:, 2], -self.bx, self.bx),
                            np.clip(P[:, 1] / P[:, 2], -self.by, self.by))
        x1, y1, x2, y2 = max(u.min(), 0.0), max(v.min(), 0.0), min(u.max(), W - 1.0), min(v.max(), H - 1.0)
        if x2 <= x1 or y2 <= y1:
            return None
        return (float(x1), float(y1), float(x2), float(y2))

    def ground_point(self, u, v, ground_z):
        """Pixel -> ray -> ego-frame ground plane z = ground_z; returns ego (x, y) and the horizontal ray, or None.

        The nuPlan ego frame's origin is the rear axle, about 0.3 m above the road, so the plane is not z = 0
        (RESEARCH_LOG, Task 5 amendment): ground_z is the train+val median bottom of vehicle boxes.
        """
        xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.D).reshape(2)
        d = self.R @ np.array([xn, yn, 1.0])
        if d[2] >= -1e-9:
            return None
        s = (ground_z - self.t[2]) / d[2]
        if s <= 0:
            return None
        g = self.t + s * d
        return g[:2], d[:2] / max(np.hypot(d[0], d[1]), 1e-9)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def hungarian(obj_boxes, obj_cat, det_boxes, det_cls, thr):
    """Max-IoU assignment over class-compatible pairs with IoU >= thr. Returns {obj_i: (det_j, iou)}."""
    iou = iou_matrix(obj_boxes, det_boxes)
    if not iou.size:
        return {}
    ok = np.array([[d in COMPAT[c] for d in det_cls] for c in obj_cat], bool)
    w = np.where(ok & (iou >= thr), iou, 0.0)
    r, c = linear_sum_assignment(-w)
    return {int(i): (int(j), float(w[i, j])) for i, j in zip(r, c) if w[i, j] > 0}


def size_priors(logs):
    rows = []
    for log in logs:
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        rows += con.execute("select c.name, t.length, t.width, t.height from track t "
                            "join category c on t.category_token = c.token").fetchall()
        con.close()
    d = pd.DataFrame(rows, columns=["category", "length", "width", "height"])
    return {c: [float(d[d.category == c][k].median()) for k in ("length", "width", "height")] + [int((d.category == c).sum())]
            for c in ELIGIBLE}


def ground_height(logs):
    """Median bottom height of vehicle boxes 3-30 m ahead (|lateral| < 10 m) in the ego frame, over every
    20th lidar sweep of the benchmark scenario windows of `logs` (train+val only)."""
    meta = pd.read_csv(ROOT / "configs" / "benchmark_nuplan_scenarios.csv")
    zb = []
    for log in logs:
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        for _, s in meta[meta.log == log].iterrows():
            pcs = con.execute("select token, ego_pose_token from lidar_pc where timestamp between ? and ? "
                              "order by timestamp", (int(s.t0 * 1e6), int(s.t1 * 1e6))).fetchall()[::20]
            for tok, ept in pcs:
                ep = con.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token = ?", (ept,)).fetchone()
                Re, te = quat_R(*ep[3:]), np.array(ep[:3])
                for x, y, z, h in con.execute(
                        "select lb.x, lb.y, lb.z, lb.height from lidar_box lb join track t on lb.track_token = t.token "
                        "join category c on t.category_token = c.token where lb.lidar_pc_token = ? and c.name = 'vehicle'",
                        (tok,)):
                    pe = (np.array([x, y, z]) - te) @ Re
                    if 3 < pe[0] < 30 and abs(pe[1]) < 10:
                        zb.append(pe[2] - h / 2)
        con.close()
    zb = np.asarray(zb)
    return {"ground_z_ego_m": float(np.median(zb)), "p25_m": float(np.percentile(zb, 25)),
            "p75_m": float(np.percentile(zb, 75)), "n_boxes": int(len(zb)), "logs": len(logs)}


def dashed_rect(img, p1, p2, color, th=2, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash), 1)
        for k in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            e = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, s, e, color, th)


COLORS = {"vehicle": (60, 200, 60), "pedestrian": (0, 165, 255), "bicycle": (255, 140, 0)}


def overlay(path, img_path, objs, dets, matches, title):
    img = cv2.imread(str(img_path))
    for o in objs:
        if o["box"] is None:
            continue
        p1, p2 = (int(o["box"][0]), int(o["box"][1])), (int(o["box"][2]), int(o["box"][3]))
        if o["eligible"]:
            cv2.rectangle(img, p1, p2, COLORS[o["category"]], 3)
        else:
            dashed_rect(img, p1, p2, (170, 170, 170), 2)
    for mode, color in (("cheap", (40, 40, 230)), ("full", (230, 40, 230))):
        boxes, cls = dets[mode]
        for b, c in zip(boxes, cls):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2 if mode == "cheap" else 1)
        for oi, (dj, iou) in matches[mode].items():
            ob, db = objs[oi]["box"], boxes[dj]
            cv2.line(img, (int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2)),
                     (int((db[0] + db[2]) / 2), int((db[1] + db[3]) / 2)), color, 2)
    cv2.rectangle(img, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "projected: vehicle green, pedestrian orange, bicycle blue, static grey dashed | "
                     "detections@0.25: 320 red (thick), 640 magenta (thin), lines = matches (IoU>=0.3)",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(path), cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/kongwoang/datasets/nuplan")
    ap.add_argument("--map_root", default="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0")
    ap.add_argument("--n_overlays", type=int, default=20)
    args = ap.parse_args()
    t_start = time.time()

    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuplan"]
    split_of = {l: k for k in ("train", "val", "test") for l in splits[k]}
    det_summary = json.loads((WORK / "detect_summary.json").read_text())
    thresholds = {"0.25": 0.25, "s1": float(det_summary["s1_full_threshold"])}
    priors = size_priors(sorted(splits["train"] + splits["val"]))
    ground = ground_height(sorted(splits["train"] + splits["val"]))
    ground_z = ground["ground_z_ego_m"]
    print(f"  ground plane z in the ego frame (train+val): {ground}", flush=True)
    print(f"  S1 FULL threshold {thresholds['s1']:.4f}; size priors {priors}", flush=True)

    idx = pd.read_csv(WORK / "image_index.csv.gz")
    raw = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_idm_raw.csv")
    states = raw[["scenario", "log", "iteration"]].drop_duplicates()
    assert len(states) == 1440

    builder = NuPlanScenarioBuilder(
        data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60,
                         timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                         expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert set(scen) == set(states.scenario), "scenario set differs from Track B"

    z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
    model = DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"])
    geomf = PerceptionFilter(model)
    rng = np.random.default_rng(0)
    pick = set(map(tuple, states[["scenario", "iteration"]].to_numpy()[
        rng.choice(len(states), args.n_overlays, replace=False)].tolist()))
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    OUTC.mkdir(parents=True, exist_ok=True)

    cams, dets_cache, all_img = {}, {}, {}
    spec, obj_rows, state_rows, dt_rows, iou_rows = {}, [], [], [], []
    fp_dropped = {b: {"no_ground_hit": 0, "not_ahead": 0, "beyond_range": 0} for b in BRANCHES}
    ground_check, overlay_files, lift_err = [], [], []
    for n_sc, (tok, g) in enumerate(states.groupby("scenario", sort=True), 1):
        sc = scen[tok]
        log = sc.log_name
        if log not in cams:
            cams[log] = Camera(log)
            dets_cache[log] = {m: DetCache(DET / MODE_CACHE[m] / f"{log}.npz") for m in MODE_CACHE}
            con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
            all_img[log] = np.array(sorted(r[0] for r in con.execute(
                "select i.timestamp from image i join camera c on i.camera_token = c.token "
                "where c.channel = 'CAM_F0'")), np.int64)
            con.close()
        cam, li = cams[log], idx[idx.log == log].sort_values("timestamp")
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        its = sorted(int(i) for i in g.iteration)
        need_k = sorted({k for it in its for k in range(it - BUFFER + 1, it + 1)})
        for k in need_k:
            t_k = sc.get_time_point(k).time_us
            ts = li.timestamp.to_numpy()
            j = int(np.argmin(np.abs(ts - t_k)))
            im = li.iloc[j]
            j_all = int(np.argmin(np.abs(all_img[log] - t_k)))
            dt_rows.append({"scenario": tok, "iteration": k, "decision": k in its, "dt_us": int(im.timestamp - t_k),
                            "nearest_in_db_fetched": bool(all_img[log][j_all] == im.timestamp)})
            ep = con.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token = ?",
                             (bytes.fromhex(im.ego_pose_token),)).fetchone()
            Re, te, yaw_e = quat_R(*ep[3:]), np.array(ep[:3]), quat_yaw(*ep[3:])
            lb = {r[0].hex(): r[1:] for r in con.execute(
                "select token, x, y, z, yaw, width, length, height from lidar_box where lidar_pc_token = ?",
                (bytes.fromhex(sc._lidarpc_tokens[k]),))}
            objs = list(sc.get_tracked_objects_at_iteration(k).tracked_objects)
            ego = sc.get_ego_state_at_iteration(k)
            dist, lat, coarse, fwd = geomf._geometry(objs, ego) if objs else (np.zeros(0),) * 4
            recs = []
            for oi, o in enumerate(objs):
                cat = str(o.tracked_object_type.name).lower()
                x, y, zc, yaw, wd, ln, ht = lb[o.metadata.token]
                cg = np.array([x, y, zc]) + (CORNER_SIGNS * [ln / 2, wd / 2, ht / 2]) @ np.array(
                    [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]).T
                pe = (cg - te) @ Re                       # ego frame (rows: R^T p)
                pcam = (pe - cam.t) @ cam.R               # camera frame
                centre_cam = ((np.array([x, y, zc]) - te) @ Re - cam.t) @ cam.R
                box = cam.project_box(pcam) if centre_cam[2] > 0 else None
                recs.append({"token": o.metadata.token, "category": cat, "eligible": cat in ELIGIBLE,
                             "box": box, "in_cam": box is not None,
                             "height_px": (box[3] - box[1]) if box else 0.0,
                             "dist": float(dist[oi]), "lat": float(lat[oi]), "coarse": str(coarse[oi])})
                if cat == "vehicle" and box is not None and float(dist[oi]) < 30:
                    ground_check.append(float(pe[:, 2].min()))
                if k in its and cat == "vehicle" and box is not None and box[3] < H - 2 and float(dist[oi]) < 60:
                    # the FP lift applied to a TRUE projected box: how far from the true centre it lands
                    gp = cam.ground_point(0.5 * (box[0] + box[2]), box[3], ground_z)
                    if gp is not None:
                        (gx, gy), hray = gp
                        est = np.array([gx, gy]) + hray * priors["vehicle"][0] / 2
                        tru = ((np.array([x, y, zc]) - te) @ Re)[:2]
                        lift_err.append((float(np.hypot(*(est - tru))), float(np.hypot(*est) - np.hypot(*tru)),
                                         float(np.hypot(*tru))))
            frame = int(im.frame)
            dd = {m: dets_cache[log][m].det(dets_cache[log][m].index[frame]) for m in MODE_CACHE}
            elig = [i for i, r in enumerate(recs) if r["eligible"] and r["in_cam"]]
            matches, unmatched = {}, {}
            for cname, (mode, tkey, iou_thr) in COMBOS.items():
                d = dd[mode]
                keep = d["conf"] >= thresholds[tkey]
                kidx = np.flatnonzero(keep)
                boxes, cls = d["xyxy"][keep], d["coarse"][keep]
                mt = hungarian([recs[i]["box"] for i in elig], [recs[i]["category"] for i in elig], boxes, cls, iou_thr)
                matches[cname] = {elig[i]: v for i, v in mt.items()}
                used = {v[0] for v in mt.values()}
                unmatched[cname] = [(boxes[jj], cls[jj], int(kidx[jj])) for jj in range(len(boxes)) if jj not in used]
                if k in its:
                    iou_rows += [{"combo": cname, "iou": v[1]} for v in mt.values()]
            for bname, (cname, with_fp) in BRANCHES.items():
                removed = frozenset(recs[i]["token"] for i in elig if i not in matches[cname])
                fps = []
                if with_fp:
                    fmode = COMBOS[cname][0]
                    for b, c, kd in unmatched[cname]:
                        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], ground_z)
                        if gp is None:
                            fp_dropped[bname]["no_ground_hit"] += 1
                            continue
                        (gx, gy), hray = gp
                        if gx < FP_MIN_AHEAD:
                            fp_dropped[bname]["not_ahead"] += 1
                            continue
                        if np.hypot(gx, gy) > FP_MAX_RANGE:
                            fp_dropped[bname]["beyond_range"] += 1
                            continue
                        cat = DET_TO_CATEGORY[str(c)]
                        L, Wd, Ht, _ = priors[cat]
                        ce = np.array([gx, gy, 0.0]) + np.array([hray[0], hray[1], 0.0]) * L / 2
                        cgl = Re @ ce + te
                        # token per (mode, image, cached detection): the same detection is the same agent in
                        # every branch, so identical branches can share one planner call
                        fps.append((f"fp:{fmode}:{im.image_token}:{kd}", cat, float(cgl[0]), float(cgl[1]),
                                    yaw_e, float(L), float(Wd), float(Ht)))
                spec[(tok, k, bname)] = (removed, fps)
            spec[(tok, k, "identity")] = (frozenset(), [])
            if k in its:
                row = {"scenario": tok, "log": log, "split": split_of[log], "iteration": k,
                       "image_token": im.image_token, "dt_ms": (im.timestamp - t_k) / 1e3,
                       "n_tracks": len(recs), "n_eligible_in_cam": len(elig),
                       "n_static_in_cam": sum(1 for r in recs if r["in_cam"] and not r["eligible"])}
                for bname in BRANCHES:
                    rem, fps = spec[(tok, k, bname)]
                    row[f"kept_{bname}"] = len(elig) - len(rem)
                    row[f"fp_{bname}"] = len(fps)
                state_rows.append(row)
                for i in elig:
                    r = recs[i]
                    obj_rows.append({"scenario": tok, "log": log, "iteration": k, "token": r["token"],
                                     "category": r["category"], "dist": r["dist"], "lat": r["lat"],
                                     "height_px": r["height_px"],
                                     **{f"hit_{c}": i in matches[c] for c in COMBOS},
                                     **{f"iou_{c}": matches[c][i][1] if i in matches[c] else 0.0 for c in COMBOS}})
                if (tok, k) in pick:
                    f = OVERLAYS / f"{len(overlay_files):02d}_{log}_{tok}_{k}.jpg"
                    ov_d = {m: (dd[m]["xyxy"][dd[m]["conf"] >= 0.25], dd[m]["coarse"][dd[m]["conf"] >= 0.25])
                            for m in MODE_CACHE}
                    overlay(f, IMG_ROOT / im.filename_jpg, recs, ov_d,
                            {"cheap": matches["cheap_025_iou30"], "full": matches["full_025_iou30"]},
                            f"{log} | scenario {tok} | iteration {k} | dt {(im.timestamp - t_k) / 1e3:+.1f} ms | "
                            f"eligible in camera {len(elig)}, matched 320 {len(matches['cheap_025_iou30'])}, "
                            f"640 {len(matches['full_025_iou30'])}")
                    overlay_files.append(f.name)
        con.close()
        print(f"  [{n_sc:2d}/60] {tok} {log}: {len(its)} states, {len(need_k)} iterations, "
              f"{time.time() - t_start:6.0f}s", flush=True)

    with open(OUTC / "branches.pkl", "wb") as f:
        pickle.dump({"spec": spec, "branches": list(BRANCHES) + ["identity"], "thresholds": thresholds,
                     "priors": priors}, f)
    objs = pd.DataFrame(obj_rows)
    objs.to_csv(OUTC / "objects_decision.csv.gz", index=False)          # the edge env has no pyarrow
    st = pd.DataFrame(state_rows)
    st.to_csv(WORK / "branch_state_summary.csv.gz", index=False)
    dt = pd.DataFrame(dt_rows)
    ious = pd.DataFrame(iou_rows)

    mono = {log: cams[log].monotone() for log in cams}
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    checks = {
        "states": len(st), "iterations": len(dt), "scenarios": int(st.scenario.nunique()), "logs": int(st.log.nunique()),
        "s1_full_threshold": thresholds["s1"], "size_priors_train_val_median_LWH_n": priors,
        "dt": {"abs_ms_quantiles": dict(zip(map(str, q), (np.quantile(np.abs(dt.dt_us), q) / 1e3).round(2).tolist())),
               "n_iterations_over_50ms": int((np.abs(dt.dt_us) > DT_FLAG_US).sum()),
               "n_decision_iterations_over_50ms": int(((np.abs(dt.dt_us) > DT_FLAG_US) & dt.decision).sum()),
               "n_states_any_buffer_image_over_50ms": int(dt.assign(f=np.abs(dt.dt_us) > DT_FLAG_US)
                                                          .groupby("scenario").f.sum().gt(0).sum()),
               "nearest_db_image_not_fetched": int((~dt.nearest_in_db_fetched).sum())},
        "distortion_monotone_to_clamp": {l: {"monotone": m[0], "r_max": round(m[1], 4), "min_derivative": round(m[2], 4)}
                                         for l, m in mono.items()},
        "all_distortion_monotone": bool(all(m[0] for m in mono.values())),
        "ground_plane_vehicle_bottom_z_ego_within_30m": {
            "n": len(ground_check), "median_m": float(np.median(ground_check)) if ground_check else None,
            "p25_m": float(np.percentile(ground_check, 25)) if ground_check else None,
            "p75_m": float(np.percentile(ground_check, 75)) if ground_check else None},
        "fp_lift_ground_plane_train_val": {**ground, "camera_height_above_ground_m":
                                           float(np.mean([c.t[2] for c in cams.values()]) - ground_z)},
        "fp_lift_on_true_vehicle_boxes_decision_iterations": (lambda e: {
            "n": len(e), "xy_error_m_quantiles": dict(zip(map(str, q), np.quantile(e[:, 0], q).round(2).tolist())),
            "range_error_m_quantiles": dict(zip(map(str, q), np.quantile(e[:, 1], q).round(2).tolist())),
            "median_abs_range_error_by_band": {f"{a}-{b}m": float(np.median(np.abs(e[(e[:, 2] >= a) & (e[:, 2] < b), 1])))
                                               for a, b in ((0, 15), (15, 30), (30, 60))
                                               if ((e[:, 2] >= a) & (e[:, 2] < b)).any()}})(np.asarray(lift_err)),
        "match_iou_quantiles": {c: dict(zip(map(str, q), np.quantile(ious[ious.combo == c].iou, q).round(3).tolist()))
                                for c in COMBOS if (ious.combo == c).any()},
        "match_counts": ious.combo.value_counts().to_dict(),
        "fp_dropped_by_reason_all_iterations": fp_dropped,
        "per_state_means": st.drop(columns=["scenario", "log", "split", "image_token"]).mean().round(3).to_dict(),
        "overlays": overlay_files,
        "seconds": time.time() - t_start,
    }
    (WORK / "checks_projection.json").write_text(json.dumps(checks, indent=1, default=float))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("distortion_monotone_to_clamp",)}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
