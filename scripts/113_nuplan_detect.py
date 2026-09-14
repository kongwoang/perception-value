#!/usr/bin/env python
"""Task 5 B1: YOLOv8s 320/640 TensorRT detections on the fetched nuPlan CAM_F0 scenario-window images.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44).  The engines are the nuScenes pair, 192x320 and
384x640: their 16:9 input matches CAM_F0's 1920x1080 exactly, as it matches nuScenes CAM_FRONT.  Detection
runs through the existing `TwoFidelityDetector` TensorRT path at conf >= 0.10, NMS IoU 0.65, max 100, and
is cached in the `rap.cache` format, one npz per (mode, log), frames in timestamp order.

Per-frame preprocess / inference / postprocess times are CUDA-synchronised and stored as frame scalars;
JPEG decode is timed separately.  Energy comes from a dedicated pass per mode over 300 frames (100 decoded
images, three times) with CPU and GPU rail power sampled at 20 Hz, less an idle baseline.
"""
from __future__ import annotations

import argparse, json, sqlite3, sys, threading, time, urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import cache, power                                                   # noqa: E402
from rap.detect import MODES, TwoFidelityDetector                              # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

DB_DIR = Path("/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini")
IMG_ROOT = Path("/home/kongwoang/datasets/nuplan/sensor_blobs_cam_f0")
WORK = ROOT / "results" / "raw" / "nuplan_task5"
MODE_NAMES = ("ns_cheap_320", "ns_full_640")
OUT = Path(CACHE) / "nuplan_det"


def image_index() -> pd.DataFrame:
    plan = pd.read_csv(WORK / "plan_members.csv.gz")
    rows = []
    for log, g in plan.groupby("log"):
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        q = con.execute("select i.token, i.filename_jpg, i.timestamp, i.ego_pose_token from image i "
                        "join camera c on i.camera_token = c.token where c.channel = 'CAM_F0'").fetchall()
        con.close()
        by_name = {fn: (tok.hex(), int(ts), ep.hex()) for tok, fn, ts, ep in q}
        for fn, split in zip(g.filename_jpg, g.split):
            tok, ts, ep = by_name[fn]
            rows.append({"log": log, "split": split, "filename_jpg": fn, "image_token": tok,
                         "timestamp": ts, "ego_pose_token": ep})
    d = pd.DataFrame(rows).sort_values(["log", "timestamp"]).reset_index(drop=True)
    d["frame"] = d.groupby("log").cumcount()
    return d


def throttle():
    try:
        s = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/jetson/status", timeout=5).read())
        return {"throttle": s.get("throttle"), "temperatures": s.get("temperatures"), "fan": s.get("fan")}
    except Exception as e:                                                     # noqa: BLE001
        return {"error": str(e)}


def timed_detect(det, img, mode):
    sync = torch.cuda.synchronize
    sync(); t0 = time.perf_counter()
    x, r, pad, _ = det.preprocess(img, mode)
    sync(); t1 = time.perf_counter()
    raw = det.forward(x, mode)
    sync(); t2 = time.perf_counter()
    d = det.postprocess(raw, mode, r, pad, img.shape)
    sync(); t3 = time.perf_counter()
    return d, (1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2))


def sample_power(stop, out):
    while not stop.is_set():
        out.append((time.perf_counter(), power.read_power_mw()))
        time.sleep(0.05)


def energy_pass(det, idx, n_unique=100, repeats=3):
    imgs = [cv2.imread(str(IMG_ROOT / f)) for f in idx.filename_jpg.iloc[:: max(len(idx) // n_unique, 1)][:n_unique]]
    res = {"power_available": power.AVAILABLE}
    if not power.AVAILABLE:
        return res
    stop, idle = threading.Event(), []
    th = threading.Thread(target=sample_power, args=(stop, idle), daemon=True); th.start()
    time.sleep(10.0); stop.set(); th.join()
    rails = sorted(idle[0][1])
    base = {k: float(np.mean([s[k] for _, s in idle if k in s])) for k in rails}
    res["idle_mw"] = base
    for m in MODE_NAMES:
        mode = MODES[m]
        for _ in range(10):
            timed_detect(det, imgs[0], mode)
        stop, samples = threading.Event(), []
        th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True); th.start()
        t0 = time.perf_counter()
        for _ in range(repeats):
            for im in imgs:
                timed_detect(det, im, mode)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        stop.set(); th.join()
        n = repeats * len(imgs)
        mean = {k: float(np.mean([s[k] for _, s in samples if k in s])) for k in rails}
        excess = {k: mean[k] - base[k] for k in rails}
        cg = excess.get("CPU", 0.0) + excess.get("GPU", 0.0)
        res[m] = {"frames": n, "seconds": el, "ms_per_frame": 1e3 * el / n, "mean_mw": mean,
                  "excess_mw": excess, "cpu_gpu_excess_mw": cg, "cpu_gpu_mj_per_frame": cg * el / n}
        print(f"  energy {m}: {1e3 * el / n:.1f} ms/frame, CPU+GPU excess {cg / 1e3:.2f} W, "
              f"{cg * el / n:.1f} mJ/frame", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/kongwoang/models/yolo/yolov8s.pt")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    idx = image_index()
    missing = [f for f in idx.filename_jpg if not (IMG_ROOT / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} planned images are not on disk, e.g. {missing[:2]}")
    idx.to_csv(WORK / "image_index.csv.gz", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    idx.to_csv(OUT / "index.csv", index=False)
    print(f"  {len(idx):,} images, {idx.log.nunique()} logs", flush=True)

    det = TwoFidelityDetector(args.weights, backend="trt")
    first = cv2.imread(str(IMG_ROOT / idx.filename_jpg.iloc[0]))
    for m in MODE_NAMES:
        for _ in range(20):
            timed_detect(det, first, MODES[m])
    state_before = throttle()

    t_all = time.time()
    for log, g in idx.groupby("log", sort=True):
        paths = {m: OUT / m / f"{log}.npz" for m in MODE_NAMES}
        if not args.force and all(p.exists() for p in paths.values()):
            continue
        store = {m: {"dets": [], "scalars": {}} for m in MODE_NAMES}
        t0 = time.time()
        for fn in g.filename_jpg:
            a = time.perf_counter()
            img = cv2.imread(str(IMG_ROOT / fn))
            dec = 1e3 * (time.perf_counter() - a)
            assert img is not None and img.shape == (1080, 1920, 3), fn
            for m in MODE_NAMES:
                d, (pre, inf, post) = timed_detect(det, img, MODES[m])
                store[m]["dets"].append(d)
                sc = store[m]["scalars"]
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    sc.setdefault(k, []).append(d[k])
                for k, v in (("ms_pre", pre), ("ms_inf", inf), ("ms_post", post), ("ms_decode", dec)):
                    sc.setdefault(k, []).append(v)
        for m in MODE_NAMES:
            cache.save(paths[m], g.frame.tolist(), store[m]["dets"], None, store[m]["scalars"])
        nd = {m: int(sum((d["conf"] >= 0.25).sum() for d in store[m]["dets"])) for m in MODE_NAMES}
        print(f"  [{log}] {len(g):4d} frames {time.time() - t0:6.1f}s dets@0.25 {nd}", flush=True)

    summary = {"images": len(idx), "logs": int(idx.log.nunique()), "seconds_main_pass": time.time() - t_all,
               "engines": {m: str(Path(det.engine_dir) / f"yolov8s_{m}.engine") for m in MODE_NAMES},
               "state_before": state_before}
    for m in MODE_NAMES:
        z = [cache.DetCache(OUT / m / f"{log}.npz") for log in sorted(idx.log.unique())]
        conf = np.concatenate([c.z["conf"] for c in z])
        ms = {k: np.concatenate([c.z[f"fs_{k}"] for c in z]) for k in ("ms_pre", "ms_inf", "ms_post", "ms_decode")}
        tot = ms["ms_pre"] + ms["ms_inf"] + ms["ms_post"]
        summary[m] = {"frames": int(sum(len(c) for c in z)), "dets_conf010": int(len(conf)),
                      "dets_conf025": int((conf >= 0.25).sum()),
                      **{f"{k}_median": float(np.median(v)) for k, v in ms.items()},
                      "ms_total_median": float(np.median(tot)), "ms_total_p90": float(np.percentile(tot, 90))}
    # sensitivity (ii): the Task 1 S1 rule, FULL threshold count-matched to CHEAP at 0.25 on train+val logs
    from rap.tables import match_detection_counts
    tv = sorted(idx[idx.split.isin(["train", "val"])].log.unique())
    summary["s1_full_threshold"] = float(match_detection_counts(
        [cache.DetCache(OUT / MODE_NAMES[0] / f"{l}.npz") for l in tv],
        [cache.DetCache(OUT / MODE_NAMES[1] / f"{l}.npz") for l in tv], 0.25))
    summary["s1_logs"] = tv
    print(f"  S1 FULL threshold on {len(tv)} train+val logs: {summary['s1_full_threshold']:.4f}", flush=True)
    summary["energy"] = energy_pass(det, idx)
    summary["state_after"] = throttle()
    (WORK / "detect_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: summary[m] for m in MODE_NAMES}, indent=1))


if __name__ == "__main__":
    main()
