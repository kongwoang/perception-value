#!/usr/bin/env python
"""Run both perception fidelities over nuScenes CAM_FRONT keyframes."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import cache, features, mono, runmeta       # noqa: E402
from rap.detect import MODES, TwoFidelityDetector    # noqa: E402
from rap.nusc import FRAME_DT, NuScenesDB            # noqa: E402
from rap.paths import CACHE                          # noqa: E402

CAM_HEIGHT = 1.51   # nuScenes CAM_FRONT height above the road [m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--weights", default="/home/kongwoang/models/yolo/yolov8s.pt")
    ap.add_argument("--cheap", default="ns_cheap_320")
    ap.add_argument("--full", default="ns_full_640")
    ap.add_argument("--scenes", type=int, default=0, help="limit number of scenes")
    ap.add_argument("--out", default=str(CACHE / "nusc_det"))
    ap.add_argument("--engine_stem", default=None)
    args = ap.parse_args()

    run = runmeta.new_run("nusc_detect", vars(args))
    db = NuScenesDB(args.dataroot, args.version)
    det = TwoFidelityDetector(args.weights, backend="trt", engine_stem=args.engine_stem)
    modes = [args.cheap, args.full]
    outdir = Path(args.out)
    scenes = db.scenes[: args.scenes] if args.scenes else db.scenes

    total_f, total_t = 0, 0.0
    kept = []
    for sc in scenes:
        toks = db.samples(sc)
        paths = [db.image_path(t) for t in toks]
        if not all(p.exists() for p in paths):
            continue                       # blob for this scene not downloaded
        name = db.scene_name(sc)
        store = {m: {"dets": [], "geos": [], "scalars": {}} for m in modes}
        prev_det = {m: None for m in modes}
        prev_small = None
        calib = db.calib(toks[0])
        t0 = time.time()
        for fi, tk in enumerate(toks):
            img = cv2.imread(str(db.image_path(tk)))
            small = cv2.cvtColor(cv2.resize(img, (MODES[args.cheap].net_w,
                                                  MODES[args.cheap].net_h)), cv2.COLOR_BGR2GRAY)
            for m in modes:
                d, _ = det.detect(img, MODES[m])
                g = mono.predicted_geometry(d, prev_det[m], calib, CAM_HEIGHT, FRAME_DT)
                store[m]["dets"].append(d)
                store[m]["geos"].append(g)
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    store[m]["scalars"].setdefault(k, []).append(d[k])
                prev_det[m] = d
            for k, v in features.image_stats(small, prev_small).items():
                store[args.cheap]["scalars"].setdefault(k, []).append(v)
            store[args.cheap]["scalars"].setdefault("img_h", []).append(img.shape[0])
            store[args.cheap]["scalars"].setdefault("img_w", []).append(img.shape[1])
            prev_small = small
        dt = time.time() - t0
        total_f += len(toks); total_t += dt
        kept.append(name)
        for m in modes:
            cache.save(outdir / m / f"{name}.npz", list(range(len(toks))),
                       store[m]["dets"], store[m]["geos"], store[m]["scalars"])
        nd = {m: int(sum(len(d["conf"]) for d in store[m]["dets"])) for m in modes}
        print(f"[{name}] {len(toks):3d} frames {dt:6.1f}s dets={nd}")

    (run / "summary.json").write_text(json.dumps(
        {"version": args.version, "scenes": kept, "frames": total_f, "seconds": total_t,
         "modes": modes}, indent=2))
    print(f"\n{len(kept)} scenes, {total_f} frames in {total_t:.0f}s -> {outdir}")
    print("wrote", run)


if __name__ == "__main__":
    main()
