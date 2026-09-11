#!/usr/bin/env python
"""Run both perception fidelities over every KITTI tracking frame and cache the results.

Each image is decoded once and fed to both modes, so CHEAP and FULL see byte-identical
input. Monocular geometry and image statistics are computed in the same causal pass
(previous frame only) so the feature stage never needs to look forward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import cache, features, kitti, mono, runmeta   # noqa: E402
from rap.detect import MODES, TwoFidelityDetector       # noqa: E402
from rap.paths import CACHE                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/kongwoang/models/yolo/yolov8s.pt")
    ap.add_argument("--backend", default="trt", choices=["torch", "trt"])
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--extra_modes", nargs="*", default=[])
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="frames per sequence (0 = all)")
    ap.add_argument("--out", default=str(CACHE / "det"))
    ap.add_argument("--engine_stem", default=None,
                    help="engine filename stem if it differs from the weights stem")
    args = ap.parse_args()

    modes = [args.cheap, args.full] + list(args.extra_modes)
    run = runmeta.new_run("detect", vars(args))
    det = TwoFidelityDetector(args.weights, backend=args.backend,
                              engine_stem=args.engine_stem)
    seqs = args.sequences or kitti.sequences()
    outdir = Path(args.out)

    totals = {"frames": 0, "t": 0.0}
    for seq in seqs:
        fids = kitti.frame_ids(seq)
        if args.limit:
            fids = fids[: args.limit]
        if len(fids) == 0:
            print(f"[{seq}] no images on disk, skipping")
            continue
        store = {m: {"dets": [], "geos": [], "scalars": {}} for m in modes}
        calib = kitti.load_calib(seq)
        prev_det = {m: None for m in modes}
        prev_small = None
        t0 = time.time()

        for fi, fid in enumerate(fids):
            img = cv2.imread(str(kitti.image_path(seq, fid)))
            if img is None:
                raise FileNotFoundError(kitti.image_path(seq, fid))
            small = cv2.cvtColor(cv2.resize(img, (MODES[args.cheap].net_w, MODES[args.cheap].net_h)),
                                 cv2.COLOR_BGR2GRAY)
            for m in modes:
                d, _ = det.detect(img, MODES[m])
                g = mono.predicted_geometry(d, prev_det[m], calib)
                store[m]["dets"].append(d)
                store[m]["geos"].append(g)
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    store[m]["scalars"].setdefault(k, []).append(d[k])
                prev_det[m] = d
            stats = features.image_stats(small, prev_small)
            for k, v in stats.items():
                store[args.cheap]["scalars"].setdefault(k, []).append(v)
            store[args.cheap]["scalars"].setdefault("img_h", []).append(img.shape[0])
            store[args.cheap]["scalars"].setdefault("img_w", []).append(img.shape[1])
            prev_small = small

        dt = time.time() - t0
        totals["frames"] += len(fids)
        totals["t"] += dt
        for m in modes:
            cache.save(outdir / m / f"{seq}.npz", list(fids), store[m]["dets"],
                       store[m]["geos"], store[m]["scalars"])
        nd = {m: int(sum(len(d["conf"]) for d in store[m]["dets"])) for m in modes}
        print(f"[{seq}] {len(fids):4d} frames  {dt:6.1f}s  ({dt/len(fids)*1e3:5.1f} ms/frame all modes)  dets={nd}")

    (run / "detect_summary.json").write_text(json.dumps(
        {"sequences": seqs, "modes": modes, **totals}, indent=2))
    print(f"\n{totals['frames']} frames in {totals['t']:.0f}s -> {outdir}")
    print("wrote", run)


if __name__ == "__main__":
    main()
