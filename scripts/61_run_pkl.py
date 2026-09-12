#!/usr/bin/env python
"""Phase 0F Stage 1: run PKL on prebuilt submissions.

Prior-work provenance
  repo   : https://github.com/nv-tlabs/planning-centric-metrics
  commit : f6865f2b473303f2ff01a477bf6de4dce7109742
  scoring: unmodified.

Planner weights: PKL's own Google Drive links (planner.pt, masks_trainval.json) no longer
resolve -- both return a Google sign-in page and gdown reports "cannot retrieve the public
link", so `calculate_pkl`'s auto-download is dead. The weights are instead taken from the
TIP repository (https://github.com/qcraftai/tip, commit e52fd48), which vendors them and
states it is adapted from PKL. Verified: the checkpoint loads into PKL's
`compile_model(cin=5, cout=16, with_skip=True)` with strict=True, 149 keys, zero missing
or unexpected, and the masks are the expected (16, 256, 256) binary array.

One deliberate deviation, for memory rather than science: `load_gt` insists on a named
split and would load ground truth for all 28k trainval samples, which this board cannot
hold. `create_splits_scenes` is patched so the split contains exactly our scenes. The box
construction inside `load_gt` runs unmodified.

Sign convention: lower PKL is better, so G_PKL = PKL_cheap - PKL_full, positive meaning
FULL is better -- the same sign as every other gain in this project.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "pkl"))

from nuscenes.eval.common import loaders as L                                   # noqa: E402
from nuscenes.eval.common.data_classes import EvalBoxes                         # noqa: E402
from nuscenes.eval.detection.config import config_factory                       # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox                   # noqa: E402
from nuscenes.map_expansion.map_api import NuScenesMap                          # noqa: E402
from nuscenes.nuscenes import NuScenes                                          # noqa: E402
from planning_centric_metrics import calculate_pkl                              # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.paths import CACHE                                                     # noqa: E402

MAPS = ("singapore-hollandvillage", "singapore-queenstown",
        "boston-seaport", "singapore-onenorth")


def subset(boxes: EvalBoxes, tokens) -> EvalBoxes:
    out = EvalBoxes()
    keep = set(tokens)
    for t in boxes.boxes:
        if t in keep:
            out.add_boxes(t, boxes[t])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subs", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--variant", default="oracle")
    ap.add_argument("--cheap", default="ns_cheap_320")
    ap.add_argument("--full", default="ns_full_640")
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--nworkers", type=int, default=2)
    ap.add_argument("--modelpath", default=str(ROOT / "third_party" / "tip" / "planner.pt"))
    ap.add_argument("--mask_json",
                    default=str(ROOT / "third_party" / "tip" / "masks_trainval.json"))
    ap.add_argument("--tag", default="pkl")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    subs = Path(args.subs)
    man = json.loads((subs / "manifest.json").read_text())
    scene_names = set(man["scenes"])

    # restrict the split to our scenes so load_gt stays within memory
    _orig = L.create_splits_scenes
    L.create_splits_scenes = lambda: {**_orig(), "val": sorted(scene_names)}

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    dcfg = config_factory("detection_cvpr_2019")
    gt = L.load_gt(nusc, "val", DetectionBox, verbose=False)
    print(f"  GT loaded for {len(gt.boxes)} samples across {len(scene_names)} scenes")
    L.add_center_dist(nusc, gt)
    gt = L.filter_eval_boxes(nusc, gt, dcfg.class_range, verbose=False)
    tokens = sorted(gt.boxes)

    nusc_maps = {m: NuScenesMap(dataroot=args.dataroot, map_name=m) for m in MAPS}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out = {}
    for mode in (args.cheap, args.full):
        path = subs / f"{args.variant}__{mode}.json"
        pred, _ = L.load_prediction(str(path), dcfg.max_boxes_per_sample, DetectionBox,
                                    verbose=False)
        pred = subset(pred, tokens)
        L.add_center_dist(nusc, pred)
        pred = L.filter_eval_boxes(nusc, pred, dcfg.class_range, verbose=False)
        print(f"  PKL for {mode} on {len(tokens)} samples ...")
        info = calculate_pkl(gt, pred, tokens, nusc, nusc_maps, device,
                             nworkers=args.nworkers, bsz=args.bsz, verbose=False,
                             modelpath=args.modelpath, mask_json=args.mask_json)
        out[mode] = info
        print(f"    mean PKL {info['mean']:.4f}  median {info['median']:.4f}")

    per = pd.DataFrame({"sample_token": tokens,
                        "pkl_cheap": [out[args.cheap]["full"][t] for t in tokens],
                        "pkl_full": [out[args.full]["full"][t] for t in tokens]})
    per["G_PKL"] = per.pkl_cheap - per.pkl_full
    per.to_csv(run / f"pkl_{args.variant}.csv", index=False)
    per.to_csv(subs / f"pkl_{args.variant}.csv", index=False)
    print(f"\n  G_PKL mean {per.G_PKL.mean():+.4f}  frac>0 {float((per.G_PKL>0).mean()):.3f}  "
          f"|  PKL cheap {per.pkl_cheap.mean():.3f} full {per.pkl_full.mean():.3f}")
    (run / "summary.json").write_text(json.dumps(
        {"variant": args.variant, "n": len(tokens),
         "pkl_cheap_mean": float(per.pkl_cheap.mean()),
         "pkl_full_mean": float(per.pkl_full.mean()),
         "G_PKL_mean": float(per.G_PKL.mean())}, indent=2))
    print("wrote", run)


if __name__ == "__main__":
    main()
