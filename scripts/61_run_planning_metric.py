#!/usr/bin/env python
"""Phase 0F Stage 1: run PKL or TIP on prebuilt submissions.

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

TIP (Transcendental Idealism of Planner, ICML 2023)
  repo   : https://github.com/qcraftai/tip
  commit : e52fd48de0624a9c54e93c8436f6bd7529b2a5d3
  scoring: unmodified. Its `parse_result` states the scores are converted "to comply with
           PKL's definition", so higher is worse for both.

Sign convention: higher is worse for both metrics, so G = score_cheap - score_full, positive
meaning FULL is better -- the same sign as every other gain in this project.
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


def load_tip():
    """Import TIP lazily: its utils module pulls in seaborn, which PKL does not need."""
    sys.path.insert(0, str(ROOT / "third_party" / "tip"))
    from trans_idealism_planner import calculate_tip
    return calculate_tip


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
    ap.add_argument("--metric", default="pkl", choices=["pkl", "tip"])
    ap.add_argument("--tag", default="pkl")
    ap.add_argument("--nchunks", type=int, default=1,
                    help="split the scene list into this many chunks (bounds peak memory)")
    ap.add_argument("--chunk", type=int, default=0, help="which chunk to evaluate")
    ap.add_argument("--skip_existing", action="store_true",
                    help="return immediately if this chunk's CSV is already written")
    args = ap.parse_args()

    subs = Path(args.subs)
    # Idempotent: a chunk whose CSV already exists is skipped, so an interrupted sweep can
    # be restarted without recomputing what landed.  One chunk here died of CUDA OOM only
    # because an unrelated job ran alongside it; this board has room for exactly one.
    suffix = "" if args.nchunks == 1 else f"_c{args.chunk:02d}"
    done = subs / f"{args.metric}_{args.variant}{suffix}.csv"
    if args.skip_existing and done.exists():
        print(f"  {done.name} exists -- skipping")
        return

    run = runmeta.new_run(args.tag, vars(args))
    man = json.loads((subs / "manifest.json").read_text())
    all_scenes = sorted(man["scenes"])

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    loc_of = {sc["name"]: nusc.get("log", sc["log_token"])["location"]
              for sc in nusc.scene if sc["name"] in set(all_scenes)}

    # NuScenes metadata, the four map expansions, GT and predictions together exhaust the
    # 15 GB unified memory on Xavier, leaving the planner no room on the GPU.  Evaluating a
    # slice of scenes per process bounds peak usage; PKL and TIP are per-sample scores, so
    # chunk boundaries cannot change any result.  Ordering by location keeps each chunk
    # inside as few map expansions as possible.
    ordered = sorted(all_scenes, key=lambda n: (loc_of[n], n))
    if args.nchunks > 1:
        k, r = divmod(len(ordered), args.nchunks)
        lo = args.chunk * k + min(args.chunk, r)
        hi = lo + k + (1 if args.chunk < r else 0)
        scene_names = set(ordered[lo:hi])
    else:
        scene_names = set(ordered)
    print(f"  chunk {args.chunk}/{args.nchunks}: {len(scene_names)} scenes")

    # restrict the split to our scenes so load_gt stays within memory
    _orig = L.create_splits_scenes
    L.create_splits_scenes = lambda: {**_orig(), "val": sorted(scene_names)}

    dcfg = config_factory("detection_cvpr_2019")
    gt = L.load_gt(nusc, "val", DetectionBox, verbose=False)
    print(f"  GT loaded for {len(gt.boxes)} samples across {len(scene_names)} scenes")
    L.add_center_dist(nusc, gt)
    gt = L.filter_eval_boxes(nusc, gt, dcfg.class_range, verbose=False)
    tokens = sorted(gt.boxes)

    need = {loc_of[n] for n in scene_names}
    print(f"  loading {len(need)} of {len(MAPS)} maps: {sorted(need)}")
    nusc_maps = {m: NuScenesMap(dataroot=args.dataroot, map_name=m)
                 for m in MAPS if m in need}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out = {}
    for mode in (args.cheap, args.full):
        path = subs / f"{args.variant}__{mode}.json"
        pred, _ = L.load_prediction(str(path), dcfg.max_boxes_per_sample, DetectionBox,
                                    verbose=False)
        pred = subset(pred, tokens)
        L.add_center_dist(nusc, pred)
        pred = L.filter_eval_boxes(nusc, pred, dcfg.class_range, verbose=False)
        print(f"  {args.metric.upper()} for {mode} on {len(tokens)} samples ...")
        if args.metric == "pkl":
            info = calculate_pkl(gt, pred, tokens, nusc, nusc_maps, device,
                                 nworkers=args.nworkers, bsz=args.bsz, verbose=False,
                                 modelpath=args.modelpath, mask_json=args.mask_json)
        else:
            calculate_tip = load_tip()
            info = calculate_tip(gt, pred, tokens, nusc, nusc_maps, device,
                                 nworkers=args.nworkers, bsz=args.bsz, verbose=False,
                                 model_path=args.modelpath, mask_json=args.mask_json)
        out[mode] = info
        print(f"    mean {args.metric.upper()} {info['mean']:.4f}  median {info['median']:.4f}")

    m = args.metric
    per = pd.DataFrame({"sample_token": tokens,
                        f"{m}_cheap": [out[args.cheap]["full"][t] for t in tokens],
                        f"{m}_full": [out[args.full]["full"][t] for t in tokens]})
    per[f"G_{m.upper()}"] = per[f"{m}_cheap"] - per[f"{m}_full"]
    per.to_csv(run / f"{m}_{args.variant}{suffix}.csv", index=False)
    per.to_csv(subs / f"{m}_{args.variant}{suffix}.csv", index=False)
    print(f"\n  G_{m.upper()} mean {per[f'G_{m.upper()}'].mean():+.4f}  "
          f"frac>0 {float((per[f'G_{m.upper()}']>0).mean()):.3f}  |  "
          f"{m} cheap {per[f'{m}_cheap'].mean():.3f} full {per[f'{m}_full'].mean():.3f}")
    (run / "summary.json").write_text(json.dumps(
        {"metric": m, "variant": args.variant, "n": len(tokens),
         "cheap_mean": float(per[f"{m}_cheap"].mean()),
         "full_mean": float(per[f"{m}_full"].mean()),
         "gain_mean": float(per[f"G_{m.upper()}"].mean())}, indent=2))
    print("wrote", run)


if __name__ == "__main__":
    main()
