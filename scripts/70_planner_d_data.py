#!/usr/bin/env python
"""Phase 0G: cache BEV rasters and future-ego-trajectory targets for Planner D.

Rasters are built by calling the released PKL planner's own rasteriser, so Planner C and
Planner D consume an identical representation and C-vs-D isolates the planner.  Ground-truth
boxes go through the same `load_gt` + `filter_eval_boxes` path used for the test split, so
training and test rasters are constructed identically rather than merely similarly.

For the test split three rasters are cached per sample -- ground truth, CHEAP and FULL -- from
the same submission JSONs Phase 0F used, so every Planner D seed and variant is evaluated on
byte-identical inputs.

Rasters are stored bit-packed (they are binary masks), which turns 1.3 MB per sample into
41 KB.  Work is chunked by scene because `load_gt` over tens of thousands of samples exhausts
this board's unified memory.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "pkl"))

from nuscenes.eval.common import loaders as L                                   # noqa: E402
from nuscenes.eval.common.data_classes import EvalBoxes                        # noqa: E402
from nuscenes.eval.detection.config import config_factory                      # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox                  # noqa: E402
from nuscenes.map_expansion.map_api import NuScenesMap                         # noqa: E402
from nuscenes.nuscenes import NuScenes                                         # noqa: E402
from pyquaternion import Quaternion                                            # noqa: E402

from planning_centric_metrics.planning_kl import (get_grid, get_local_map,     # noqa: E402
                                                 get_other_objs, raster_render,
                                                 samp2ego, samp2mapname)
from rap import runmeta                                                        # noqa: E402
from rap.ego_traj import ego_velocity, future_waypoints                        # noqa: E402
from rap.planner_d import HORIZONS                                             # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

MAPS = ("singapore-hollandvillage", "singapore-queenstown",
        "boston-seaport", "singapore-onenorth")
STRETCH, LAYERS, LINES = 70.0, ["road_segment", "lane"], ["road_divider", "lane_divider"]
DX, BX, (NX, NY) = get_grid([-17.0, -38.5, 60.0, 38.5], [0.3, 0.3])


def subset(boxes: EvalBoxes, tokens) -> EvalBoxes:
    out = EvalBoxes()
    keep = set(tokens)
    for t in boxes.boxes:
        if t in keep:
            out.add_boxes(t, boxes[t])
    return out


def render(samp, nusc, boxes, nusc_maps):
    """The 5-channel raster, plus the ego-frame boxes it was drawn from.

    The boxes are returned so the D-Aug variant can redraw the object channel from corrupted
    boxes without re-querying the map, which is the expensive part.
    """
    ego = samp2ego(samp, nusc)
    lmap = get_local_map(nusc_maps[samp2mapname(samp, nusc)],
                         [ego["x"], ego["y"], ego["hcos"], ego["hsin"]],
                         STRETCH, LAYERS, LINES)
    lobjs, lws = get_other_objs(boxes, np.array([ego["x"], ego["y"],
                                                 ego["hcos"], ego["hsin"]]))
    r = raster_render(lmap, [ego["l"], ego["w"]], lobjs, lws,
                      NX, NY, LAYERS, LINES, BX, DX).astype(np.uint8)
    return r, np.asarray(lobjs, np.float32), np.asarray(lws, np.float32)


def scene_poses(nusc, tokens):
    """Timestamps [s], world xy and world yaw for every sample, from the LIDAR_TOP ego pose.

    The same pose source `samp2ego` uses, so the raster's ego frame and the trajectory
    target's ego frame are the same frame by construction.
    """
    ts, xy, yaw = [], [], []
    for tok in tokens:
        samp = nusc.get("sample", tok)
        sd = nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        r = Quaternion(pose["rotation"]).rotation_matrix
        ts.append(samp["timestamp"] / 1e6)
        xy.append(pose["translation"][:2])
        yaw.append(np.arctan2(r[1, 0], r[0, 0]))
    return np.array(ts), np.array(xy), np.array(yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subs", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--variant", default="oracle", help="submission geometry, test split only")
    ap.add_argument("--cheap", default="ns_cheap_320")
    ap.add_argument("--full", default="ns_full_640")
    ap.add_argument("--out", default=str(CACHE / "planner_d"))
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--keep_clamped", action="store_true",
                    help="keep frames whose horizon runs past the end of the scene")
    ap.add_argument("--max_train_scenes", type=int, default=200)
    ap.add_argument("--tag", default="planner_d_data")
    args = ap.parse_args()

    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    # the test rasters are geometry-variant specific (CHEAP/FULL come from that variant's
    # submission), so the variant must be in the filename or a mono run silently overwrites the
    # oracle rasters that Planner C, Planner D and 74 all read
    vtag = f"_{args.variant}" if args.split == "test" else ""
    done = out / f"chunk{vtag}_n{args.nchunks}c{args.chunk:02d}.npz"
    if args.skip_existing and done.exists():
        print(f"  {done.name} exists -- skipping"); return
    run = runmeta.new_run(args.tag, vars(args))

    split = json.loads((ROOT / "configs/phase0g_scene_split.json").read_text())
    names = split[args.split]
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    loc_of = {sc["name"]: nusc.get("log", sc["log_token"])["location"]
              for sc in nusc.scene if sc["name"] in set(names)}
    ordered = sorted(names, key=lambda n: (loc_of[n], n))
    if args.split == "train" and args.max_train_scenes:
        # every k-th scene of the (location, name) ordering: preserves location balance, uses
        # no RNG, and the cap was fixed by measured render throughput before training
        step = max(1, len(ordered) // args.max_train_scenes)
        ordered = ordered[::step][:args.max_train_scenes]
        print(f"  training scenes capped to {len(ordered)} (every {step}th of the split)")
    if args.nchunks > 1:
        k, r = divmod(len(ordered), args.nchunks)
        lo = args.chunk * k + min(args.chunk, r)
        names = ordered[lo:lo + k + (1 if args.chunk < r else 0)]
    else:
        names = ordered
    print(f"  {args.split} chunk {args.chunk}/{args.nchunks}: {len(names)} scenes")
    if not names:
        np.savez_compressed(done, packed=np.zeros((0, 1), np.uint8)); return

    scene_set = set(names)
    _orig = L.create_splits_scenes
    L.create_splits_scenes = lambda: {**_orig(), "val": sorted(scene_set)}
    dcfg = config_factory("detection_cvpr_2019")
    gt = L.load_gt(nusc, "val", DetectionBox, verbose=False)
    L.add_center_dist(nusc, gt)
    gt = L.filter_eval_boxes(nusc, gt, dcfg.class_range, verbose=False)

    preds = {}
    if args.split == "test":
        for tag, mode in (("cheap", args.cheap), ("full", args.full)):
            p, _ = L.load_prediction(str(Path(args.subs) / f"{args.variant}__{mode}.json"),
                                     dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
            p = subset(p, list(gt.boxes))
            L.add_center_dist(nusc, p)
            preds[tag] = L.filter_eval_boxes(nusc, p, dcfg.class_range, verbose=False)

    need = {loc_of[n] for n in names}
    nusc_maps = {m: NuScenesMap(dataroot=args.dataroot, map_name=m)
                 for m in MAPS if m in need}

    name_to_scene = {sc["name"]: sc for sc in nusc.scene}
    kinds = ["gt"] + (["cheap", "full"] if args.split == "test" else [])
    rec = {k: [] for k in kinds}
    toks, tgts, vels, clamps, scenes = [], [], [], [], []
    objs, objws, nobj = [], [], []      # ego-frame GT boxes, for D-Aug re-rendering
    for name in names:
        sc = name_to_scene[name]
        tokens, tok = [], sc["first_sample_token"]
        while tok:
            tokens.append(tok); tok = nusc.get("sample", tok)["next"]
        ts, xy, yaw = scene_poses(nusc, tokens)
        for i, tk in enumerate(tokens):
            if tk not in gt.boxes:
                continue
            wp, _, clamped = future_waypoints(ts, xy, yaw, i, HORIZONS)
            if clamped > 0 and not args.keep_clamped:
                # the 4 s horizon runs past the end of the scene, so the target would be a
                # stationary ego rather than a real trajectory (Phase 0G amendment 2026-09-12)
                continue
            samp = nusc.get("sample", tk)
            r, lo, lw = render(samp, nusc, gt[tk], nusc_maps)
            rec["gt"].append(r)
            objs.append(lo.reshape(-1, 4)); objws.append(lw.reshape(-1, 2))
            nobj.append(len(lo))
            for k in kinds[1:]:
                rec[k].append(render(samp, nusc, preds[k][tk], nusc_maps)[0])
            toks.append(tk); tgts.append(wp.astype(np.float32))
            vels.append(ego_velocity(ts, xy, yaw, i).astype(np.float32))
            clamps.append(clamped); scenes.append(name)
        print(f"    {name}: {len(toks)} cumulative samples", flush=True)

    payload = {"sample_token": np.array(toks), "scene": np.array(scenes),
               "target": np.stack(tgts) if tgts else np.zeros((0, 16, 2), np.float32),
               "ego_v": np.stack(vels) if vels else np.zeros((0, 2), np.float32),
               "clamped": np.array(clamps, np.int16),
               "shape": np.array([len(toks), 5, NX, NY], np.int32),
               "obj_xy_cs": (np.concatenate(objs) if objs else np.zeros((0, 4), np.float32)),
               "obj_lw": (np.concatenate(objws) if objws else np.zeros((0, 2), np.float32)),
               "obj_count": np.array(nobj, np.int32)}
    for k in kinds:
        arr = np.stack(rec[k]).astype(bool) if rec[k] else np.zeros((0, 5, NX, NY), bool)
        payload[f"packed_{k}"] = np.packbits(arr.reshape(len(arr), -1), axis=1)
    np.savez_compressed(done, **payload)
    np.savez_compressed(run / done.name, **payload)
    frac = float(np.mean(np.array(clamps) > 0)) if clamps else 0.0
    print(f"\n  {len(toks)} samples, {frac:.3f} with a horizon past the scene end")
    print("  wrote", done)


if __name__ == "__main__":
    main()
