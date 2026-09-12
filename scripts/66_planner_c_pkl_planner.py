#!/usr/bin/env python
"""Phase 0F: Planner C -- PKL's own published planner used as the downstream decision maker.

The strongest objection to this project is that every decision cost comes from a planner the
authors wrote, so a planning-aware metric built around a different planner cannot be expected
to predict it.  This script removes the objection by making the downstream decision maker
*PKL's own planner*, at the weights its authors released.

That planner (`compile_model(cin=5, cout=16)`) emits, for each of 16 future timesteps
(0.25 s to 4.0 s), a spatial heatmap of where the ego will be on a 0.3 m BEV grid.  Its
decision is therefore the path it intends.  So:

    path(mode)[k] = argmax over the reachable mask of the heatmap for timestep k,
                    converted to metres
    J_C(mode)     = mean over k of || path(mode)[k] - path(ground truth)[k] ||

J_C is the displacement, in metres, between the path the planner intends given a mode's
detections and the path the same planner intends given ground-truth boxes.  There is no
hand-written cost function anywhere in it, and no threshold: the units are metres of path
deviation.  dJ_C = J_C(cheap) - J_C(full) is positive when the expensive detector brings the
planner's intended path closer to the one perfect perception would have produced.

PKL is a distributional divergence of the same heatmaps; J_C is the decision-level deviation
of their argmax.  Asking whether G_PKL ranks frames by dJ_C is therefore maximally favourable
to PKL -- same planner, same weights, same forward pass, same inputs.
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
from nuscenes.eval.common.data_classes import EvalBoxes                        # noqa: E402
from nuscenes.eval.detection.config import config_factory                      # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox                  # noqa: E402
from nuscenes.map_expansion.map_api import NuScenesMap                         # noqa: E402
from nuscenes.nuscenes import NuScenes                                         # noqa: E402
from planning_centric_metrics.planning_kl import EvalLoader                    # noqa: E402
from planning_centric_metrics.models import compile_model                      # noqa: E402
from planning_centric_metrics.planning_kl import get_grid                      # noqa: E402
from rap import runmeta                                                        # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

MAPS = ("singapore-hollandvillage", "singapore-queenstown",
        "boston-seaport", "singapore-onenorth")
# The planner's BEV grid, taken from their own get_grid with the arguments EvalLoader uses,
# rather than re-derived here.  Only the resolution matters: J_C is a difference of two
# positions on the same grid, so any constant origin offset cancels.
_DX, _BX, (_NX, _NY) = get_grid([-17.0, -38.5, 60.0, 38.5], [0.3, 0.3])
GRID_RES = np.asarray(_DX)[:2]
# bx is the first cell centre, so the grid origin is bx - dx/2.  The constant cancels in
# `deviation`, which differences two paths on the same grid, but keep the mapping exact.
GRID_LO = np.asarray(_BX)[:2] - GRID_RES / 2.0
STRETCH, LAYERS, LINES = 70.0, ["road_segment", "lane"], ["road_divider", "lane_divider"]


def subset(boxes: EvalBoxes, tokens) -> EvalBoxes:
    out = EvalBoxes()
    keep = set(tokens)
    for t in boxes.boxes:
        if t in keep:
            out.add_boxes(t, boxes[t])
    return out


def paths_from_logits(logits: torch.Tensor, masks: torch.Tensor):
    """argmax path (metres) and probability-weighted path (metres), per sample, per timestep.

    `logits` is (B, 16, nx, ny); `masks` is (16, nx, ny) marking the pixels each timestep can
    reach.  Pixels outside the mask are excluded, because the planner was trained and is
    evaluated only there.
    """
    b, t, nx, ny = logits.shape
    flat = logits.reshape(b, t, nx * ny)
    mflat = masks.reshape(t, nx * ny)
    neg = torch.finfo(flat.dtype).min
    masked = torch.where(mflat[None], flat, torch.full_like(flat, neg))

    idx = masked.argmax(-1)                                            # (B, 16)
    ix, iy = idx // ny, idx % ny
    hard = torch.stack([ix, iy], -1).float()

    p = torch.softmax(masked, dim=-1)                                  # (B,16,nx*ny)
    gx = torch.arange(nx, device=logits.device, dtype=p.dtype)
    gy = torch.arange(ny, device=logits.device, dtype=p.dtype)
    px = (p.reshape(b, t, nx, ny).sum(-1) * gx).sum(-1)
    py = (p.reshape(b, t, nx, ny).sum(-2) * gy).sum(-1)
    soft = torch.stack([px, py], -1)
    return hard.cpu().numpy(), soft.cpu().numpy()


def to_metres(pix: np.ndarray) -> np.ndarray:
    return pix * GRID_RES + GRID_LO


def run_mode(gt, pred, tokens, nusc, nusc_maps, model, masks, device, bsz, nworkers):
    """Intended paths for the ground-truth raster and for one mode's raster."""
    ds = EvalLoader(gt, pred, tokens, nusc, nusc_maps, STRETCH, LAYERS, LINES)
    dl = torch.utils.data.DataLoader(ds, batch_size=bsz, shuffle=False,
                                     num_workers=nworkers)
    g_hard, g_soft, p_hard, p_soft = [], [], [], []
    from tqdm import tqdm
    for gtxs, predxs in tqdm(dl):
        with torch.no_grad():
            gl = model(gtxs.to(device))
            pl = model(predxs.to(device))
        a, b = paths_from_logits(gl, masks); g_hard.append(a); g_soft.append(b)
        a, b = paths_from_logits(pl, masks); p_hard.append(a); p_soft.append(b)
    cat = lambda xs: np.concatenate(xs, 0)
    return cat(g_hard), cat(g_soft), cat(p_hard), cat(p_soft)


def deviation(mode_pix: np.ndarray, gt_pix: np.ndarray) -> dict:
    a, b = to_metres(mode_pix), to_metres(gt_pix)
    d = np.linalg.norm(a - b, axis=-1)                                 # (N, 16) metres
    return {"mean_m": d.mean(1), "max_m": d.max(1), "final_m": d[:, -1],
            "lat_mean_m": np.abs(a[..., 1] - b[..., 1]).mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subs", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--variant", default="oracle")
    ap.add_argument("--cheap", default="ns_cheap_320")
    ap.add_argument("--full", default="ns_full_640")
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--modelpath", default=str(ROOT / "third_party" / "tip" / "planner.pt"))
    ap.add_argument("--mask_json",
                    default=str(ROOT / "third_party" / "tip" / "masks_trainval.json"))
    ap.add_argument("--nchunks", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--tag", default="planner_c")
    args = ap.parse_args()

    subs = Path(args.subs)
    suffix = "" if args.nchunks == 1 else f"_n{args.nchunks}c{args.chunk:02d}"
    done = subs / f"planC_{args.variant}{suffix}.csv"
    if args.skip_existing and done.exists():
        print(f"  {done.name} exists -- skipping"); return
    run = runmeta.new_run(args.tag, vars(args))

    # Claim the CUDA context and a reusable allocator pool *before* the nuScenes tables, the
    # ground truth and the map expansions fill host memory.  On this board GPU memory is the
    # same physical RAM, and one scene slice failed three times in a row asking for its first
    # 20 MiB after the maps had loaded -- the context could no longer be created, even though
    # only 33 MiB was allocated.  Reserving early, then freeing into the caching allocator,
    # lets every later allocation come from a pool that is already ours.
    if torch.cuda.is_available():
        torch.cuda.init()
        pool = torch.empty(int(600e6 // 4), dtype=torch.float32, device="cuda:0")
        del pool
        print(f"  reserved CUDA pool: {torch.cuda.memory_reserved() / 1e6:.0f} MB")

    man = json.loads((subs / "manifest.json").read_text())
    all_scenes = sorted(man["scenes"])
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    loc_of = {sc["name"]: nusc.get("log", sc["log_token"])["location"]
              for sc in nusc.scene if sc["name"] in set(all_scenes)}
    ordered = sorted(all_scenes, key=lambda n: (loc_of[n], n))
    if args.nchunks > 1:
        k, r = divmod(len(ordered), args.nchunks)
        lo = args.chunk * k + min(args.chunk, r)
        scene_names = set(ordered[lo:lo + k + (1 if args.chunk < r else 0)])
    else:
        scene_names = set(ordered)
    print(f"  chunk {args.chunk}/{args.nchunks}: {len(scene_names)} scenes")

    _orig = L.create_splits_scenes
    L.create_splits_scenes = lambda: {**_orig(), "val": sorted(scene_names)}

    dcfg = config_factory("detection_cvpr_2019")
    gt = L.load_gt(nusc, "val", DetectionBox, verbose=False)
    L.add_center_dist(nusc, gt)
    gt = L.filter_eval_boxes(nusc, gt, dcfg.class_range, verbose=False)
    tokens = sorted(gt.boxes)
    need = {loc_of[n] for n in scene_names}
    nusc_maps = {m: NuScenesMap(dataroot=args.dataroot, map_name=m)
                 for m in MAPS if m in need}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.modelpath, map_location="cpu"))
    model.eval()
    masks = (torch.Tensor(json.loads(Path(args.mask_json).read_text())) == 1).to(device)

    out, gt_hard, gt_soft = {}, None, None
    for mode in (args.cheap, args.full):
        pred, _ = L.load_prediction(str(subs / f"{args.variant}__{mode}.json"),
                                    dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
        pred = subset(pred, tokens)
        L.add_center_dist(nusc, pred)
        pred = L.filter_eval_boxes(nusc, pred, dcfg.class_range, verbose=False)
        print(f"  Planner C for {mode} on {len(tokens)} samples ...")
        gh, gs, ph, ps = run_mode(gt, pred, tokens, nusc, nusc_maps, model, masks,
                                  device, args.bsz, args.nworkers)
        if gt_hard is None:
            gt_hard, gt_soft = gh, gs
        else:
            # the ground-truth raster does not depend on the mode; assert it came out the same
            assert np.allclose(gt_hard, gh), "ground-truth path differs between modes"
        out[mode] = (ph, ps)

    df = pd.DataFrame({"sample_token": tokens})
    for mode, tag in ((args.cheap, "cheap"), (args.full, "full")):
        for kind, gtp, idx in (("", gt_hard, 0), ("_soft", gt_soft, 1)):
            dv = deviation(out[mode][idx], gtp)
            for k, v in dv.items():
                df[f"JC{kind}_{k}_{tag}"] = v
    df["JC_cheap"] = df["JC_mean_m_cheap"]
    df["JC_full"] = df["JC_mean_m_full"]
    df["dJC"] = df["JC_cheap"] - df["JC_full"]
    df["dJC_soft"] = df["JC_soft_mean_m_cheap"] - df["JC_soft_mean_m_full"]
    df.to_csv(run / f"planC_{args.variant}{suffix}.csv", index=False)
    df.to_csv(done, index=False)

    print(f"\n  J_C cheap {df.JC_cheap.mean():.3f} m   full {df.JC_full.mean():.3f} m")
    print(f"  dJC mean {df.dJC.mean():+.4f} m   frac>0 {float((df.dJC > 0).mean()):.3f}   "
          f"|dJC|>0.1m on {int((df.dJC.abs() > 0.1).sum())}/{len(df)}")
    print("\nwrote", run)


if __name__ == "__main__":
    main()
