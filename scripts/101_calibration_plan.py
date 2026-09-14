#!/usr/bin/env python
"""Task 1, nuScenes q_plan: PKL's planner scored against the real trajectory at per-mode thresholds.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 1 pre-registration").  For each geometry and
mode, submissions built once at conf 0.10 (`60_build_submissions.py --op_conf 0.10`) are filtered
at each threshold: the monocular lift and the oracle-geometry GT match are per box, so the
submission at t is exactly the 0.10 submission restricted to score >= t.  The object channel of
each raster is redrawn from those boxes through the same load_prediction -> add_center_dist ->
filter_eval_boxes -> get_other_objs path and PKL's corner rasterisation; the map and ego channels do
not depend on detections and are taken from the cached ground-truth raster.  Planner C and ADE/FDE
are exactly 74's, including batch size and frame order, so a threshold's rasters reach the network
in the same batches 74 used.

Checks, all required before anything is written as a result:
  3  the 0.10 submissions filtered at 0.25 equal the current submission files, box for box;
  4  rasters rebuilt at 0.25 equal the cached CHEAP and FULL test rasters bit for bit;
  1  Planner C on the rebuilt 0.25 rasters reproduces planC_vs_truth[_mono].csv exactly.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "pkl"))
from nuscenes.eval.common import loaders as L                                   # noqa: E402
from nuscenes.eval.detection.config import config_factory                      # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox                  # noqa: E402
from nuscenes.nuscenes import NuScenes                                         # noqa: E402
from planning_centric_metrics.models import compile_model                      # noqa: E402
from planning_centric_metrics.planning_kl import get_other_objs, samp2ego      # noqa: E402
from rap import runmeta                                                        # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402
from rap.planner_d import ade_fde, object_channel                              # noqa: E402

_s = importlib.util.spec_from_file_location("m74", ROOT / "scripts" / "74_plannerC_vs_truth.py")
m74 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m74)

GRID05 = [round(0.10 + 0.05 * i, 2) for i in range(13)]
MODES = {"cheap": "ns_cheap_320", "full": "ns_full_640"}
NX = NY = 256


def load_boxes(nusc, path, tokens, dcfg):
    p, _ = L.load_prediction(str(path), dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
    keep = set(tokens)
    for tok in list(p.boxes):
        if tok not in keep:
            del p.boxes[tok]
    for tok in tokens:                      # samples with no boxes still need an (empty) entry
        p.boxes.setdefault(tok, [])
    L.add_center_dist(nusc, p)
    return L.filter_eval_boxes(nusc, p, dcfg.class_range, verbose=False)


def rebuild(packed_gt, boxes_by_token, tokens, egos, t):
    """Packed rasters whose object channel holds only the boxes scoring >= t."""
    out = np.empty_like(packed_gt)
    ncell = 5 * NX * NY
    for i, tok in enumerate(tokens):
        r = np.unpackbits(packed_gt[i])[:ncell].reshape(5, NX, NY).astype(bool)
        bs = [b for b in boxes_by_token[tok] if b.detection_score >= t]
        lobjs, lws = get_other_objs(bs, egos[tok])
        r[3] = object_channel(lobjs, lws) > 0 if len(bs) else False
        out[i] = np.packbits(r.reshape(-1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subs010", default=str(CACHE / "nusc_submissions_calib"))
    ap.add_argument("--subs025", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--outcomes", default=None, help="calibration_outcomes run dir (for S1/S3 FULL thresholds)")
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--tag", default="calibration_plan")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    oc = Path(args.outcomes) if args.outcomes else Path(sorted(glob.glob(str(ROOT / "results/raw/*_calibration_outcomes")))[-1])
    thr = json.loads((oc / "thresholds_S0_S3.json").read_text())
    extra_full = sorted({thr[k][s][1] for k in ("nusc_oracle", "nusc_mono") for s in ("S1", "S3")})
    tlist = {"cheap": list(GRID05), "full": sorted(set(GRID05) | set(extra_full))}
    print(f"  thresholds from {oc.name}: FULL extras {extra_full}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.init()
        pool = torch.empty(int(600e6 // 4), dtype=torch.float32, device="cuda:0"); del pool
    model = m74.compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(str(ROOT / "third_party" / "tip" / "planner.pt"), map_location="cpu"))
    model.eval()
    masks = (torch.Tensor(json.loads((ROOT / "third_party" / "tip" / "masks_trainval.json").read_text())) == 1).to(device)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    dcfg = config_factory("detection_cvpr_2019")
    checks, index = [], []
    for variant in ("oracle", "mono"):
        files = sorted((CACHE / "planner_d" / "test").glob(f"chunk_{variant}_*.npz"))
        acc = {k: [] for k in ("sample_token", "scene", "target", "packed_gt", "packed_cheap", "packed_full")}
        for f in files:
            z = np.load(f, allow_pickle=False)
            if len(z["sample_token"]) == 0:
                continue
            for k in acc:
                acc[k].append(z[k])
        d = {k: np.concatenate(v) for k, v in acc.items()}
        tokens = [str(t) for t in d["sample_token"]]
        egos = {}
        for tok in tokens:
            e = samp2ego(nusc.get("sample", tok), nusc)
            egos[tok] = np.array([e["x"], e["y"], e["hcos"], e["hsin"]])
        ref = pd.read_csv(CACHE / "planner_d" / f"planC_vs_truth{'' if variant == 'oracle' else '_' + variant}.csv")
        ref = ref.set_index("sample_token").loc[tokens]

        for role, mode in MODES.items():
            # check 3: the 0.10 submission filtered at 0.25 is the recorded submission
            s10 = json.loads((Path(args.subs010) / f"{variant}__{mode}.json").read_text())["results"]
            s25 = json.loads((Path(args.subs025) / f"{variant}__{mode}.json").read_text())["results"]
            same = set(s10) == set(s25) and all(
                [b for b in s10[tok] if b["detection_score"] >= 0.25] == s25[tok] for tok in s25)
            checks.append({"label": f"check3 {variant} {mode}: 0.10 submission filtered at 0.25 == recorded",
                           "pass": bool(same)})
            print(f"  {'PASS' if same else 'FAIL'} {checks[-1]['label']}", flush=True)
            boxes = load_boxes(nusc, Path(args.subs010) / f"{variant}__{mode}.json", tokens, dcfg)
            by_tok = {tok: list(boxes[tok]) for tok in tokens}
            for t in tlist[role]:
                t0 = time.time()
                packed = rebuild(d["packed_gt"], by_tok, tokens, egos, t)
                if abs(t - 0.25) < 1e-12:
                    same = bool(np.array_equal(packed, d[f"packed_{role}"]))
                    ndiff = int((packed != d[f"packed_{role}"]).any(1).sum())
                    checks.append({"label": f"check4 {variant} {role}: rasters rebuilt at 0.25 == cached",
                                   "pass": same, "frames_differing": ndiff})
                    print(f"  {'PASS' if same else 'FAIL'} {checks[-1]['label']} ({ndiff} frames differ)", flush=True)
                paths = m74.argmax_paths(model, masks, packed, device, args.bsz)
                ade, fde = ade_fde(paths, d["target"])
                name = f"plan__{variant}__{role}__t{t:.6f}.npz"
                np.savez_compressed(run / name, sample_token=np.array(tokens, dtype="U32"),
                                    scene=d["scene"].astype("U32"), JC_ade=ade, JC_fde=fde)
                index.append({"variant": variant, "role": role, "mode": mode, "threshold": t, "file": name})
                if abs(t - 0.25) < 1e-12:
                    diff = float(max(np.max(np.abs(ade - ref[f"JC_ade_{role}"].to_numpy())),
                                     np.max(np.abs(fde - ref[f"JC_fde_{role}"].to_numpy()))))
                    checks.append({"label": f"check1 {variant} {role}: Planner C at 0.25 == planC_vs_truth",
                                   "pass": diff <= 1e-9, "max_abs_diff": diff})
                    print(f"  {'PASS' if diff <= 1e-9 else 'FAIL'} {checks[-1]['label']} (max |d| {diff:.2e})", flush=True)
                print(f"  {variant:6s} {role:5s} t={t:.4f}  mean ADE {ade.mean():.4f}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(index).to_csv(run / "plan_index.csv", index=False)
    allpass = all(c["pass"] for c in checks)
    (run / "checks.json").write_text(json.dumps({"all_pass": allpass, "checks": checks}, indent=1))
    print(f"\n  checks 1, 3, 4 (q_plan): {'ALL PASS' if allpass else 'FAILED -- no scheme may be scored'}")
    print("  wrote", run)
    sys.exit(0 if allpass else 3)


if __name__ == "__main__":
    main()
