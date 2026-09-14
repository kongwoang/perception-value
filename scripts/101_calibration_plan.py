#!/usr/bin/env python
"""Task 1, nuScenes q_plan: PKL's planner scored against the real trajectory at per-mode thresholds.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 1 pre-registration").  For each geometry and
mode, submissions built once at conf 0.10 (`60_build_submissions.py --op_conf 0.10`) are filtered
at each threshold: the monocular lift and the oracle-geometry GT match are per box, so the
submission at t is exactly the 0.10 submission restricted to score >= t.  The object channel of
each raster is redrawn from those boxes through the same load_prediction -> add_center_dist ->
filter_eval_boxes -> get_other_objs path and PKL's corner rasterisation; the map and ego channels do
not depend on detections and are taken from the cached ground-truth raster.  Planner C and ADE/FDE
are exactly 74's, including batch size and frame order.

Two processes, because on this board's unified memory the nuScenes tables and the planner cannot
share one (the single-process version ran out of CUDA memory; RESEARCH_LOG 11:10):
  --stage boxes  devkit, CPU only: check 3, then every filtered box in the ego frame at 0.10, with
                 its score, per sample -- get_other_objs transforms each box on its own, so subsetting
                 by score afterwards equals transforming the subset
  --stage plan   GPU, no nuScenes tables: rasters per threshold, check 4, Planner C, check 1

Checks, all required before 102 scores anything:
  3  the 0.10 submissions filtered at 0.25 equal the current submission files, box for box;
  4  rasters rebuilt at 0.25 equal the cached CHEAP and FULL test rasters bit for bit;
  1  Planner C on the rebuilt 0.25 rasters reproduces planC_vs_truth[_mono].csv exactly.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "pkl"))
from rap import runmeta                                                        # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

GRID05 = [round(0.10 + 0.05 * i, 2) for i in range(13)]
MODES = {"cheap": "ns_cheap_320", "full": "ns_full_640"}
VARIANTS = ("oracle", "mono")
NX = NY = 256


def chunk_tokens(variant):
    files = sorted((CACHE / "planner_d" / "test").glob(f"chunk_{variant}_*.npz"))
    toks = [np.load(f, allow_pickle=False)["sample_token"] for f in files]
    return [str(t) for t in np.concatenate([t for t in toks if len(t)])]


def stage_boxes(args):
    from nuscenes.eval.common import loaders as L
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.nuscenes import NuScenes
    from planning_centric_metrics.planning_kl import get_other_objs, samp2ego

    run = runmeta.new_run("calibration_plan_boxes", vars(args))
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    dcfg = config_factory("detection_cvpr_2019")
    checks = []
    for variant in VARIANTS:
        tokens = chunk_tokens(variant)
        egos = {}
        for tok in tokens:
            e = samp2ego(nusc.get("sample", tok), nusc)
            egos[tok] = np.array([e["x"], e["y"], e["hcos"], e["hsin"]])
        for role, mode in MODES.items():
            s10 = json.loads((Path(args.subs010) / f"{variant}__{mode}.json").read_text())["results"]
            s25 = json.loads((Path(args.subs025) / f"{variant}__{mode}.json").read_text())["results"]
            same = set(s10) == set(s25) and all(
                [b for b in s10[tok] if b["detection_score"] >= 0.25] == s25[tok] for tok in s25)
            checks.append({"label": f"check3 {variant} {mode}: 0.10 submission filtered at 0.25 == recorded",
                           "pass": bool(same)})
            print(f"  {'PASS' if same else 'FAIL'} {checks[-1]['label']}", flush=True)
            p, _ = L.load_prediction(str(Path(args.subs010) / f"{variant}__{mode}.json"),
                                     dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
            keep = set(tokens)
            for tok in list(p.boxes):
                if tok not in keep:
                    del p.boxes[tok]
            for tok in tokens:
                p.boxes.setdefault(tok, [])
            L.add_center_dist(nusc, p)
            p = L.filter_eval_boxes(nusc, p, dcfg.class_range, verbose=False)
            lo, lw, sc, off = [], [], [], [0]
            for tok in tokens:
                bs = list(p[tok])
                if bs:
                    a, b = get_other_objs(bs, egos[tok])
                    lo.append(np.asarray(a, np.float64).reshape(-1, 4))
                    lw.append(np.asarray(b, np.float64).reshape(-1, 2))
                    sc.append(np.array([x.detection_score for x in bs], np.float64))
                off.append(off[-1] + len(bs))
            np.savez_compressed(run / f"boxes__{variant}__{role}.npz", sample_token=np.array(tokens, "U32"),
                                offsets=np.array(off, np.int64),
                                lobjs=np.concatenate(lo) if lo else np.zeros((0, 4)),
                                lws=np.concatenate(lw) if lw else np.zeros((0, 2)),
                                score=np.concatenate(sc) if sc else np.zeros(0))
            print(f"  {variant:6s} {role:5s} {off[-1]} boxes over {len(tokens)} samples", flush=True)
    allpass = all(c["pass"] for c in checks)
    (run / "checks.json").write_text(json.dumps({"all_pass": allpass, "checks": checks}, indent=1))
    print(f"  check 3: {'ALL PASS' if allpass else 'FAILED'}; wrote {run}")
    sys.exit(0 if allpass else 3)


def stage_plan(args):
    import torch
    from planning_centric_metrics.models import compile_model
    from rap.planner_d import ade_fde, object_channel
    _s = importlib.util.spec_from_file_location("m74", ROOT / "scripts" / "74_plannerC_vs_truth.py")
    m74 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(m74)

    boxes_run = Path(args.boxes) if args.boxes else Path(sorted(glob.glob(str(ROOT / "results/raw/*_calibration_plan_boxes")))[-1])
    carried = json.loads((boxes_run / "checks.json").read_text())
    assert carried["all_pass"], f"check 3 failed in {boxes_run.name}"
    oc = Path(args.outcomes) if args.outcomes else Path(sorted(glob.glob(str(ROOT / "results/raw/*_calibration_outcomes")))[-1])
    thr = json.loads((oc / "thresholds_S0_S3.json").read_text())
    extra_full = sorted({thr[k][s][1] for k in ("nusc_oracle", "nusc_mono") for s in ("S1", "S3")})
    tlist = {"cheap": list(GRID05), "full": sorted(set(GRID05) | set(extra_full))}
    run = runmeta.new_run(args.tag, {**vars(args), "boxes_run": boxes_run.name, "outcomes_run": oc.name})
    print(f"  boxes from {boxes_run.name}; thresholds from {oc.name}: FULL extras {extra_full}")

    device = torch.device("cuda:0")
    torch.cuda.init()
    pool = torch.empty(int(600e6 // 4), dtype=torch.float32, device="cuda:0"); del pool
    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(str(ROOT / "third_party" / "tip" / "planner.pt"), map_location="cpu"))
    model.eval()
    masks = (torch.Tensor(json.loads((ROOT / "third_party" / "tip" / "masks_trainval.json").read_text())) == 1).to(device)

    checks, index = list(carried["checks"]), []
    ncell = 5 * NX * NY
    for variant in VARIANTS:
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
        ref = pd.read_csv(CACHE / "planner_d" / f"planC_vs_truth{'' if variant == 'oracle' else '_' + variant}.csv")
        ref = ref.set_index("sample_token").loc[tokens]
        for role in MODES:
            bz = np.load(boxes_run / f"boxes__{variant}__{role}.npz", allow_pickle=False)
            assert [str(t) for t in bz["sample_token"]] == tokens, "box file and raster chunks disagree on sample order"
            off, lobjs, lws, score = bz["offsets"], bz["lobjs"], bz["lws"], bz["score"]
            for t in tlist[role]:
                t0 = time.time()
                packed = np.empty_like(d["packed_gt"])
                for i in range(len(tokens)):
                    r = np.unpackbits(d["packed_gt"][i])[:ncell].reshape(5, NX, NY).astype(bool)
                    a, b = off[i], off[i + 1]
                    sel = score[a:b] >= t
                    r[3] = (object_channel(lobjs[a:b][sel], lws[a:b][sel]) > 0) if sel.any() else False
                    packed[i] = np.packbits(r.reshape(-1))
                if abs(t - 0.25) < 1e-12:
                    ndiff = int((packed != d[f"packed_{role}"]).any(1).sum())
                    checks.append({"label": f"check4 {variant} {role}: rasters rebuilt at 0.25 == cached",
                                   "pass": ndiff == 0, "frames_differing": ndiff})
                    print(f"  {'PASS' if ndiff == 0 else 'FAIL'} {checks[-1]['label']} ({ndiff} frames differ)", flush=True)
                paths = m74.argmax_paths(model, masks, packed, device, args.bsz)
                ade, fde = ade_fde(paths, d["target"])
                name = f"plan__{variant}__{role}__t{t:.6f}.npz"
                np.savez_compressed(run / name, sample_token=np.array(tokens, dtype="U32"),
                                    scene=d["scene"].astype("U32"), JC_ade=ade, JC_fde=fde)
                index.append({"variant": variant, "role": role, "mode": MODES[role], "threshold": t, "file": name})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["boxes", "plan"])
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subs010", default=str(CACHE / "nusc_submissions_calib"))
    ap.add_argument("--subs025", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--boxes", default=None, help="calibration_plan_boxes run dir (stage plan)")
    ap.add_argument("--outcomes", default=None, help="calibration_outcomes run dir (for S1/S3 FULL thresholds)")
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--tag", default="calibration_plan")
    args = ap.parse_args()
    stage_boxes(args) if args.stage == "boxes" else stage_plan(args)


if __name__ == "__main__":
    main()
