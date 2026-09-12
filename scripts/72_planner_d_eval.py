#!/usr/bin/env python
"""Phase 0G: evaluate a trained Planner D on the CHEAP / FULL / GT test rasters.

Produces one row per test frame with the primary cost -- error against the *real* future ego
trajectory -- plus the two pre-registered controls: final displacement error, and
self-consistency against Planner D's own ground-truth-conditioned output.

The self-consistency cost exists only to be comparable with Planner C, and is secondary for
the same reason Planner C is: it measures a planner against itself.  The paper-facing Planner D
result is the real-trajectory ADE.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import runmeta                                                        # noqa: E402
from rap.planner_d import PlannerD, ade_fde, constant_velocity                 # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

NX = NY = 256
CELLS = 5 * NX * NY


def load_test(files):
    keys = ("sample_token", "scene", "target", "ego_v", "clamped",
            "packed_gt", "packed_cheap", "packed_full")
    acc = {k: [] for k in keys}
    for f in files:
        z = np.load(f, allow_pickle=False)
        if len(z["sample_token"]) == 0:
            continue
        for k in keys:
            acc[k].append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


@torch.no_grad()
def paths(model, packed, device, bsz=32) -> np.ndarray:
    out = []
    for a in range(0, len(packed), bsz):
        blk = packed[a:a + bsz]
        x = np.stack([np.unpackbits(p)[:CELLS].reshape(5, NX, NY).astype(np.float32)
                      for p in blk])
        out.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint written by 71_planner_d_train")
    ap.add_argument("--data", default=str(CACHE / "planner_d"))
    ap.add_argument("--variant", default="oracle", help="submission geometry of the rasters")
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    name = Path(args.ckpt).stem.replace("_best", "")
    run = runmeta.new_run(args.tag or f"planner_d_eval_{name}", vars(args))

    files = sorted((Path(args.data) / "test").glob("chunk_*.npz"))
    if not files:
        raise SystemExit("no cached Planner D test data; run 70_planner_d_data.py --split test")
    d = load_test(files)
    n = len(d["sample_token"])
    print(f"  {n} test frames from {len(files)} chunks")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PlannerD().to(device)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    print(f"  {Path(args.ckpt).name}: epoch {ck.get('epoch')}, val ADE "
          f"{ck.get('val_ade', float('nan')):.3f} m")

    p = {k: paths(model, d[f"packed_{k}"], device, args.bsz) for k in ("gt", "cheap", "full")}
    true = d["target"]

    rows = {"sample_token": d["sample_token"], "scene": d["scene"],
            "clamped": d["clamped"]}
    for k in ("gt", "cheap", "full"):
        ade, fde = ade_fde(p[k], true)
        rows[f"JD_ade_{k}"] = ade
        rows[f"JD_fde_{k}"] = fde
        self_d = np.linalg.norm(p[k] - p["gt"], axis=-1)
        rows[f"JD_self_{k}"] = self_d.mean(1)
    df = pd.DataFrame(rows)
    # primary decision value: does FULL bring the planner's trajectory closer to the truth
    df["dJD_ade"] = df.JD_ade_cheap - df.JD_ade_full
    df["dJD_fde"] = df.JD_fde_cheap - df.JD_fde_full
    df["dJD_self"] = df.JD_self_cheap - df.JD_self_full

    cv_ade, _ = ade_fde(constant_velocity(d["ego_v"]), true)
    out = Path(args.data) / f"planD_{name}_{args.variant}.csv"
    df.to_csv(out, index=False)
    df.to_csv(run / out.name, index=False)

    nz = (df.dJD_ade.abs() > 1e-9).sum()
    neg = (df.dJD_ade < -1e-9).sum()
    summary = {
        "checkpoint": Path(args.ckpt).name, "n_frames": int(n),
        "test_ade_gt": float(df.JD_ade_gt.mean()),
        "test_ade_cheap": float(df.JD_ade_cheap.mean()),
        "test_ade_full": float(df.JD_ade_full.mean()),
        "test_ade_constant_velocity": float(cv_ade.mean()),
        "dJD_ade_mean": float(df.dJD_ade.mean()),
        "frac_dJD_pos": float((df.dJD_ade > 1e-9).mean()),
        "frac_dJD_zero": float((df.dJD_ade.abs() <= 1e-9).mean()),
        "frac_dJD_neg": float((df.dJD_ade < -1e-9).mean()),
        "harmful_frac_among_affected": float(neg / max(nz, 1)),
        "frames_clamped_horizon": int((df.clamped > 0).sum()),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  test ADE  GT {summary['test_ade_gt']:.3f}  CHEAP "
          f"{summary['test_ade_cheap']:.3f}  FULL {summary['test_ade_full']:.3f}  "
          f"(const-vel {summary['test_ade_constant_velocity']:.3f}) m")
    print(f"  dJD_ade mean {summary['dJD_ade_mean']:+.4f} m   "
          f"+{summary['frac_dJD_pos']:.3f} / 0 {summary['frac_dJD_zero']:.3f} / "
          f"-{summary['frac_dJD_neg']:.3f}   harmful among affected "
          f"{summary['harmful_frac_among_affected']:.3f}")
    print("  wrote", out)


if __name__ == "__main__":
    main()
