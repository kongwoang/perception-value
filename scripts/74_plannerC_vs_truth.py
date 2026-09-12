#!/usr/bin/env python
"""Phase 0G: score PKL's planner against the REAL future trajectory, not against itself.

Planner C as used in Phase 0F measured the displacement between the path PKL's planner intends
given a mode's detections and the path it intends given ground-truth boxes.  That made PKL's
eta of 0.872 largely an internal-consistency result: PKL is a divergence of the same heatmaps
whose argmax J_C displaces.

Here the same planner is scored the way Planner D is -- by error against the real future ego
trajectory, which is cached alongside the test rasters.  Two things follow that the Phase 0F
construction could not give:

  * a **viability check** for Planner C on the same terms Planner D failed, against the same
    constant-velocity baseline;
  * a **non-circular** decision value, V_C = ADE(cheap) - ADE(full) against the truth, which
    breaks the shared-functional objection because the target is now external to the planner.

The rasters are reused from the Planner D cache, so no map queries are needed and CHEAP, FULL
and GT inputs are byte-identical to the ones Planner D saw.
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

from planning_centric_metrics.models import compile_model                      # noqa: E402
from planning_centric_metrics.planning_kl import get_grid                      # noqa: E402
from rap import runmeta                                                        # noqa: E402
from rap.planner_d import ade_fde, constant_velocity                           # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402

NX = NY = 256
CELLS = 5 * NX * NY
DX, BX, _ = get_grid([-17.0, -38.5, 60.0, 38.5], [0.3, 0.3])
RES, LO = np.asarray(DX)[:2], np.asarray(BX)[:2]


@torch.no_grad()
def argmax_paths(model, masks, packed, device, bsz=16) -> np.ndarray:
    """Metre-space argmax path per sample, restricted to each timestep's reachable mask."""
    out = []
    for a in range(0, len(packed), bsz):
        x = np.stack([np.unpackbits(p)[:CELLS].reshape(5, NX, NY).astype(np.float32)
                      for p in packed[a:a + bsz]])
        logits = model(torch.from_numpy(x).to(device))
        b, t, nx, ny = logits.shape
        flat = logits.reshape(b, t, nx * ny)
        mflat = masks.reshape(t, nx * ny)
        neg = torch.finfo(flat.dtype).min
        masked = torch.where(mflat[None], flat, torch.full_like(flat, neg))
        idx = masked.argmax(-1)
        pix = torch.stack([idx // ny, idx % ny], -1).float().cpu().numpy()
        out.append(pix * RES + LO)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CACHE / "planner_d"))
    ap.add_argument("--modelpath", default=str(ROOT / "third_party" / "tip" / "planner.pt"))
    ap.add_argument("--mask_json",
                    default=str(ROOT / "third_party" / "tip" / "masks_trainval.json"))
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--tag", default="plannerC_vs_truth")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))

    files = sorted((Path(args.data) / "test").glob("chunk_*.npz"))
    keys = ("sample_token", "scene", "target", "ego_v",
            "packed_gt", "packed_cheap", "packed_full")
    acc = {k: [] for k in keys}
    for f in files:
        z = np.load(f, allow_pickle=False)
        if len(z["sample_token"]) == 0:
            continue
        for k in keys:
            acc[k].append(z[k])
    d = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"  {len(d['sample_token'])} test frames from {len(files)} chunks")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.init()
        pool = torch.empty(int(600e6 // 4), dtype=torch.float32, device="cuda:0"); del pool
    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.modelpath, map_location="cpu"))
    model.eval()
    masks = (torch.Tensor(json.loads(Path(args.mask_json).read_text())) == 1).to(device)

    p = {k: argmax_paths(model, masks, d[f"packed_{k}"], device, args.bsz)
         for k in ("gt", "cheap", "full")}
    true = d["target"]

    rows = {"sample_token": d["sample_token"], "scene": d["scene"]}
    for k in ("gt", "cheap", "full"):
        ade, fde = ade_fde(p[k], true)
        rows[f"JC_ade_{k}"], rows[f"JC_fde_{k}"] = ade, fde
        rows[f"JC_self_{k}"] = np.linalg.norm(p[k] - p["gt"], axis=-1).mean(1)
    df = pd.DataFrame(rows)
    df["dJC_ade"] = df.JC_ade_cheap - df.JC_ade_full          # non-circular decision value
    df["dJC_self"] = df.JC_self_cheap - df.JC_self_full       # the Phase 0F construction
    out = Path(args.data) / "planC_vs_truth.csv"
    df.to_csv(out, index=False); df.to_csv(run / out.name, index=False)

    cv_ade, cv_fde = ade_fde(constant_velocity(d["ego_v"]), true)
    nz = int((df.dJC_ade.abs() > 1e-9).sum())
    neg = int((df.dJC_ade < -1e-9).sum())
    s = {
        "n_frames": len(df),
        "JC_ade_gt": float(df.JC_ade_gt.mean()),
        "JC_ade_cheap": float(df.JC_ade_cheap.mean()),
        "JC_ade_full": float(df.JC_ade_full.mean()),
        "constant_velocity_ade": float(cv_ade.mean()),
        "improvement_vs_cv": 1.0 - float(df.JC_ade_gt.mean()) / float(cv_ade.mean()),
        "viability_pass": bool(float(df.JC_ade_gt.mean()) <= 0.90 * float(cv_ade.mean())),
        "dJC_ade_mean": float(df.dJC_ade.mean()),
        "frac_pos": float((df.dJC_ade > 1e-9).mean()),
        "frac_neg": float((df.dJC_ade < -1e-9).mean()),
        "harmful_among_affected": neg / max(nz, 1),
        "corr_selfcost_vs_truthcost": float(np.corrcoef(df.dJC_self, df.dJC_ade)[0, 1]),
    }
    (run / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"\n  PKL planner ADE against the real trajectory:")
    print(f"    GT raster    {s['JC_ade_gt']:.3f} m")
    print(f"    CHEAP raster {s['JC_ade_cheap']:.3f} m")
    print(f"    FULL raster  {s['JC_ade_full']:.3f} m")
    print(f"    constant velocity {s['constant_velocity_ade']:.3f} m  "
          f"({100*s['improvement_vs_cv']:+.1f}%)  "
          f"viability {'PASS' if s['viability_pass'] else 'FAIL'}")
    print(f"  dJC_ade mean {s['dJC_ade_mean']:+.4f} m  +{s['frac_pos']:.3f}/-{s['frac_neg']:.3f}"
          f"  harmful among affected {s['harmful_among_affected']:.3f}")
    print(f"  corr(Phase 0F self-consistency value, truth-referenced value) "
          f"{s['corr_selfcost_vs_truthcost']:+.3f}")
    print("  wrote", out)


if __name__ == "__main__":
    main()
