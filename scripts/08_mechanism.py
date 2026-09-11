#!/usr/bin/env python
"""Where does extra perception compute actually help, and does it coincide with what matters?

Extra resolution recovers small, distant objects. Criticality is concentrated on near,
in-corridor, closing objects. Those two gradients point in opposite directions, so the
value of compute lives in their overlap — which is the reason the research question is
not answerable from either signal alone. This script measures the overlap directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import geometry as G, kitti, runmeta      # noqa: E402
from rap.cache import DetCache                     # noqa: E402
from rap.paths import CACHE                        # noqa: E402
from rap.risk import RiskConfig, match             # noqa: E402

DIST_EDGES = [0, 10, 20, 30, 45, 60, 1000]
HEIGHT_EDGES = [10, 20, 30, 40, 60, 90, 140, 10000]
CRIT_EDGES = [0.0, 0.05, 0.15, 0.30, 0.50, 1.01]


def object_table(seqs, det_dir, modes, cfg, model) -> pd.DataFrame:
    """One row per GT object per sequence, with whether each mode detected it."""
    caches = {m: {s: DetCache(det_dir / m / f"{s}.npz") for s in seqs} for m in modes}
    rows = []
    for s in seqs:
        geom = G.sequence_geometry(s)
        crit_all = G.criticality_for(geom, model)
        for i, fr in enumerate(caches[modes[0]][s].frames):
            fr = int(fr)
            sel = geom["frame"] == fr
            g, c = geom[sel], crit_all[sel]
            ok = (g["y2"] - g["y1"]) >= cfg.min_gt_height
            g, c = g[ok], c[ok]
            if not len(g):
                continue
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            hit = {}
            for m in modes:
                cache = caches[m][s]
                d = cache.det(cache.index[fr])
                k = d["conf"] >= cfg.op_conf
                best, _ = match(gt, np.array(["v"] * len(gt)), d["xyxy"][k].astype(float),
                                d["conf"][k], d["coarse"][k], cfg)
                hit[m] = best >= cfg.iou_thr
            for j in range(len(g)):
                rows.append({
                    "seq": s, "frame": fr, "type": g["type"][j],
                    "height_px": float(g["y2"][j] - g["y1"][j]),
                    "dist_m": float(g["long_near"][j]),
                    "lat_m": float(g["lat_cen"][j]),
                    "ttc_s": float(min(g["ttc"][j], 1e3)),
                    "crit": float(c[j]),
                    **{f"hit_{m}": bool(hit[m][j]) for m in modes},
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--op_conf", type=float, default=0.25)
    ap.add_argument("--crit_model", default="composite")
    args = ap.parse_args()

    det = Path(args.det)
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    cfg = RiskConfig(op_conf=args.op_conf)
    model = G.CRITICALITY_MODELS[args.crit_model]
    run = runmeta.new_run("mechanism", vars(args))

    obj = object_table(seqs, det, [args.cheap, args.full], cfg, model)
    obj.to_pickle(run / "objects.pkl")
    hc, hf = f"hit_{args.cheap}", f"hit_{args.full}"
    obj["recovered"] = (~obj[hc]) & obj[hf]
    obj["lost"] = obj[hc] & (~obj[hf])

    print(f"{len(obj)} GT objects over {obj.seq.nunique()} sequences")
    print(f"recall  cheap={obj[hc].mean():.3f}  full={obj[hf].mean():.3f}")
    print(f"recovered by FULL: {obj.recovered.sum()} ({obj.recovered.mean():.1%})   "
          f"lost by FULL: {obj.lost.sum()} ({obj.lost.mean():.1%})")

    def table(col, edges, label):
        b = pd.cut(obj[col], edges, right=False)
        t = obj.groupby(b, observed=True).agg(
            n=("crit", "size"), mean_crit=("crit", "mean"),
            recall_cheap=(hc, "mean"), recall_full=(hf, "mean"),
            recovered=("recovered", "mean"))
        t["risk_recovered"] = obj.groupby(b, observed=True).apply(
            lambda d: float((d["crit"] * d["recovered"]).sum()))
        t["share_of_risk_recovered"] = t["risk_recovered"] / t["risk_recovered"].sum()
        t["crit_mass"] = obj.groupby(b, observed=True)["crit"].sum() / obj["crit"].sum()
        print(f"\nby {label}:")
        print(t.round(3).to_string())
        return t

    t_dist = table("dist_m", DIST_EDGES, "ego distance (m)")
    t_h = table("height_px", HEIGHT_EDGES, "GT box height (px)")
    t_crit = table("crit", CRIT_EDGES, "criticality")
    for name, t in [("by_distance", t_dist), ("by_height", t_h), ("by_criticality", t_crit)]:
        t.to_csv(run / f"mechanism_{name}.csv")

    # the decisive cross-tab: is recovered risk concentrated where criticality is high?
    obj["dbin"] = pd.cut(obj.dist_m, DIST_EDGES, right=False)
    obj["cbin"] = pd.cut(obj.crit, CRIT_EDGES, right=False)
    cross = obj.pivot_table(index="dbin", columns="cbin", values="recovered",
                            aggfunc="mean", observed=True)
    print("\nP(recovered by FULL) by distance x criticality:")
    print(cross.round(3).to_string())
    cross.to_csv(run / "mechanism_cross.csv")

    summary = {
        "n_objects": int(len(obj)),
        "recall_cheap": float(obj[hc].mean()), "recall_full": float(obj[hf].mean()),
        "n_recovered": int(obj.recovered.sum()), "n_lost": int(obj.lost.sum()),
        "mean_crit_all": float(obj.crit.mean()),
        "mean_crit_recovered": float(obj.loc[obj.recovered, "crit"].mean()),
        "mean_crit_missed_by_both": float(obj.loc[~obj[hc] & ~obj[hf], "crit"].mean()),
        "mean_dist_all": float(obj.dist_m.mean()),
        "mean_dist_recovered": float(obj.loc[obj.recovered, "dist_m"].mean()),
        "corr_crit_recovered": float(np.corrcoef(obj.crit, obj.recovered.astype(float))[0, 1]),
        "corr_dist_recovered": float(np.corrcoef(obj.dist_m, obj.recovered.astype(float))[0, 1]),
        "share_risk_recovered_within_30m": float(
            t_dist["share_of_risk_recovered"].iloc[:3].sum()),
        "share_crit_mass_within_30m": float(t_dist["crit_mass"].iloc[:3].sum()),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
