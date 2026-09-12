#!/usr/bin/env python
"""Phase 0F: re-score every fidelity pair with Planner B, an independent downstream planner.

Phase 0E's standing weakness is that all decision value comes from one hand-written
threshold controller.  This script recomputes the per-frame decision gain with
`rap.planner_b` -- a receding-horizon sampling planner whose cost is the *executed*
trajectory scored against the true obstacle set -- so the Phase 0F conclusions can be
stated for two planners that share no decision logic.

Writes one table per fidelity cell with `JB_cheap`, `JB_full`, `dJB`, and the chosen plan,
joinable on (seq, frame) to the Phase 0E tables and hence to the PKL/TIP scores.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import planner_b as B, runmeta                                         # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.decision import DetCache, KittiAdapter, _apply_range_source            # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                   # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402


def build_b(det_dir: Path, cheap: str, full: str, seqs, cfg: RiskConfig,
            adapter, range_source: str, pb: B.PlannerBParams,
            cb: B.CostBParams) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in seqs:
        geom = adapter.geometry(s)
        speeds = adapter.speeds(s)
        c = DetCache(det_dir / cheap / f"{s}.npz")
        f = DetCache(det_dir / full / f"{s}.npz")
        prev_c = prev_f = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            truth = (g["long_near"].astype(float), g["lat_min"].astype(float),
                     g["lat_max"].astype(float), g["ttc"].astype(float))

            out = {}
            for tag, cache, idx, prev in (("cheap", c, i, prev_c), ("full", f, j, prev_f)):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.op_conf
                geo_k = {kk: vv[k] for kk, vv in geo.items()}
                bx = d["xyxy"][k].astype(float)
                z, lo, hi, tt = _apply_range_source(bx, geo_k, g, range_source, rng, 0.0)
                cand = B.plan(z, lo, hi, tt, v, prev, pb, cb)
                out[tag] = (cand, B.executed_cost(cand, *truth, v, prev, pb, cb))
            gt_cand = B.plan(*truth, v, None, pb, cb)
            prev_c, prev_f = out["cheap"][0], out["full"][0]

            rows.append({
                "seq": s, "frame": fr, "v_ego": v,
                "planB_cheap": out["cheap"][0], "planB_full": out["full"][0],
                "planB_gt": gt_cand,
                "JB_cheap": out["cheap"][1]["J"], "JB_full": out["full"][1]["J"],
                "dJB": out["cheap"][1]["J"] - out["full"][1]["J"],
                "collB_cheap": out["cheap"][1]["collision"],
                "collB_full": out["full"][1]["collision"],
                "clearB_cheap": out["cheap"][1]["min_clearance"],
                "clearB_full": out["full"][1]["min_clearance"],
                "same_planB": int(out["cheap"][0] == out["full"][0]),
                "a_lon_cheap": out["cheap"][1]["a_lon"], "a_lon_full": out["full"][1]["a_lon"],
                "d_lat_cheap": out["cheap"][1]["d_lat"], "d_lat_full": out["full"][1]["d_lat"],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="planner_b")
    ap.add_argument("--params", default="default", choices=list(B.PARAMS_B))
    ap.add_argument("--costs", default="default", choices=list(B.COSTS_B))
    ap.add_argument("--kitti_only", action="store_true")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    cfg = RiskConfig()
    pb, cb = B.PARAMS_B[args.params], B.COSTS_B[args.costs]

    kseqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    specs = [
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         KittiAdapter, "oracle"),
        ("KITTI", "YOLOv8s", "cheap_384", "full_640", CACHE / "det", kseqs,
         KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_512", "full_640", CACHE / "det512", kseqs,
         KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_cheap_320", "rt_full_640", CACHE / "rtdetr_kitti",
         kseqs, KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_mid_480", "rt_full_640", CACHE / "rtdetr_kitti_mid",
         kseqs, KittiAdapter, "mono"),
    ]
    if not args.kitti_only:
        db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
        nad = make_adapter(db)
        nseqs = [p.stem for p in
                 sorted((Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))]
        specs += [
            ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
             nseqs, nad, "mono"),
            ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
             nseqs, nad, "oracle"),
        ]

    summary = []
    for ds, det, cm, fm, dd, seqs, ad, geo in specs:
        dd = Path(dd)
        if not (dd / cm).exists():
            print(f"skip {ds}/{det}/{cm}: no cache"); continue
        d = build_b(dd, cm, fm, seqs, cfg, ad, geo, pb, cb)
        key = f"{ds}__{det}__{cm}to{fm}__{geo}".replace("-", "")
        d.to_pickle(run / f"planB__{key}.pkl")
        allc, allf = float(d.JB_cheap.sum()), float(d.JB_full.sum())
        dj = d.dJB.to_numpy()
        k = int(round(0.20 * len(d)))
        sel = np.zeros(len(d), bool); sel[np.argsort(-dj, kind="stable")[:k]] = True
        orc = float(np.where(sel, d.JB_full, d.JB_cheap).sum())
        summary.append({
            "dataset": ds, "detector": det, "cheap_mode": cm, "full_mode": fm,
            "geometry": geo, "n_frames": len(d), "n_seqs": int(d.seq.nunique()),
            "JB_all_cheap": allc, "JB_all_full": allf, "JB_oracle_at_20pct": orc,
            "oracle_reduction_frac": (allc - orc) / max(allc, 1e-9),
            "all_full_reduction_frac": (allc - allf) / max(allc, 1e-9),
            "frames_dJB_nonzero": int((np.abs(dj) > 1e-9).sum()),
            "frames_dJB_pos": int((dj > 1e-9).sum()), "frames_dJB_neg": int((dj < -1e-9).sum()),
            "plan_change_rate": float((d.same_planB == 0).mean()),
            "collision_rate_cheap": float(d.collB_cheap.mean()),
            "collision_rate_full": float(d.collB_full.mean()),
        })
        r = summary[-1]
        print(f"{key}: {len(d)} frames  plan changes {100*r['plan_change_rate']:.1f}%  "
              f"oracle@20% -{100*r['oracle_reduction_frac']:.2f}%  "
              f"all-full -{100*r['all_full_reduction_frac']:.2f}%  "
              f"|dJB|>0 {r['frames_dJB_nonzero']} (+{r['frames_dJB_pos']}/-{r['frames_dJB_neg']})")

    out = Path(RESULTS) / "final"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(summary)
    m.to_csv(out / "phase0f_planner_b_summary.csv", index=False)
    m.to_csv(run / "phase0f_planner_b_summary.csv", index=False)
    print(f"\nwrote {out / 'phase0f_planner_b_summary.csv'}\nwrote {run}")


if __name__ == "__main__":
    main()
