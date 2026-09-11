#!/usr/bin/env python
"""Phase 0E Stages 2-3: perception-metric robustness, and the multi-metric perception oracle.

Stage 2 asks whether the claim survives every reasonable definition of "perception
improved". Stage 3 answers the reviewer attack "you picked the wrong metric" by giving an
oracle a *rich* description of how perception changed -- and nothing about the decision --
and measuring how far that gets.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, decision, geometry as G, planner as P, predict, runmeta  # noqa: E402
from rap import percep_metrics as PM                                             # noqa: E402
from rap.cache import DetCache                                                   # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE                                                      # noqa: E402
from rap.risk import RiskConfig                                                   # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}
DIST_BINS = [0, 15, 30, 50, 1000]
NOCHK = lambda c: None


def eta(df, score, col, quota=0.20):
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def primitives(det_dir, cheap, full, seqs, cfg, adapter, min_h) -> pd.DataFrame:
    """Per-frame perception-loss primitives for both modes, plus distance-binned recall."""
    from rap.kitti import TYPE_TO_COARSE
    rows = []
    for s in seqs:
        geom = adapter.geometry(s)
        crit = G.criticality_for(geom, G.PRIMARY)
        c = DetCache(Path(det_dir) / cheap / f"{s}.npz")
        f = DetCache(Path(det_dir) / full / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            sel = geom["frame"] == fr
            g, cg = geom[sel], crit[sel]
            ok = (g["y2"] - g["y1"]) >= min_h
            g, cg = g[ok], cg[ok]
            gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                  if len(g) else np.zeros((0, 4)))
            gcls = np.array([TYPE_TO_COARSE.get(t, "vehicle") for t in g["type"]])
            rec = {"seq": s, "frame": fr}
            prim = {}
            for tag, cache, idx in (("cheap", c, i), ("full", f, f.index[fr])):
                pr = PM.frame_losses(gt, gcls, cg, cache.det(idx), cfg)
                prim[tag] = pr
                for k, v in pr.items():
                    rec[f"{tag}_{k}"] = v
            for name, w in PM.METRICS.items():
                rec[f"dE_{name}"] = PM.combine(prim["cheap"], w) - PM.combine(prim["full"], w)
            # distance-binned recall improvement: which ranges FULL rescued
            if len(g):
                from rap.risk import match as _m
                hits = {}
                for tag, cache, idx in (("cheap", c, i), ("full", f, f.index[fr])):
                    d = cache.det(idx)
                    k = d["conf"] >= cfg.op_conf
                    b, _ = _m(gt, gcls, d["xyxy"][k].astype(float), d["conf"][k],
                              d["coarse"][k], cfg)
                    hits[tag] = b >= cfg.iou_thr
                dist = g["long_near"]
                for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
                    m = (dist >= lo) & (dist < hi)
                    rec[f"drecall_{lo}_{hi}"] = float(
                        (hits["full"][m].sum() - hits["cheap"][m].sum())) if m.any() else 0.0
            else:
                for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
                    rec[f"drecall_{lo}_{hi}"] = 0.0
            rows.append(rec)
    df = pd.DataFrame(rows)
    for tag in ("cheap", "full"):
        df[f"{tag}_n_det"] = df[f"{tag}_n_det"].astype(float)
    df["d_n_det"] = df["full_n_det"] - df["cheap_n_det"]
    df["d_conf_sum"] = df["full_conf_sum"] - df["cheap_conf_sum"]
    df["d_mean_iou"] = df["full_mean_iou_matched"] - df["cheap_mean_iou_matched"]
    return df


def oracle_features(df: pd.DataFrame) -> list[str]:
    """Rich perception-change description. No decision cost, no action, no planner state."""
    cols = [c for c in df.columns if c.startswith(("dE_", "drecall_"))]
    cols += ["d_n_det", "d_conf_sum", "d_mean_iou",
             "cheap_fn", "cheap_fp", "cheap_loc", "cheap_n_det", "cheap_mean_iou_matched",
             "full_fn", "full_fp", "full_loc", "full_n_det"]
    return [c for c in cols if c in df.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="percep_metrics")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    cfg = RiskConfig()

    det = Path(CACHE / "det")
    kseqs = [p.stem for p in sorted((det / "cheap_320").glob("*.npz"))]
    db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
    nad = make_adapter(db)
    nd = Path(CACHE / "nusc_det_tv")
    nseqs = [p.stem for p in sorted((nd / "ns_cheap_320").glob("*.npz"))]

    configs = {}
    for label, dd, cm, fm, seqs, ad in [
        ("KITTI 320->640", det, "cheap_320", "full_640", kseqs, decision.KittiAdapter),
        ("KITTI 512->640", CACHE / "det512", "cheap_512", "full_640", kseqs, decision.KittiAdapter),
        ("nuScenes 320->640", nd, "ns_cheap_320", "ns_full_640", nseqs, nad),
    ]:
        d = decision.build(Path(dd), cm, fm, seqs, cfg, P.PlannerParams(), P.CostParams(),
                           G.PRIMARY, adapter=ad)
        pr = primitives(dd, cm, fm, seqs, cfg, ad, cfg.min_gt_height)
        configs[label] = d.merge(pr, on=["seq", "frame"], how="left", validate="one_to_one")
        print(f"{label}: {len(configs[label])} frames")

    # ---------------- Stage 2: every perception metric ----------------
    rows = []
    for cname, d in configs.items():
        for tname, col in TASKS.items():
            dj = (d[col[0]] - d[col[1]]).to_numpy()
            for mname in PM.METRICS:
                de = d[f"dE_{mname}"].to_numpy()
                r = {"config": cname, "task": tname, "metric": mname,
                     "corr": float(stats.spearmanr(de, dj).correlation)}
                for q in QUOTAS:
                    r[f"eta@{int(q*100)}"] = eta(d, de, col, q)
                rows.append(r)
    m2 = pd.DataFrame(rows)
    m2.to_csv(run / "metric_robustness.csv", index=False)
    print("\n=== STAGE 2: eta@20 by perception metric ===")
    piv = m2.pivot_table(index="metric", columns=["config", "task"], values="eta@20")
    print(piv.round(3).to_string())

    # ---------------- Stage 3: multi-metric perception oracle ----------------
    print("\n=== STAGE 3: multi-metric perception ORACLE (diagnostic upper bound) ===")
    rows3 = []
    for cname, d in configs.items():
        feats = oracle_features(d)
        d = d.copy()
        for c in feats:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        for tname, col in TASKS.items():
            d["_dJ"] = d[col[0]] - d[col[1]]
            best_single = max((eta(d, d[f"dE_{m}"], col) for m in PM.METRICS))
            r = {"config": cname, "task": tname, "n_features": len(feats),
                 "best_single_metric_eta20": best_single}
            for mdl in ("linear", "gbm"):
                pred = predict.loso(d, feats, mdl, "_dJ", "reg", checker=NOCHK).pred
                r[f"{mdl}_spearman"] = float(stats.spearmanr(pred, d["_dJ"]).correlation)
                for q in QUOTAS:
                    r[f"{mdl}_eta@{int(q*100)}"] = eta(d, pred, col, q)
            rows3.append(r)
            print(f"  {cname:20s} {tname:13s} best-single={best_single:+.3f}  "
                  f"linear={r['linear_eta@20']:+.3f}  gbm={r['gbm_eta@20']:+.3f}  "
                  f"(gbm rho={r['gbm_spearman']:+.3f})")
    m3 = pd.DataFrame(rows3)
    m3.to_csv(run / "multimetric_oracle.csv", index=False)

    for cname, d in configs.items():
        d.to_pickle(run / f"{cname.replace(' ', '_').replace('->', 'to')}.pkl")
    (run / "summary.json").write_text(json.dumps(
        {"stage2": rows, "stage3": rows3}, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
