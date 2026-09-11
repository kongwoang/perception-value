#!/usr/bin/env python
"""Phase 0E Stages 9-17: the core matrix across every configuration.

One row per (dataset, detector, fidelity pair, geometry, task). Everything the final
verdict rests on is computed here so the whole claim can be audited from one CSV.
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
from rap.paths import CACHE, RESULTS                                             # noqa: E402
from rap.risk import RiskConfig                                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                              # noqa: E402
_pm = import_module("50_percep_metrics")

QUOTAS = [0.10, 0.20, 0.30, 0.50]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}
NOCHK = lambda c: None


def eta(df, score, col, quota=0.20):
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def topk_overlap(a, b, frac):
    k = int(round(frac * len(a)))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / max(k, 1)


def trivial_heuristics(d):
    z = d.feat_min_z_corridor.to_numpy() if "feat_min_z_corridor" in d else np.zeros(len(d))
    return {
        "ego speed": d.v_ego.to_numpy(),
        "empty-frame": d.empty_cheap.to_numpy().astype(float),
        "few detections": -d.n_cheap.to_numpy().astype(float),
        "n detections": d.n_cheap.to_numpy().astype(float),
        "low mean confidence": -d.conf_mean.to_numpy(),
        "uncertainty": d.unc_sum.to_numpy(),
        "criticality": d.crit_sum.to_numpy(),
        "closest object": -z,
        "speed x empty": d.v_ego.to_numpy() * (1 + 3 * d.empty_cheap.to_numpy()),
        "speed + detections": stats.rankdata(d.v_ego) + stats.rankdata(d.n_cheap),
        "speed x closest": d.v_ego.to_numpy() * np.clip(1 / np.maximum(z, 1.0), 0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="core_matrix")
    ap.add_argument("--skip_multimetric", action="store_true")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()

    kseqs = [p.stem for p in sorted((Path(CACHE / "det") / "cheap_320").glob("*.npz"))]
    db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
    nad = make_adapter(db)
    nseqs = [p.stem for p in sorted((Path(CACHE / "nusc_det_tv") / "ns_cheap_320").glob("*.npz"))]

    SPECS = [
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "oracle"),
        ("KITTI", "YOLOv8s", "cheap_384", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_512", "full_640", CACHE / "det512", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_cheap_320", "rt_full_640", CACHE / "rtdetr_kitti",
         kseqs, decision.KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_mid_480", "rt_full_640", CACHE / "rtdetr_kitti_mid",
         kseqs, decision.KittiAdapter, "mono"),
        ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
         nseqs, nad, "mono"),
        ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
         nseqs, nad, "oracle"),
    ]

    rows, tables = [], {}
    for ds, detname, cm, fm, dd, seqs, ad, geo in SPECS:
        dd = Path(dd)
        if not (dd / cm).exists():
            print(f"skip {ds}/{detname}/{cm}: no cache"); continue
        key = f"{ds}|{detname}|{cm}->{fm}|{geo}"
        d = decision.build(dd, cm, fm, seqs, cfg, pp, cp, G.PRIMARY,
                           range_source=geo, adapter=ad)
        d = decision.add_perception_gain(d, dd, cm, fm, seqs, cfg, adapter=ad)
        prim = _pm.primitives(dd, cm, fm, seqs, cfg, ad, cfg.min_gt_height)
        d = d.merge(prim, on=["seq", "frame"], how="left", validate="one_to_one")
        tables[key] = d
        print(f"{key}: {len(d)} frames")

        jl = (d.J_cheap - d.J_full).to_numpy()
        jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
        overlap20 = topk_overlap(jl, jt, 0.20)
        heur = trivial_heuristics(d)

        for tname, col in TASKS.items():
            dj = (d[col[0]] - d[col[1]]).to_numpy()
            act = "same_action" if tname == "longitudinal" else "lat_same_action"
            improved = d["dE"] > 0
            best_metric = max(PM.METRICS, key=lambda m: eta(d, d[f"dE_{m}"], col))
            hh = {k: eta(d, v, col) for k, v in heur.items()}
            bh = max(hh, key=lambda k: hh[k])
            mm = np.nan
            if not args.skip_multimetric:
                feats = _pm.oracle_features(d)
                dd2 = d.copy()
                for c in feats:
                    dd2[c] = pd.to_numeric(dd2[c], errors="coerce").fillna(0.0)
                dd2["_dJ"] = dj
                mm = eta(dd2, predict.loso(dd2, feats, "gbm", "_dJ", "reg",
                                           checker=NOCHK).pred, col)
            r = {
                "dataset": ds, "detector": detname, "cheap_mode": cm, "full_mode": fm,
                "task": tname, "geometry": geo,
                "perception_metric": "E1_fn_only", "best_perception_metric": best_metric,
                "n_sequences": int(d.seq.nunique()), "n_frames": len(d),
                "action_change_rate": float((d[act] == 0).mean()),
                "improved_same_action_rate": float((improved & (d[act] == 1)).sum() /
                                                   max(improved.sum(), 1)),
                "harmful_full_rate": float((dj < -1e-9).mean()),
                "corr_deltaE_deltaJ": float(stats.spearmanr(d["dE"], dj).correlation),
                "eta_random_20": float(np.mean([eta(d, np.random.default_rng(k).random(len(d)), col)
                                                for k in range(6)])),
                "eta_uncertainty_20": eta(d, d.unc_sum, col),
                "eta_criticality_20": eta(d, d.crit_sum, col),
                "eta_perception_oracle_20": eta(d, d["dE"], col),
                "eta_best_perception_metric_20": eta(d, d[f"dE_{best_metric}"], col),
                "eta_multimetric_perception_oracle_20": mm,
                "eta_decision_oracle_20": 1.0,
                "top20_cross_task_overlap": overlap20,
                "best_trivial_heuristic": bh, "best_trivial_eta20": hh[bh],
            }
            for q in QUOTAS:
                r[f"eta_perception_oracle_{int(q*100)}"] = eta(d, d["dE"], col, q)
            rows.append(r)
            print(f"   {tname:13s} corr={r['corr_deltaE_deltaJ']:+.3f} "
                  f"etaE={r['eta_perception_oracle_20']:+.3f} "
                  f"multi={mm if np.isnan(mm) else round(mm,3)} "
                  f"best-trivial={bh} {hh[bh]:+.3f} overlap20={overlap20:.3f}")

    out = Path(RESULTS) / "final"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(out / "core_matrix.csv", index=False)
    m.to_csv(run / "core_matrix.csv", index=False)
    for k, v in tables.items():
        v.to_pickle(run / (k.replace("|", "__").replace("->", "to").replace("/", "_") + ".pkl"))
    print(f"\nwrote {out/'core_matrix.csv'} ({len(m)} rows)")
    print("\nwrote", run)


if __name__ == "__main__":
    main()
