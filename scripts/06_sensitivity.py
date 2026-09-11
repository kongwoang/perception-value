#!/usr/bin/env python
"""Does the conclusion depend on how 'criticality' and 'perception error' were defined?

Re-runs the decisive comparisons under every reasonable alternative definition. A
result that only exists for one hand-tuned risk model is not a result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, geometry as G, predict, runmeta, tables   # noqa: E402
from rap import features as F                                     # noqa: E402
from rap.cache import DetCache                                    # noqa: E402
from rap.paths import CACHE                                       # noqa: E402
from rap.risk import RiskConfig                                   # noqa: E402

KEY_ARMS = ["B_uncertainty", "C_complexity", "D_criticality", "E_unc_crit",
            "F_all_visual", "G_all"]


def build(seqs, det_dir, cheap, full, model, cfg, geom_cache, with_features=True):
    frames = [tables.build_sequence(
        s, DetCache(det_dir / cheap / f"{s}.npz"), DetCache(det_dir / full / f"{s}.npz"),
        model, cfg, geom_cache[s], with_features=with_features) for s in seqs]
    return pd.concat(frames, ignore_index=True)


def compact(df: pd.DataFrame, model_name="gbm", quota=0.20) -> dict:
    v = df["value_task"].to_numpy()
    order = np.argsort(-v)
    cum = np.cumsum(v[order])
    tot = cum[-1]
    out = {
        "frac_value_gt0": float(np.mean(v > 1e-9)),
        "gain_top20": float(cum[int(0.2 * len(v)) - 1] / tot) if tot > 1e-12 else np.nan,
        "risk_reduction_rel": float(tot / df.risk_cheap.sum()) if df.risk_cheap.sum() > 0 else np.nan,
    }
    scored, oof = {}, {}
    for arm in KEY_ARMS:
        res = predict.loso(df, F.columns_for(arm), model_name, "value_task", "reg", arm=arm)
        sc = predict.score(res)
        scored[arm], oof[arm] = sc, res.pred
        out[f"rho_{arm}"] = sc["spearman"]
        out[f"rhoseq_{arm}"] = sc["per_seq_median"]
    for a, b in [("B_uncertainty", "E_unc_crit"), ("F_all_visual", "G_all")]:
        t = predict.paired_test(scored[a], scored[b])
        out[f"delta_{a}_to_{b}"] = t["delta_median"]
        out[f"p_{a}_to_{b}"] = t["wilcoxon_p"]
        out[f"nbetter_{a}_to_{b}"] = t["n_better"]
        out[f"n_{a}_to_{b}"] = t["n"]

    scores = {
        "highest_uncertainty": df["feat_binent_sum"].to_numpy(),
        "lowest_confidence": -df["feat_conf_mean"].to_numpy(),
        "scene_complexity": df["feat_n_det"].to_numpy(),
        "criticality_heuristic": df["feat_crit_sum"].to_numpy(),
        "learned_uncertainty": oof["B_uncertainty"],
        "learned_unc_crit": oof["E_unc_crit"],
        "learned_all_visual": oof["F_all_visual"],
        "learned_all": oof["G_all"],
    }
    b = budget.evaluate(df, scores, [quota], seeds=32)
    for _, r in b.iterrows():
        out[f"eta20_{r.policy}"] = r.eta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--crit_models", nargs="+",
                    default=["composite", "uniform", "corridor", "proximity", "ttc", "binary",
                             "composite_wide", "composite_narrow", "composite_near",
                             "composite_far", "composite_ttcheavy", "composite_proxheavy"])
    ap.add_argument("--iou_thrs", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--op_confs", nargs="+", type=float, default=[0.15, 0.25, 0.40])
    ap.add_argument("--fp_lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--errors", nargs="+", default=["miss", "soft_iou"])
    ap.add_argument("--class_aware", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    run = runmeta.new_run("sensitivity", vars(args))
    det = Path(args.det)
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    geom_cache = {s: G.sequence_geometry(s) for s in seqs}
    base = RiskConfig()

    configs = []
    for m in args.crit_models:
        configs.append(("crit_model", m, m, base))
    for t in args.iou_thrs:
        configs.append(("iou_thr", t, "composite", RiskConfig(iou_thr=t)))
    for c in args.op_confs:
        configs.append(("op_conf", c, "composite", RiskConfig(op_conf=c)))
    for l in args.fp_lambdas:
        configs.append(("fp_lambda", l, "composite", RiskConfig(fp_lambda=l)))
    for e in args.errors:
        configs.append(("error", e, "composite", RiskConfig(error=e)))
    for ca in args.class_aware:
        configs.append(("class_aware", ca, "composite", RiskConfig(class_aware=bool(ca))))

    feat_cache: dict[tuple, pd.DataFrame] = {}
    rows = []
    for axis, value, mname, cfg in configs:
        key = (mname, round(cfg.op_conf, 6))
        if key in feat_cache:
            df = tables.with_cached_features(
                build(seqs, det, args.cheap, args.full, G.CRITICALITY_MODELS[mname], cfg,
                      geom_cache, with_features=False),
                feat_cache[key])
        else:
            df = build(seqs, det, args.cheap, args.full, G.CRITICALITY_MODELS[mname], cfg,
                       geom_cache)
            feat_cache[key] = df[["seq", "frame"] + [c for c in df.columns
                                                     if c.startswith("feat_")]].copy()
        r = {"axis": axis, "value": value, "crit_model": mname,
             "iou_thr": cfg.iou_thr, "op_conf": cfg.op_conf, "error": cfg.error,
             "fp_lambda": cfg.fp_lambda, "class_aware": int(cfg.class_aware)}
        r.update(compact(df))
        rows.append(r)
        print(f"{axis:12s}={str(value):14s} rho_B={r['rho_B_uncertainty']:+.3f} "
              f"rho_E={r['rho_E_unc_crit']:+.3f} rho_G={r['rho_G_all']:+.3f}  "
              f"eta20 unc={r.get('eta20_learned_uncertainty', float('nan')):.3f} "
              f"unc+crit={r.get('eta20_learned_unc_crit', float('nan')):.3f}")
        pd.DataFrame(rows).to_csv(run / "sensitivity.csv", index=False)

    print("\nwrote", run)


if __name__ == "__main__":
    main()
