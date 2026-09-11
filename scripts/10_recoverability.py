#!/usr/bin/env python
"""Phase 0B kill test: is "cheap will fail" the same question as "extra compute will help"?

Runs entirely on the existing KITTI detection cache. Builds object-level fail/recover
labels, fits the three factors separately, and compares the factorised frame scores
against the baseline that would make the whole method unnecessary:
`uncertainty x criticality`.

If the simple product already matches the three-factor decomposition, there is no method
contribution and Phase 0B should stop here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, geometry as G, objects, predict, runmeta   # noqa: E402
from rap import features as F                                      # noqa: E402
from rap.cache import DetCache                                     # noqa: E402
from rap.paths import CACHE, PROCESSED                             # noqa: E402
from rap.risk import RiskConfig                                    # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]


def build_objects(det: Path, cheap: str, full: str, model, cfg, seqs) -> pd.DataFrame:
    return pd.concat(
        [objects.build_sequence(s, DetCache(det / cheap / f"{s}.npz"),
                                DetCache(det / full / f"{s}.npz"), model, cfg,
                                G.sequence_geometry(s)) for s in seqs],
        ignore_index=True)


def fit_factors(obj: pd.DataFrame, model_name: str) -> dict:
    """p_fail, p_recover|fail and p_gain, each leave-one-sequence-out."""
    a = obj[obj["cand"] >= 0].reset_index(drop=True)
    cols = objects.object_columns()
    chk = objects.assert_no_object_leakage
    out = {"index": a.index.to_numpy(), "table": a}

    r = predict.loso(a, cols, model_name, "cheap_fail", "clf", checker=chk)
    out["p_fail"] = r.pred
    out["auc_fail"] = predict.score(r)

    a = a.assign(_recover=a["full_ok"].astype(float))
    r = predict.loso(a, cols, model_name, "_recover", "clf", checker=chk,
                     fit_mask=a["cheap_fail"].to_numpy(bool))
    out["p_recover"] = r.pred
    sub = a["cheap_fail"].to_numpy(bool)
    out["auc_recover"] = {
        "auc": float(_safe_auc(a.loc[sub, "_recover"], r.pred[sub])),
        "n": int(sub.sum()), "base_rate": float(a.loc[sub, "_recover"].mean()),
        "_per_seq": _per_seq_auc(a[sub], r.pred[sub], "_recover"),
    }

    a = a.assign(_gain=(a["gain"] > 0).astype(float))
    r = predict.loso(a, cols, model_name, "_gain", "clf", checker=chk)
    out["p_gain"] = r.pred
    out["auc_gain"] = predict.score(r)
    return out


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def _per_seq_auc(df, pred, target):
    out = {}
    for s, idx in df.groupby("seq").groups.items():
        pos = df.index.get_indexer(idx)
        out[s] = float(_safe_auc(df.loc[idx, target], pred[pos]))
    return out


def frame_scores(obj_anchored: pd.DataFrame, parts: dict, frames: pd.DataFrame) -> dict:
    """Sum each object-level product up to the frame, aligned to the frame table."""
    a = obj_anchored
    key = pd.MultiIndex.from_arrays([frames["seq"], frames["frame"]])
    c = a["o_c_hat"].to_numpy()
    conf = a["o_conf"].to_numpy()

    terms = {
        # the baseline that would make a recoverability model unnecessary
        "unc_x_crit_obj": (1.0 - conf) * c,
        "M3_fail_x_crit": parts["p_fail"] * c,
        "M4_gain_x_crit": parts["p_gain"] * c,
        "M5_fail_x_recover_x_crit": parts["p_fail"] * parts["p_recover"] * c,
        # ablation: recoverability alone, criticality dropped
        "recover_only": parts["p_fail"] * parts["p_recover"],
        # ablation: perfect factor order check -- p_gain should ~ p_fail * p_recover
        "M5_gain_direct_x_crit": parts["p_gain"] * c,
    }
    out = {}
    for name, v in terms.items():
        s = pd.Series(v, index=pd.MultiIndex.from_arrays([a["seq"], a["frame"]])) \
              .groupby(level=[0, 1]).sum()
        out[name] = s.reindex(key, fill_value=0.0).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--frames", default=str(PROCESSED / "frames_composite.pkl"))
    ap.add_argument("--model", default="gbm")
    ap.add_argument("--crit_model", default="composite")
    ap.add_argument("--op_conf", type=float, default=0.25)
    ap.add_argument("--lat_cheap", type=float, default=10.145)
    ap.add_argument("--lat_full", type=float, default=16.024)
    ap.add_argument("--energy_cheap", type=float, default=24.45)
    ap.add_argument("--energy_full", type=float, default=39.04)
    ap.add_argument("--tag", default="recoverability")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig(op_conf=args.op_conf)
    model = G.CRITICALITY_MODELS[args.crit_model]
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    frames = pd.read_pickle(args.frames).reset_index(drop=True)

    print(f"building object table over {len(seqs)} sequences ...")
    obj = build_objects(det, args.cheap, args.full, model, cfg, seqs)
    obj.to_pickle(run / "objects.pkl")
    cov = objects.anchor_coverage(obj)
    print(json.dumps(cov, indent=2))

    print("\n== failure is not recoverability ==")
    fail = obj["cheap_fail"]
    print(f"  P(cheap fails)                      {fail.mean():.3f}  (n={len(obj):,})")
    print(f"  P(full recovers | cheap fails)      {obj.loc[fail, 'full_ok'].mean():.3f}")
    print(f"  P(full also fails | cheap fails)    {1 - obj.loc[fail, 'full_ok'].mean():.3f}")
    print(f"  P(full breaks it | cheap succeeded) {1 - obj.loc[~fail & obj.has_gt, 'full_ok'].mean():.3f}")

    print("\nfitting the three factors (leave-one-sequence-out) ...")
    parts = fit_factors(obj, args.model)
    a = parts["table"]
    print(f"  p_fail      AUC {parts['auc_fail']['auc']:.3f}  "
          f"(per-seq median {parts['auc_fail']['per_seq_median']:.3f})")
    print(f"  p_recover   AUC {parts['auc_recover']['auc']:.3f} on the {parts['auc_recover']['n']:,} "
          f"failures, base rate {parts['auc_recover']['base_rate']:.3f}")
    print(f"  p_gain      AUC {parts['auc_gain']['auc']:.3f}  "
          f"(per-seq median {parts['auc_gain']['per_seq_median']:.3f})")

    # how different are the two factors, really?
    pf, pr = parts["p_fail"], parts["p_recover"]
    sub = a["cheap_fail"].to_numpy(bool)
    print(f"\n  corr(p_fail, p_recover) over all candidates   {np.corrcoef(pf, pr)[0,1]:+.3f}")
    print(f"  corr(p_fail, p_recover) over failures only    {np.corrcoef(pf[sub], pr[sub])[0,1]:+.3f}")
    print(f"  corr(p_fail, 1-conf)                          "
          f"{np.corrcoef(pf, 1-a['o_conf'].to_numpy())[0,1]:+.3f}")
    print(f"  corr(p_recover, 1-conf)                       "
          f"{np.corrcoef(pr, 1-a['o_conf'].to_numpy())[0,1]:+.3f}")

    # ---- frame-level scores ----
    scores = frame_scores(a, parts, frames)

    # frame-level references, reusing the Phase-0 arms unchanged
    for arm, label in [("B_uncertainty", "M0_uncertainty"), ("D_criticality", "M1_criticality"),
                       ("G_all", "M2_monolithic")]:
        r = predict.loso(frames, F.columns_for(arm), args.model, "value_task", "reg", arm=arm)
        scores[label] = r.pred
    unc = frames["feat_binent_sum"].to_numpy()
    crit = frames["feat_crit_sum"].to_numpy()
    scores["unc_x_crit_frame"] = unc * crit

    print("\n== budgeted selection at matched compute ==")
    b = budget.evaluate(frames, scores, QUOTAS, mode="pooled")
    b = budget.add_compute_columns(b, args.lat_cheap, args.lat_full,
                                   args.energy_cheap, args.energy_full)
    b.to_csv(run / "budget.csv", index=False)
    order = ["random", "M0_uncertainty", "M1_criticality", "unc_x_crit_frame",
             "unc_x_crit_obj", "M2_monolithic", "M3_fail_x_crit", "M4_gain_x_crit",
             "M5_fail_x_recover_x_crit", "M5_gain_direct_x_crit", "recover_only", "oracle"]
    piv = b[b["mode"] == "pooled"].pivot(index="policy", columns="quota", values="eta")
    print(piv.reindex([o for o in order if o in piv.index]).round(3).to_string())

    # ---- per-sequence consistency of the decisive comparison ----
    print("\n== per-sequence, 20% quota ==")
    pos = {s: np.flatnonzero(frames["seq"].to_numpy() == s) for s in frames["seq"].unique()}
    rows = []
    for s, idx in pos.items():
        sub_f = frames.iloc[idx].reset_index(drop=True)
        sc = {k: np.asarray(v)[idx] for k, v in scores.items()}
        r = budget.evaluate(sub_f, sc, [0.20], seeds=8, mode="pooled")
        r["seq"] = s
        rows.append(r)
    per_seq = pd.concat(rows, ignore_index=True)
    per_seq.to_csv(run / "budget_per_sequence.csv", index=False)
    wide = per_seq.pivot(index="seq", columns="policy", values="eta")

    decisive = []
    for base, cand in [("unc_x_crit_obj", "M5_fail_x_recover_x_crit"),
                       ("unc_x_crit_frame", "M5_fail_x_recover_x_crit"),
                       ("M3_fail_x_crit", "M5_fail_x_recover_x_crit"),
                       ("M2_monolithic", "M5_fail_x_recover_x_crit"),
                       ("M0_uncertainty", "M5_fail_x_recover_x_crit"),
                       ("unc_x_crit_obj", "M4_gain_x_crit")]:
        if base not in wide or cand not in wide:
            continue
        d = (wide[cand] - wide[base]).dropna()
        w = stats.wilcoxon(d, alternative="greater") if len(d) >= 5 and d.abs().sum() > 0 else None
        rec = {"baseline": base, "candidate": cand, "n_seq": int(len(d)),
               "n_better": int((d > 0).sum()), "median_delta": float(d.median()),
               "wilcoxon_p": float(w.pvalue) if w else np.nan}
        decisive.append(rec)
        print(f"  {base:24s} -> {cand:26s} better in {rec['n_better']:2d}/{rec['n_seq']}  "
              f"median deta={rec['median_delta']:+.3f}  p={rec['wilcoxon_p']:.4g}")

    summary = {"anchor_coverage": cov, "factors": {
        "auc_fail": parts["auc_fail"]["auc"], "auc_recover": parts["auc_recover"]["auc"],
        "auc_gain": parts["auc_gain"]["auc"],
        "p_recover_given_fail": float(obj.loc[fail, "full_ok"].mean()),
        "corr_pfail_precover": float(np.corrcoef(pf, pr)[0, 1]),
    }, "decisive": decisive}
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
