#!/usr/bin/env python
"""Phase 0G, checks on the deployable gate (84) before any of its numbers are written.

On braking with monocular geometry the gate captured 0.465 of the oracle prize at 20%, above every
diagnostic signal, while on oracle geometry it reached 0.175.  A result that convenient gets:
  1. confidence intervals on the PAIRED difference against random and against the cheap-side
     uncertainty the eta tables already use (`unc_sum`), not intervals against zero;
  2. escalating only frames the gate predicts to be helped, at most k -- V is sign-varying, so a
     ranking that fills every slot also buys escalations it expects to hurt;
  3. what the fitted models lean on, so the result can be explained rather than only quoted.

Leakage was checked by reading code, not by this script: the cached geometry the features use is
`mono.predicted_geometry` from the cheap boxes, camera calibration and the previous frame's cheap
boxes (scripts/40_nusc_detect.py), and the mono decision tables consume exactly those arrays
(`decision._apply_range_source` returns them untouched for source "mono").  Under monocular
geometry the gate therefore sees the inputs of the cheap decision itself -- information a vehicle
has before escalating.  Under oracle geometry the decision uses GT ranges the gate cannot see.
"""
from __future__ import annotations

import argparse, glob, importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta                                                # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.tables import feature_columns                                          # noqa: E402

_spec = importlib.util.spec_from_file_location("gate84", ROOT / "scripts" / "84_deployable_gate.py")
g84 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g84)

QUOTAS = g84.QUOTAS
TIE_SEEDS = g84.TIE_SEEDS
# name -> (column or model, deployable)
TABLE_SIGNALS = {"unc_sum (eta-table uncertainty)": ("unc_sum", True),
                 "feat_ent_mean (gate uncertainty)": ("feat_ent_mean", True),
                 "crit_sum (GT geometry)": ("crit_sum", False),
                 "dE_E6_risk_weighted (needs FULL)": ("dE_E6_risk_weighted", False)}
POS_ONLY = " [escalate only pred>0]"


def gain(v, score, k, seed, pos_only=False):
    """Total cost removed by escalating the top-k frames by `score` (= sum of V over them)."""
    score = np.asarray(score, float)
    o = np.lexsort((np.random.default_rng(seed).random(len(score)), -score))[:k]
    if pos_only:
        o = o[score[o] > 0]
    return float(v[o].sum()), len(o)


def prize(v, k):
    return float(np.sort(v)[::-1][:k].sum())


def point(v, score, q, pos_only=False):
    k = int(round(q * len(v)))
    p = prize(v, k)
    g = [gain(v, score, k, s, pos_only) for s in range(TIE_SEEDS)]
    return np.mean([x for x, _ in g]) / p, np.mean([n for _, n in g])


def paired_boot(v, scenes, scores, q, nboot, rng):
    """Scene bootstrap of eta for every signal, and random's expected eta, on the same draws."""
    uniq = np.unique(scenes)
    idx = {u: np.flatnonzero(scenes == u) for u in uniq}
    p0 = prize(v, int(round(q * len(v))))
    draws = {n: [] for n in scores}
    draws["random"] = []
    dropped = 0
    for _ in range(nboot):
        t = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        vt = v[t]
        k = int(round(q * len(t)))
        p = prize(vt, k)
        if p <= max(1e-12, 0.25 * p0):
            dropped += 1
            continue
        draws["random"].append(k / len(t) * vt.sum() / p)      # expectation over random k-subsets
        for n, sc in scores.items():
            draws[n].append(gain(vt, sc[t], k, 0, n.endswith(POS_ONLY))[0] / p)
    return {n: np.asarray(x) for n, x in draws.items()}, dropped


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def importances(d, fcols, mdl, top=15):
    m = predict.make_model(mdl, "reg", 0)
    X = np.nan_to_num(d[fcols].to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    m.fit(X, d["_V"].to_numpy(np.float64))
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    if hasattr(est, "feature_importances_"):
        w, kind = np.asarray(est.feature_importances_, float), "impurity importance"
    elif hasattr(est, "coef_"):
        w, kind = np.ravel(est.coef_).astype(float), "standardised coefficient"
    else:
        return []
    o = np.argsort(-np.abs(w))[:top]
    return [{"feature": fcols[i], "weight": float(w[i]), "kind": kind, "rank": r + 1}
            for r, i in enumerate(o)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_tv"))
    ap.add_argument("--mode", default="ns_cheap_320")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--tag", default="deployable_gate_checks")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    rng = np.random.default_rng(0)

    feats = g84.build_features(Path(args.det), args.mode, RiskConfig())
    fcols = feature_columns(feats)                      # assert_no_leakage runs here
    tm = pd.read_csv(Path(CACHE) / "nusc_token_map.csv")
    cm = sorted(glob.glob(str(ROOT / "results/raw/*core_matrix_postreview")))
    cm = Path(cm[-1]) if cm else ROOT / "results/raw/20260912_071140_core_matrix"
    ref = pd.read_csv(Path(RESULTS) / "final" / "phase0g_deployable_gate.csv")
    print(f"  {len(feats)} frames, {len(fcols)} cheap-side features; decision tables {cm.name}")

    rows, imp = [], []
    for variant in ("oracle", "mono"):
        b = pd.read_pickle(cm / f"nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{variant}.pkl")
        sfx = "" if variant == "oracle" else f"_{variant}"
        p = pd.read_csv(Path(CACHE) / "planner_d" / f"planC_vs_truth{sfx}.csv")
        extra = [c for c, _ in TABLE_SIGNALS.values() if c in b and c not in feats]
        base = b[["seq", "frame", "J_cheap", "J_full"] + extra].merge(
            feats, on=["seq", "frame"], how="inner", validate="one_to_one")
        base = base.merge(tm, on=["seq", "frame"], how="left")
        targets = {"brake": (base, "J_cheap", "J_full"),
                   "plan": (base.merge(p[["sample_token", "JC_ade_cheap", "JC_ade_full"]],
                                       on="sample_token", how="inner"), "JC_ade_cheap", "JC_ade_full")}
        for tname, (d, ccol, fcol) in targets.items():
            d = d.reset_index(drop=True).copy()
            d["_V"] = d[ccol] - d[fcol]
            for c in fcols:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
            v = d["_V"].to_numpy(float)
            scores, deploy = {}, {}
            for n, (c, dep) in TABLE_SIGNALS.items():
                if c in d:
                    scores[n], deploy[n] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(), dep
            for mdl in ("linear", "gbm"):
                pred = predict.loso(d, fcols, mdl, "_V", "reg").pred
                scores[f"{mdl} (OOF by scene)"] = pred
                scores[f"{mdl} (OOF by scene){POS_ONLY}"] = pred
                deploy[f"{mdl} (OOF by scene)"] = True
                deploy[f"{mdl} (OOF by scene){POS_ONLY}"] = True
                for r in importances(d, fcols, mdl):
                    imp.append({"geometry": variant, "target": tname, "model": mdl, **r})
            for q in QUOTAS:
                draws, dropped = paired_boot(v, d.seq.to_numpy(), scores, q, args.nboot, rng)
                k = int(round(q * len(v)))
                rnd = k / len(v) * v.sum() / prize(v, k)
                rows.append({"geometry": variant, "target": tname, "signal": "random",
                             "deployable": True, "quota": q, "eta": rnd, "n_escalated": k,
                             "boot_dropped": dropped, "n_frames": len(d)})
                for n, sc in scores.items():
                    e, nesc = point(v, sc, q, n.endswith(POS_ONLY))
                    dr = draws[n] - draws["random"]
                    r = {"geometry": variant, "target": tname, "signal": n,
                         "deployable": deploy[n], "quota": q, "eta": e, "n_escalated": nesc,
                         "lo": ci(draws[n])[0], "hi": ci(draws[n])[1],
                         "minus_random": float(np.mean(dr)), "minus_random_lo": ci(dr)[0],
                         "minus_random_hi": ci(dr)[1], "p_le_random": float(np.mean(dr <= 0)),
                         "boot_dropped": dropped, "n_frames": len(d)}
                    u = "unc_sum (eta-table uncertainty)"
                    if u in draws and n != u:
                        du = draws[n] - draws[u]
                        r.update({"minus_unc_sum": float(np.mean(du)), "minus_unc_sum_lo": ci(du)[0],
                                  "minus_unc_sum_hi": ci(du)[1]})
                    rows.append(r)
                # wiring check: the refit gate must reproduce 84's point estimates
                for mdl in ("linear", "gbm"):
                    old = ref[(ref.geometry == variant) & (ref.target == tname) &
                              (ref.signal == f"{mdl} (OOF by scene)") & (ref.quota.round(2) == round(q, 2))]
                    new = rows[-len(scores) - 1:]
                    new = [x for x in new if x["signal"] == f"{mdl} (OOF by scene)"][0]["eta"]
                    if len(old) and abs(float(old.eta.iloc[0]) - new) > 1e-6:
                        print(f"  !! {variant} {tname} {mdl} @{q}: 84 gave {float(old.eta.iloc[0]):.4f}, "
                              f"refit gives {new:.4f}")
            for r in [x for x in rows if x["geometry"] == variant and x["target"] == tname
                      and abs(x["quota"] - 0.2) < 1e-9]:
                extra_s = (f"  vs random {r['minus_random']:+.3f} [{r['minus_random_lo']:+.3f},"
                           f"{r['minus_random_hi']:+.3f}]" if "minus_random" in r else "")
                print(f"  {variant:6s} {tname:5s} {r['signal']:48s} @20 {r['eta']:+.3f} "
                      f"(n={r['n_escalated']:.0f}){extra_s}")

    out = Path(RESULTS) / "final"
    pd.DataFrame(rows).to_csv(out / "phase0g_deployable_gate_checks.csv", index=False)
    pd.DataFrame(imp).to_csv(out / "phase0g_deployable_gate_importance.csv", index=False)
    pd.DataFrame(rows).to_csv(run / "phase0g_deployable_gate_checks.csv", index=False)
    pd.DataFrame(imp).to_csv(run / "phase0g_deployable_gate_importance.csv", index=False)
    print("  wrote", out / "phase0g_deployable_gate_checks.csv")


if __name__ == "__main__":
    main()
