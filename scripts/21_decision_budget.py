#!/usr/bin/env python
"""Phase 0C: budgeted decision-aware allocation, plus the planner and cost sweeps."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, geometry as G, planner as P, predict, runmeta   # noqa: E402
from rap.paths import CACHE, PROCESSED                                   # noqa: E402
from rap.risk import RiskConfig                                          # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                      # noqa: E402
_dv = import_module("20_decision_value")

QUOTAS = [0.10, 0.20, 0.30, 0.50]
VIS = ["feat_binent_sum", "feat_conf_mean", "feat_n_det", "feat_crit_sum"]
DEC = ["feat_dec_margin", "feat_bdist", "feat_a_req_cheap", "feat_v_ego",
       "feat_n_corridor", "feat_min_z_corridor"]
NOCHK = lambda c: None


def eta_frame(df, score, quota, col=("J_cheap", "J_full")):
    sel = budget.select_pooled(np.asarray(score, float), quota)
    tot = budget.total_risk(df, sel, col)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, budget.select_pooled((df[col[0]] - df[col[1]]).to_numpy(), quota), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def policies(df):
    out = {
        "random": np.random.default_rng(0).random(len(df)),
        "confidence (low first)": -df["feat_conf_mean"].to_numpy(),
        "uncertainty": df["feat_binent_sum"].to_numpy(),
        "scene complexity": df["feat_n_det"].to_numpy(),
        "criticality": df["feat_crit_sum"].to_numpy(),
        "uncertainty x criticality": df["feat_binent_sum"].to_numpy() * df["feat_crit_sum"].to_numpy(),
        "PERCEPTION-GAIN ORACLE": df["dE"].to_numpy(),
        "DECISION-VALUE ORACLE": df["dJ"].to_numpy(),
    }
    for name, cols in [("decision sensitivity (learned)", DEC),
                       ("visual+criticality (learned)", VIS),
                       ("all cheap-side (learned)", VIS + DEC)]:
        out[name] = predict.loso(df, cols, "gbm", "dJ", "reg", checker=NOCHK).pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--frames", default=str(PROCESSED / "frames_composite.pkl"))
    ap.add_argument("--lat_cheap", type=float, default=10.145)
    ap.add_argument("--lat_full", type=float, default=16.024)
    ap.add_argument("--tag", default="decision_budget")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig()
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    frames = pd.read_pickle(args.frames).reset_index(drop=True)

    def make(pp, cp):
        d = _dv.build(det, args.cheap, args.full, seqs, pp, cp, cfg, G.PRIMARY)
        d = d.merge(frames[["seq", "frame", "err_std_cheap", "err_std_full",
                            "feat_binent_sum", "feat_crit_sum", "feat_conf_mean", "feat_n_det"]],
                    on=["seq", "frame"], how="left", validate="one_to_one")
        d["dE"] = d["err_std_cheap"] - d["err_std_full"]
        d["dJ"] = d["V_dec"]
        return d

    df = make(P.PLANNERS["default"], P.COSTS["default"])
    df.to_pickle(run / "decision.pkl")
    pol = policies(df)

    rows = []
    for name, sc in pol.items():
        for q in QUOTAS:
            sel = budget.select_pooled(np.asarray(sc, float), q)
            rows.append({"policy": name, "quota": q, "eta": eta_frame(df, sc, q),
                         "total_J": budget.total_risk(df, sel, ("J_cheap", "J_full")),
                         "collisions": float(np.sum(np.where(sel, df.collision_full, df.collision_cheap))),
                         "shortfall": float(np.sum(np.where(sel, df.shortfall_full, df.shortfall_cheap))),
                         "excess": float(np.sum(np.where(sel, df.excess_full, df.excess_cheap))),
                         "extra_ms": q * len(df) * (args.lat_full - args.lat_cheap)})
    b = pd.DataFrame(rows)
    b["J_reduction"] = float(df.J_cheap.sum()) - b["total_J"]
    b["J_per_extra_s"] = b["J_reduction"] / (b["extra_ms"] / 1000.0)
    b.to_csv(run / "budget.csv", index=False)
    print("\n=== decision cost captured (eta), all-cheap J =", f"{df.J_cheap.sum():.0f}",
          " all-full J =", f"{df.J_full.sum():.0f} ===")
    print(b.pivot(index="policy", columns="quota", values="eta").round(3).to_string())
    print("\n=== collisions remaining at each quota (all-cheap:",
          f"{df.collision_cheap.sum():.0f}, all-full: {df.collision_full.sum():.0f}) ===")
    print(b.pivot(index="policy", columns="quota", values="collisions").round(0).to_string())

    # per-sequence consistency of the decisive comparisons
    pos = {s: np.flatnonzero(df["seq"].to_numpy() == s) for s in df["seq"].unique()}
    per = []
    for s, idx in pos.items():
        sub = df.iloc[idx].reset_index(drop=True)
        for name, sc in pol.items():
            per.append({"seq": s, "policy": name,
                        "eta": eta_frame(sub, np.asarray(sc)[idx], 0.20)})
    per = pd.DataFrame(per); per.to_csv(run / "per_sequence.csv", index=False)
    w = per.pivot(index="seq", columns="policy", values="eta")
    print("\n=== per-sequence at 20% quota ===")
    cmp_rows = []
    for a, c in [("PERCEPTION-GAIN ORACLE", "decision sensitivity (learned)"),
                 ("criticality", "decision sensitivity (learned)"),
                 ("uncertainty", "decision sensitivity (learned)"),
                 ("uncertainty x criticality", "decision sensitivity (learned)"),
                 ("visual+criticality (learned)", "decision sensitivity (learned)")]:
        d = (w[c] - w[a]).dropna()
        p = stats.wilcoxon(d, alternative="greater").pvalue if len(d) >= 5 and d.abs().sum() > 0 else np.nan
        cmp_rows.append({"baseline": a, "candidate": c, "n_better": int((d > 0).sum()),
                         "n": int(len(d)), "median_delta": float(d.median()), "p": float(p)})
        print(f"  {a:32s} -> {c:32s} {int((d>0).sum()):2d}/{len(d)}  {d.median():+.3f}  p={p:.4g}")

    # ---- sweeps: does any of this depend on the planner or the cost weights? ----
    print("\n=== planner / cost sweeps (eta at 20%) ===")
    sweep = []
    for kind, key in [("planner", k) for k in P.PLANNERS] + [("cost", k) for k in P.COSTS]:
        pp = P.PLANNERS[key] if kind == "planner" else P.PLANNERS["default"]
        cp = P.COSTS[key] if kind == "cost" else P.COSTS["default"]
        if kind == "cost" and key == "default":
            continue
        d = make(pp, cp)
        ad = float((d.same_action == 0).mean())
        dE_o = eta_frame(d, d.dE, 0.20)
        dec = predict.loso(d, DEC, "gbm", "dJ", "reg", checker=NOCHK).pred
        vis = predict.loso(d, VIS, "gbm", "dJ", "reg", checker=NOCHK).pred
        r = {"kind": kind, "variant": key, "action_differs": ad,
             "corr_dE_dJ": float(stats.spearmanr(d.dE, d.dJ).correlation),
             "corr_crit_dJ": float(stats.spearmanr(d.feat_crit_sum, d.dJ).correlation),
             "eta_dE_oracle": dE_o, "eta_criticality": eta_frame(d, d.feat_crit_sum, 0.20),
             "eta_decision_learned": eta_frame(d, dec, 0.20),
             "eta_visual_learned": eta_frame(d, vis, 0.20)}
        sweep.append(r)
        print(f"  {kind:7s} {key:18s} act_diff={ad:.3f} corr(dE,dJ)={r['corr_dE_dJ']:+.3f} "
              f"eta: dE-oracle={dE_o:.3f} crit={r['eta_criticality']:.3f} "
              f"visual={r['eta_visual_learned']:.3f} decision={r['eta_decision_learned']:.3f}")
    pd.DataFrame(sweep).to_csv(run / "sweep.csv", index=False)

    (run / "summary.json").write_text(json.dumps(
        {"comparisons": cmp_rows, "J_all_cheap": float(df.J_cheap.sum()),
         "J_all_full": float(df.J_full.sum())}, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
