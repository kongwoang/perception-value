#!/usr/bin/env python
"""Phase 0D Stage 1: does the perception/decision mismatch survive the obvious artefacts?

Everything here runs on existing KITTI detections. Nothing is trained beyond the trivial
heuristics needed to establish a ceiling. The output is the interim table that decides
whether a lateral planner is worth writing.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, decision, geometry as G, planner as P, runmeta   # noqa: E402
from rap.paths import CACHE                                              # noqa: E402
from rap.risk import RiskConfig                                          # noqa: E402

SPEED_EDGES = [0, 5, 10, 15, 20, 100]


def select_stratified(score, strata, quota):
    """Spend the budget separately inside each stratum.

    Without this a policy can satisfy a global quota by simply picking every fast frame,
    which is exactly the confound being controlled for.
    """
    sel = np.zeros(len(score), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(quota * len(idx)))
        if k:
            sel[idx[np.argsort(-np.asarray(score)[idx], kind="stable")[:k]]] = True
    return sel


def eta(df, score, quota=0.20, strata=None):
    pick = (lambda s: select_stratified(s, strata, quota)) if strata is not None \
        else (lambda s: budget.select_pooled(np.asarray(s, float), quota))
    col = ("J_cheap", "J_full")
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def partial_spearman(x, y, z):
    """Spearman between x and y after removing the rank-linear effect of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.c_[np.ones_like(rz), rz]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(stats.pearsonr(resid(rx), resid(ry))[0])


def row(name, df, strata=None, note=""):
    dE, dJ = df["dE"].to_numpy(), df["dJ"].to_numpy()
    e_E = eta(df, dE, 0.20, strata)
    return {
        "test": name, "n": len(df),
        "corr_dE_dJ": float(stats.spearmanr(dE, dJ).correlation),
        "eta_E@20": e_E,
        "eta_crit@20": eta(df, df["crit_sum"].to_numpy(), 0.20, strata),
        "eta_unc@20": eta(df, df["unc_sum"].to_numpy(), 0.20, strata),
        "eta_random@20": float(np.mean([eta(df, np.random.default_rng(k).random(len(df)),
                                            0.20, strata) for k in range(8)])),
        "eta_speed@20": eta(df, df["v_ego"].to_numpy(), 0.20, strata),
        "oracle_gap": 1.0 - e_E,
        "act_diff": float((df["same_action"] == 0).mean()),
        "note": note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--det512", default=str(CACHE / "det512"))
    ap.add_argument("--tag", default="stage1")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    seqs = [p.stem for p in sorted((det / "cheap_320").glob("*.npz"))]

    def make(det_dir, cheap, full, src="mono", sigma=0.0):
        d = decision.build(Path(det_dir), cheap, full, seqs, cfg, pp, cp, G.PRIMARY,
                           range_source=src, sigma=sigma)
        return decision.add_perception_gain(d, Path(det_dir), cheap, full, seqs, cfg)

    sigma = decision.mono_range_sigma(det, "cheap_320", seqs, cfg)
    print(f"monocular relative range error sigma = {sigma:.3f}")

    base = make(det, "cheap_320", "full_640")
    base.to_pickle(run / "base.pkl")
    sb = np.digitize(base["v_ego"], SPEED_EDGES[1:-1])

    rows = [row("all KITTI 320->640", base)]
    rows.append(row("speed-stratified budget", base, strata=sb))

    ne = base[base.empty_cheap == 0].reset_index(drop=True)
    rows.append(row("non-empty CHEAP only", ne))
    rows.append(row("non-empty + speed-stratified", ne,
                    strata=np.digitize(ne["v_ego"], SPEED_EDGES[1:-1])))
    n2 = base[base.n_cheap >= 2].reset_index(drop=True)
    rows.append(row(">=2 CHEAP candidates", n2))
    mv = base[(base.empty_cheap == 0) & (base.v_ego > 5)].reset_index(drop=True)
    rows.append(row("non-empty AND moving >5 m/s", mv))

    orc = make(det, "cheap_320", "full_640", "oracle")
    rows.append(row("oracle range", orc))
    noisy = make(det, "cheap_320", "full_640", "noisy_gt", sigma)
    rows.append(row("calibrated-noise GT range", noisy, note=f"sigma={sigma:.3f}"))
    orc_ne = orc[orc.empty_cheap == 0].reset_index(drop=True)
    rows.append(row("oracle range + non-empty", orc_ne))

    mod = make(det, "cheap_384", "full_640")
    rows.append(row("moderate pair 384->640", mod))
    rows.append(row("384->640 non-empty", mod[mod.empty_cheap == 0].reset_index(drop=True)))
    if (Path(args.det512) / "cheap_512").exists():
        m5 = make(args.det512, "cheap_512", "full_640")
        m5.to_pickle(run / "pair512.pkl")
        rows.append(row("moderate pair 512->640", m5))
        rows.append(row("512->640 non-empty",
                        m5[m5.empty_cheap == 0].reset_index(drop=True)))
    else:
        print("cheap_512 cache not ready; skipping the 512->640 rows")

    tab = pd.DataFrame(rows)
    tab.to_csv(run / "stage1.csv", index=False)
    show = ["test", "n", "act_diff", "corr_dE_dJ", "eta_E@20", "eta_crit@20",
            "eta_unc@20", "eta_random@20", "eta_speed@20", "oracle_gap"]
    print("\n" + tab[show].round(3).to_string(index=False))

    # ---- speed-stratified detail ----
    print("\n=== within speed bins (pooled budget inside each bin) ===")
    det_rows = []
    for b in range(len(SPEED_EDGES) - 1):
        m = sb == b
        if m.sum() < 200:
            continue
        d = base[m].reset_index(drop=True)
        lo, hi = SPEED_EDGES[b], SPEED_EDGES[b + 1]
        r = row(f"{lo}-{hi} m/s", d)
        det_rows.append(r)
        print(f"  {r['test']:12s} n={r['n']:5d} act_diff={r['act_diff']:.3f} "
              f"corr(dE,dJ)={r['corr_dE_dJ']:+.3f} eta_E={r['eta_E@20']:.3f} "
              f"eta_crit={r['eta_crit@20']:.3f} eta_rand={r['eta_random@20']:.3f}")
    pd.DataFrame(det_rows).to_csv(run / "speed_bins.csv", index=False)
    print(f"\n  partial Spearman(dE, dJ | ego speed) = "
          f"{partial_spearman(base.dE, base.dJ, base.v_ego):+.3f}")

    # ---- trivial heuristic ceiling (section 15) ----
    print("\n=== trivial-heuristic ceiling, eta@20 (pooled | speed-stratified) ===")
    heur = {
        "ego speed": base.v_ego.to_numpy(),
        "empty-frame indicator": base.empty_cheap.to_numpy().astype(float),
        "n cheap detections (few first)": -base.n_cheap.to_numpy().astype(float),
        "max confidence (low first)": -base.conf_mean.to_numpy(),
        "speed x empty": base.v_ego.to_numpy() * (1 + 3 * base.empty_cheap.to_numpy()),
        "speed + empty (rank sum)": stats.rankdata(base.v_ego) + 2 * stats.rankdata(base.empty_cheap),
        "criticality": base.crit_sum.to_numpy(),
        "uncertainty": base.unc_sum.to_numpy(),
        "PERCEPTION-GAIN ORACLE": base.dE.to_numpy(),
    }
    hrows = []
    for k, v in heur.items():
        a, b_ = eta(base, v, 0.20), eta(base, v, 0.20, sb)
        hrows.append({"heuristic": k, "eta_pooled": a, "eta_speed_stratified": b_})
        print(f"  {k:32s} {a:+.3f} | {b_:+.3f}")
    pd.DataFrame(hrows).to_csv(run / "heuristics.csv", index=False)

    (run / "summary.json").write_text(json.dumps(
        {"sigma": sigma, "rows": rows, "speed_bins": det_rows, "heuristics": hrows},
        indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
