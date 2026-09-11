#!/usr/bin/env python
"""Phase 0D Stage 4: does the perception/decision mismatch replicate on nuScenes?

Characterization only, per the plan: no predictor is trained here. The question is
whether an oracle on perception gain still fails to allocate compute well for the
downstream decision, on a second dataset with better native geometry.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, decision, geometry as G, planner as P, runmeta   # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                            # noqa: E402
from rap.paths import CACHE                                              # noqa: E402
from rap.risk import RiskConfig                                          # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]
SPEED_EDGES = [0, 5, 10, 15, 20, 100]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}


def select_stratified(score, strata, quota):
    sel = np.zeros(len(score), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(quota * len(idx)))
        if k:
            sel[idx[np.argsort(-np.asarray(score)[idx], kind="stable")[:k]]] = True
    return sel


def eta(df, score, col, quota=0.20, strata=None):
    pick = (lambda s: select_stratified(s, strata, quota)) if strata is not None \
        else (lambda s: budget.select_pooled(np.asarray(s, float), quota))
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def topk_overlap(a, b, frac):
    k = int(round(frac * len(a)))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / max(k, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--det", default=str(CACHE / "nusc_det"))
    ap.add_argument("--cheap", default="ns_cheap_320")
    ap.add_argument("--full", default="ns_full_640")
    ap.add_argument("--tag", default="nusc_decision")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig()
    db = NuScenesDB(args.dataroot, args.version)
    adapter = make_adapter(db)
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    print(f"{len(seqs)} scenes from {args.version}")

    def make(src="mono", sigma=0.0):
        d = decision.build(det, args.cheap, args.full, seqs, cfg, P.PlannerParams(),
                           P.CostParams(), G.PRIMARY, range_source=src, sigma=sigma,
                           adapter=adapter)
        return decision.add_perception_gain(d, det, args.cheap, args.full, seqs, cfg,
                                            adapter=adapter)

    sigma = decision.mono_range_sigma(det, args.cheap, seqs, cfg, adapter=adapter)
    base = make()
    base.to_pickle(run / "nusc.pkl")
    orc = make("oracle")
    print(f"monocular relative range error sigma = {sigma:.3f} (KITTI: 0.352)")
    print(f"frames {len(base)}  ego speed mean {base.v_ego.mean():.1f} m/s  "
          f"empty CHEAP {float(base.empty_cheap.mean()):.3f}")

    rows = []
    for cname, d in (("nuScenes mini", base), ("nuScenes mini oracle range", orc)):
        sb = np.digitize(d["v_ego"], SPEED_EDGES[1:-1])
        ne = d[d.empty_cheap == 0].reset_index(drop=True)
        for tname, col in TASKS.items():
            dj = (d[col[0]] - d[col[1]]).to_numpy()
            act = "same_action" if tname == "longitudinal" else "lat_same_action"
            improved = d["dE"] > 0
            rows.append({
                "config": cname, "task": tname, "n": len(d),
                "act_diff": float((d[act] == 0).mean()),
                "improved_same_action": float((improved & (d[act] == 1)).sum() / max(improved.sum(), 1)),
                "corr_dE_dJ": float(stats.spearmanr(d["dE"], dj).correlation),
                "corr_crit_dJ": float(stats.spearmanr(d["crit_sum"], dj).correlation),
                "eta_E@20": eta(d, d["dE"], col),
                "eta_crit@20": eta(d, d["crit_sum"], col),
                "eta_unc@20": eta(d, d["unc_sum"], col),
                "eta_speed@20": eta(d, d["v_ego"], col),
                "eta_E@20_nonempty": eta(ne, ne["dE"], col) if len(ne) > 50 else np.nan,
                "eta_E@20_speedstrat": eta(d, d["dE"], col, strata=sb),
                "oracle_gap@20": 1.0 - eta(d, d["dE"], col),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(run / "nusc_oracle_gap.csv", index=False)
    print("\n=== nuScenes ORACLE GAP ===")
    print(tab[["config", "task", "act_diff", "improved_same_action", "corr_dE_dJ",
               "eta_E@20", "eta_crit@20", "eta_speed@20", "oracle_gap@20"]].round(3).to_string(index=False))

    print("\n=== task conditionality on nuScenes ===")
    ct = []
    for cname, d in (("nuScenes mini", base), ("nuScenes mini oracle range", orc)):
        jl = (d.J_cheap - d.J_full).to_numpy()
        jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
        r = {"config": cname, "spearman": float(stats.spearmanr(jl, jt).correlation),
             "top10_overlap": topk_overlap(jl, jt, 0.10),
             "top20_overlap": topk_overlap(jl, jt, 0.20)}
        ct.append(r)
        print(f"  {cname:28s} rho={r['spearman']:+.3f} top10={r['top10_overlap']:.3f} "
              f"top20={r['top20_overlap']:.3f}")

    print("\n=== trivial heuristics on nuScenes (longitudinal), eta@20 ===")
    sb = np.digitize(base["v_ego"], SPEED_EDGES[1:-1])
    heur = {"ego speed": base.v_ego.to_numpy(),
            "empty-frame": base.empty_cheap.to_numpy().astype(float),
            "few detections": -base.n_cheap.to_numpy().astype(float),
            "criticality": base.crit_sum.to_numpy(),
            "uncertainty": base.unc_sum.to_numpy(),
            "PERCEPTION-GAIN ORACLE": base.dE.to_numpy()}
    hrows = []
    for k, v in heur.items():
        a, b_ = eta(base, v, TASKS["longitudinal"]), eta(base, v, TASKS["longitudinal"], strata=sb)
        hrows.append({"heuristic": k, "eta_pooled": a, "eta_speed_stratified": b_})
        print(f"  {k:26s} {a:+.3f} | {b_:+.3f}")

    (run / "summary.json").write_text(json.dumps(
        {"sigma": sigma, "n_scenes": len(seqs), "rows": rows,
         "task_conditionality": ct, "heuristics": hrows}, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
