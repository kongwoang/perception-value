#!/usr/bin/env python
"""Phase 0D Stages 2-3: a second downstream task, and whether it wants different frames."""
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
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--det512", default=str(CACHE / "det512"))
    ap.add_argument("--tag", default="stage2")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig()
    seqs = [p.stem for p in sorted((det / "cheap_320").glob("*.npz"))]

    def make(det_dir, cheap, full, src="mono", sigma=0.0):
        d = decision.build(Path(det_dir), cheap, full, seqs, cfg, P.PlannerParams(),
                           P.CostParams(), G.PRIMARY, range_source=src, sigma=sigma)
        return decision.add_perception_gain(d, Path(det_dir), cheap, full, seqs, cfg)

    configs = {
        "KITTI 320->640": make(det, "cheap_320", "full_640"),
        "KITTI 512->640": make(args.det512, "cheap_512", "full_640"),
        "KITTI 320->640 oracle range": make(det, "cheap_320", "full_640", "oracle"),
    }
    configs["KITTI 320->640"].to_pickle(run / "base.pkl")
    configs["KITTI 512->640"].to_pickle(run / "pair512.pkl")

    # ---------- the oracle-gap table (section 12) ----------
    rows = []
    for cname, d in configs.items():
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
                "eta_E@20_speedstrat": eta(d, d["dE"], col, strata=sb),
                "eta_E@20_nonempty": eta(ne, ne["dE"], col),
                "oracle_gap@20": 1.0 - eta(d, d["dE"], col),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(run / "oracle_gap.csv", index=False)
    print("\n=== ORACLE GAP TABLE (section 12) ===")
    print(tab[["config", "task", "act_diff", "improved_same_action", "corr_dE_dJ",
               "eta_E@20", "eta_crit@20", "eta_speed@20", "oracle_gap@20"]].round(3).to_string(index=False))

    # ---------- task conditionality (sections 7 and 14) ----------
    print("\n=== TASK CONDITIONALITY ===")
    ct = []
    for cname, d in configs.items():
        jl = (d.J_cheap - d.J_full).to_numpy()
        jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
        r = {"config": cname,
             "spearman": float(stats.spearmanr(jl, jt).correlation),
             "top10_overlap": topk_overlap(jl, jt, 0.10),
             "top20_overlap": topk_overlap(jl, jt, 0.20),
             "top30_overlap": topk_overlap(jl, jt, 0.30),
             # cross-application: rank by one task, pay the other task's cost
             "eta_long_ranked_by_lat": eta(d, jt, TASKS["longitudinal"]),
             "eta_lat_ranked_by_long": eta(d, jl, TASKS["lateral"])}
        ct.append(r)
        print(f"  {cname:30s} rho={r['spearman']:+.3f} top10={r['top10_overlap']:.3f} "
              f"top20={r['top20_overlap']:.3f} | eta(long|lat-rank)={r['eta_long_ranked_by_lat']:.3f} "
              f"eta(lat|long-rank)={r['eta_lat_ranked_by_long']:.3f}")
    pd.DataFrame(ct).to_csv(run / "task_conditionality.csv", index=False)

    # ---------- trivial heuristic ceiling per task (section 15) ----------
    print("\n=== TRIVIAL-HEURISTIC CEILING, eta@20 (pooled | speed-stratified) ===")
    d = configs["KITTI 320->640"]
    sb = np.digitize(d["v_ego"], SPEED_EDGES[1:-1])
    heur = {
        "ego speed": d.v_ego.to_numpy(),
        "empty-frame indicator": d.empty_cheap.to_numpy().astype(float),
        "few cheap detections": -d.n_cheap.to_numpy().astype(float),
        "low confidence": -d.conf_mean.to_numpy(),
        "speed x empty": d.v_ego.to_numpy() * (1 + 3 * d.empty_cheap.to_numpy()),
        "criticality": d.crit_sum.to_numpy(),
        "uncertainty": d.unc_sum.to_numpy(),
        "PERCEPTION-GAIN ORACLE": d.dE.to_numpy(),
    }
    hrows = []
    for tname, col in TASKS.items():
        print(f"  -- {tname} --")
        for k, v in heur.items():
            a, b_ = eta(d, v, col), eta(d, v, col, strata=sb)
            hrows.append({"task": tname, "heuristic": k, "eta_pooled": a,
                          "eta_speed_stratified": b_})
            print(f"     {k:26s} {a:+.3f} | {b_:+.3f}")
    pd.DataFrame(hrows).to_csv(run / "heuristics.csv", index=False)

    # ---------- negative control: a task-independent cost ----------
    print("\n=== NEGATIVE CONTROL: task-independent cost ===")
    rng = np.random.default_rng(0)
    fake = d.copy()
    fake["Jlat_cheap"] = fake["J_cheap"]
    fake["Jlat_full"] = fake["J_full"]
    print(f"  if both tasks share a cost, top20 overlap = "
          f"{topk_overlap((fake.J_cheap-fake.J_full).to_numpy(), (fake.Jlat_cheap-fake.Jlat_full).to_numpy(), 0.20):.3f} "
          f"(expect 1.000)")

    (run / "summary.json").write_text(json.dumps(
        {"oracle_gap": rows, "task_conditionality": ct, "heuristics": hrows},
        indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
