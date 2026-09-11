#!/usr/bin/env python
"""Phase 0D final consolidation: generalization matrix, per-sequence robustness, taxonomy."""
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

SPEED_EDGES = [0, 5, 10, 15, 20, 100]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}


def eta(df, score, col, quota=0.20):
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def per_group_eta(df, col, group="seq"):
    """Oracle gap computed inside each held-out sequence/scene (criterion 7)."""
    out = {}
    for g, sub in df.groupby(group):
        if len(sub) < 40:
            continue
        sub = sub.reset_index(drop=True)
        e = eta(sub, sub["dE"], col)
        if np.isfinite(e):
            out[g] = float(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final_matrix")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    cfg = RiskConfig()
    det = Path(CACHE / "det")

    kseqs = [p.stem for p in sorted((det / "cheap_320").glob("*.npz"))]

    def kitti(cheap, full, src="mono", dd=None):
        dd = Path(dd or det)
        d = decision.build(dd, cheap, full, kseqs, cfg, P.PlannerParams(), P.CostParams(),
                           G.PRIMARY, range_source=src)
        return decision.add_perception_gain(d, dd, cheap, full, kseqs, cfg)

    db = NuScenesDB("/home/kongwoang/datasets/nuscenes", "v1.0-mini")
    ad = make_adapter(db)
    nd = Path(CACHE / "nusc_det")
    nseqs = [p.stem for p in sorted((nd / "ns_cheap_320").glob("*.npz"))]

    def nusc(src="mono"):
        d = decision.build(nd, "ns_cheap_320", "ns_full_640", nseqs, cfg, P.PlannerParams(),
                           P.CostParams(), G.PRIMARY, range_source=src, adapter=ad)
        return decision.add_perception_gain(d, nd, "ns_cheap_320", "ns_full_640", nseqs,
                                            cfg, adapter=ad)

    configs = {
        "KITTI 320->640": kitti("cheap_320", "full_640"),
        "KITTI 512->640 (moderate)": kitti("cheap_512", "full_640", dd=CACHE / "det512"),
        "KITTI oracle range": kitti("cheap_320", "full_640", "oracle"),
        "nuScenes mini 320->640": nusc(),
        "nuScenes mini oracle range": nusc("oracle"),
    }
    configs["KITTI non-empty"] = configs["KITTI 320->640"][
        configs["KITTI 320->640"].empty_cheap == 0].reset_index(drop=True)

    rows, pg = [], {}
    for cname, d in configs.items():
        for tname, col in TASKS.items():
            e = eta(d, d["dE"], col)
            per = per_group_eta(d, col)
            pg[f"{cname} | {tname}"] = per
            vals = np.array(list(per.values()))
            rows.append({
                "config": cname, "task": tname, "n": len(d),
                "eta_E@20": e, "oracle_gap": 1 - e,
                "per_group_median": float(np.median(vals)) if len(vals) else np.nan,
                "per_group_max": float(vals.max()) if len(vals) else np.nan,
                "groups_above_0.5": int((vals > 0.5).sum()) if len(vals) else 0,
                "n_groups": int(len(vals)),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(run / "matrix.csv", index=False)
    print("=== ORACLE GAP + PER-SEQUENCE ROBUSTNESS (criterion 7) ===")
    print(tab.round(3).to_string(index=False))
    (run / "per_group.json").write_text(json.dumps(pg, indent=2, default=float))

    # ---------- counterexample taxonomy (section 18) ----------
    d = configs["KITTI 320->640"]
    dJl = (d.J_cheap - d.J_full).to_numpy()
    dJt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
    dE = d.dE.to_numpy()
    tax = {
        "A: large dE, zero decision effect": float(((dE >= 2) & (np.abs(dJl) < 1e-9)).mean()),
        "B: dE <= 0 but decision improves": float(((dE <= 0) & (dJl > 1e-9)).mean()),
        "C: perception improves, decision worsens": float(((dE > 0) & (dJl < -1e-9)).mean()),
        "D: valuable longitudinally, not laterally": float(((dJl > 1e-9) & (np.abs(dJt) < 1e-9)).mean()),
        "E: valuable laterally, not longitudinally": float(((dJt > 1e-9) & (np.abs(dJl) < 1e-9)).mean()),
        "F: fast ego and decision improves": float(((d.v_ego > 8) & (dJl > 1e-9)).mean()),
        "G: fast ego but escalation wasted": float(((d.v_ego > 8) & (np.abs(dJl) < 1e-9)).mean()),
    }
    print("\n=== CASE TAXONOMY, share of all KITTI frames ===")
    for k, v in tax.items():
        print(f"  {k:46s} {v:.3f}")

    # concrete examples for the report
    ex = []
    for label, mask in [("A", (dE >= 2) & (np.abs(dJl) < 1e-9)),
                        ("B", (dE <= 0) & (dJl > 1.0)),
                        ("C", (dE > 0) & (dJl < -1.0)),
                        ("D", (dJl > 1.0) & (np.abs(dJt) < 1e-9)),
                        ("E", (dJt > 0.5) & (np.abs(dJl) < 1e-9))]:
        idx = np.flatnonzero(mask)
        if len(idx):
            r = d.iloc[idx[np.argmax(np.abs(dJl[idx]) + np.abs(dJt[idx]))]]
            ex.append({"type": label, "seq": r.seq, "frame": int(r.frame),
                       "dE": float(r.dE), "dJ_long": float(r.J_cheap - r.J_full),
                       "dJ_lat": float(r.Jlat_cheap - r.Jlat_full),
                       "v_ego": float(r.v_ego), "n_cheap": int(r.n_cheap),
                       "n_full": int(r.n_full)})
    pd.DataFrame(ex).to_csv(run / "examples.csv", index=False)
    print("\n=== CONCRETE EXAMPLES ===")
    print(pd.DataFrame(ex).round(2).to_string(index=False))

    (run / "summary.json").write_text(json.dumps(
        {"matrix": rows, "taxonomy": tax, "examples": ex}, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
