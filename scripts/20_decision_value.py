#!/usr/bin/env python
"""Phase 0C kill test: does extra perception compute change the downstream DECISION?

Runs the deterministic braking controller over the existing KITTI CHEAP/FULL cache and
produces the diagnostic table the plan asks for before any modelling. If decision gain
turns out to be a monotone function of criticality or of perception gain -- or if the two
modes almost never disagree on the action -- the direction closes here.
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
from rap import budget, geometry as G, kitti, planner as P, runmeta   # noqa: E402
from rap.cache import DetCache                                       # noqa: E402
from rap.paths import CACHE, PROCESSED                               # noqa: E402
from rap.risk import RiskConfig                                      # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]


def run_sequence(seq, cheap: DetCache, full: DetCache, geom, crit,
                 pp: P.PlannerParams, cp: P.CostParams, cfg: RiskConfig,
                 shuffle_rng=None) -> pd.DataFrame:
    speeds = kitti.load_oxts(seq)[:, 8]
    rows = []
    prev = {"cheap": None, "full": None, "gt": None}
    for i, frame in enumerate(cheap.frames):
        frame = int(frame)
        j = full.index[frame]
        v_ego = float(speeds[frame]) if frame < len(speeds) else float(speeds[-1])

        sel = geom["frame"] == frame
        g, c_gt = geom[sel], crit[sel]
        ok = (g["y2"] - g["y1"]) >= cfg.min_gt_height
        g, c_gt = g[ok], c_gt[ok]

        dc, df = cheap.det(i), full.det(j)
        gc, gf = cheap.geo(i), full.geo(j)
        kc, kf = dc["conf"] >= cfg.op_conf, df["conf"] >= cfg.op_conf

        # identical call for all three geometries -- only the inputs differ
        a_c, _ = P.required_decel(gc["z"][kc], gc["lat_min"][kc], gc["lat_max"][kc],
                                  gc["ttc"][kc], v_ego, pp)
        a_f, _ = P.required_decel(gf["z"][kf], gf["lat_min"][kf], gf["lat_max"][kf],
                                  gf["ttc"][kf], v_ego, pp)
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"],
                                   g["ttc"], v_ego, pp)

        act_c, act_f = P.discrete_action(a_c, pp), P.discrete_action(a_f, pp)
        act_gt = P.discrete_action(a_gt, pp)
        Jc = P.decision_cost(act_c, a_gt, prev["cheap"], pp, cp)
        Jf = P.decision_cost(act_f, a_gt, prev["full"], pp, cp)
        prev = {"cheap": act_c, "full": act_f, "gt": act_gt}

        rows.append({
            "seq": seq, "frame": frame, "v_ego": v_ego,
            "a_req_cheap": a_c, "a_req_full": a_f, "a_req_gt": a_gt,
            "act_cheap": act_c, "act_full": act_f, "act_gt": act_gt,
            "J_cheap": Jc["J"], "J_full": Jf["J"],
            "V_dec": Jc["J"] - Jf["J"],
            "shortfall_cheap": Jc["shortfall"], "shortfall_full": Jf["shortfall"],
            "excess_cheap": Jc["excess"], "excess_full": Jf["excess"],
            "collision_cheap": Jc["collision"], "collision_full": Jf["collision"],
            "same_action": int(act_c == act_f),
            # deployable decision-sensitivity signals, from the CHEAP estimate only
            "feat_dec_margin": P.decision_margin(a_c, pp),
            "feat_bdist": P.boundary_distance(a_c, pp),
            "feat_a_req_cheap": a_c,
            "feat_v_ego": v_ego,
            "feat_n_corridor": float(P.in_corridor(gc["lat_min"][kc], gc["lat_max"][kc],
                                                   pp.corridor_half_w).sum()),
            "feat_min_z_corridor": float(np.min(
                gc["z"][kc][P.in_corridor(gc["lat_min"][kc], gc["lat_max"][kc], pp.corridor_half_w)],
                initial=200.0)),
        })
    return pd.DataFrame(rows)


def build(det: Path, cheap_m, full_m, seqs, pp, cp, cfg, crit_model) -> pd.DataFrame:
    out = []
    for s in seqs:
        geom = G.sequence_geometry(s)
        out.append(run_sequence(s, DetCache(det / cheap_m / f"{s}.npz"),
                                DetCache(det / full_m / f"{s}.npz"), geom,
                                G.criticality_for(geom, crit_model), pp, cp, cfg))
    return pd.concat(out, ignore_index=True)


def eta_for(df, score, quota, cost_cols=("J_cheap", "J_full")):
    sel = budget.select_pooled(np.asarray(score, float), quota)
    tot = budget.total_risk(df, sel, cost_cols)
    all_cheap = float(df[cost_cols[0]].sum())
    oracle_sel = budget.select_pooled((df[cost_cols[0]] - df[cost_cols[1]]).to_numpy(), quota)
    orc = budget.total_risk(df, oracle_sel, cost_cols)
    span = all_cheap - orc
    return (all_cheap - tot) / span if span > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--frames", default=str(PROCESSED / "frames_composite.pkl"))
    ap.add_argument("--planner", default="default")
    ap.add_argument("--cost", default="default")
    ap.add_argument("--op_conf", type=float, default=0.25)
    ap.add_argument("--tag", default="decision")
    args = ap.parse_args()

    run = runmeta.new_run(args.tag, vars(args))
    det = Path(args.det)
    cfg = RiskConfig(op_conf=args.op_conf)
    pp, cp = P.PLANNERS[args.planner], P.COSTS[args.cost]
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]

    df = build(det, args.cheap, args.full, seqs, pp, cp, cfg, G.PRIMARY)
    frames = pd.read_pickle(args.frames).reset_index(drop=True)
    key = ["seq", "frame"]
    df = df.merge(frames[key + ["value_task", "value_visual", "err_std_cheap",
                                "err_std_full", "feat_binent_sum", "feat_crit_sum",
                                "feat_conf_mean", "feat_n_det"]],
                  on=key, how="left", validate="one_to_one")
    df["dE"] = df["err_std_cheap"] - df["err_std_full"]        # perception gain
    df["dJ"] = df["V_dec"]                                      # decision gain
    df.to_pickle(run / "decision.pkl")

    n = len(df)
    perception_differs = (df["dE"].abs() > 1e-9) | (df["a_req_cheap"] - df["a_req_full"]).abs().gt(0.05)
    act_diff = df["same_action"] == 0
    improved = df["dE"] > 0

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(stats.spearmanr(a[m], b[m]).correlation)

    U, C, dE, dJ = (df["feat_binent_sum"].to_numpy(), df["feat_crit_sum"].to_numpy(),
                    df["dE"].to_numpy(), df["dJ"].to_numpy())
    tab = {
        "% frames cheap/full perception differ": 100 * float(perception_differs.mean()),
        "% frames cheap/full ACTION differ": 100 * float(act_diff.mean()),
        "% frames detection improved but SAME action": 100 * float((improved & ~act_diff).mean()),
        "% of detection-improved frames with same action":
            100 * float((improved & ~act_diff).sum() / max(improved.sum(), 1)),
        "% frames action change is beneficial (dJ>0)": 100 * float((act_diff & (dJ > 1e-9)).mean()),
        "% frames action change is harmful (dJ<0)": 100 * float((act_diff & (dJ < -1e-9)).mean()),
        "% frames dJ != 0": 100 * float((np.abs(dJ) > 1e-9).mean()),
        "corr(uncertainty, dJ)": corr(U, dJ),
        "corr(criticality, dJ)": corr(C, dJ),
        "corr(perception gain dE, dJ)": corr(dE, dJ),
        "corr(dE, dJ) | action differs": corr(dE[act_diff.to_numpy()], dJ[act_diff.to_numpy()]),
    }
    for q in QUOTAS:
        tab[f"eta uncertainty @{int(q*100)}"] = eta_for(df, U, q)
        tab[f"eta criticality @{int(q*100)}"] = eta_for(df, C, q)
        tab[f"eta dE-ORACLE @{int(q*100)}"] = eta_for(df, dE, q)
        tab[f"eta dJ-ORACLE @{int(q*100)}"] = eta_for(df, dJ, q)

    print(f"\n{'METRIC':52s} VALUE")
    print("-" * 68)
    for k, v in tab.items():
        print(f"{k:52s} {v:8.3f}")

    # action confusion, and where the decisions land at all
    print("\naction distribution (rows: CHEAP, cols: FULL)")
    print(pd.crosstab(df["act_cheap"].map(dict(enumerate(P.ACTION_NAMES))),
                      df["act_full"].map(dict(enumerate(P.ACTION_NAMES)))).to_string())
    print("\nGT action distribution:",
          df["act_gt"].map(dict(enumerate(P.ACTION_NAMES))).value_counts().to_dict())
    print(f"moving frames (v_ego > 1 m/s): {100*float((df.v_ego>1).mean()):.1f}%")

    (run / "diagnostic.json").write_text(json.dumps(tab, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
