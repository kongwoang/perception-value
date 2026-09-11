#!/usr/bin/env python
"""Phase 0E Stage 8: temporal replay evaluation.

Replay, not closed loop: the ego trajectory is the logged one, so an action at frame t
does not change the observation at t+1. What it adds over per-frame scoring is the cost
of a decision *history* -- hysteresis, commitment, switching, and sustained errors.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, decision, geometry as G, planner as P, runmeta, temporal as T  # noqa: E402
from rap.cache import DetCache                                                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                           # noqa: E402
from rap.paths import CACHE                                                             # noqa: E402
from rap.risk import RiskConfig                                                          # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]


def sequence_inputs(det_dir, cheap, full, seqs, cfg, adapter, pp, lp):
    """Per-sequence arrays the replay needs, for both modes and ground truth."""
    out = {}
    for s in seqs:
        geom = adapter.geometry(s)
        speeds = adapter.speeds(s)
        c = DetCache(Path(det_dir) / cheap / f"{s}.npz")
        f = DetCache(Path(det_dir) / full / f"{s}.npz")
        a = {"cheap": [], "full": [], "gt": [], "v": [], "frames": [],
             "geo_cheap": [], "geo_full": [], "geo_gt": []}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            trip = {}
            for tag, cache, idx in (("cheap", c, i), ("full", f, f.index[fr])):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.op_conf
                z, lo, hi, tt = geo["z"][k], geo["lat_min"][k], geo["lat_max"][k], geo["ttc"][k]
                trip[tag] = (z, lo, hi)
                a[tag].append(P.required_decel(z, lo, hi, tt, v, pp)[0])
                a[f"geo_{tag}"].append((z, lo, hi))
            a["gt"].append(P.required_decel(g["long_near"], g["lat_min"], g["lat_max"],
                                            g["ttc"], v, pp)[0])
            a["geo_gt"].append((g["long_near"], g["lat_min"], g["lat_max"]))
            a["v"].append(v)
            a["frames"].append(fr)
        out[s] = {k: (np.asarray(v) if k in ("cheap", "full", "gt", "v", "frames") else v)
                  for k, v in a.items()}
    return out


def replay_all(inputs, sel_by_seq, pp, cp, lp, lc, tp):
    """Run both tasks along every sequence under a given FULL-selection mask."""
    tot = {"J_long": 0.0, "J_lat": 0.0, "switches": 0.0, "lat_switches": 0.0,
           "collisions": 0.0, "lat_collisions": 0.0, "sustained": 0.0, "repeat": 0.0}
    for s, a in inputs.items():
        m = sel_by_seq[s]
        a_req = np.where(m, a["full"], a["cheap"])
        r = T.replay_longitudinal(a_req, a["gt"], pp, cp, tp)
        geo = [a["geo_full"][t] if m[t] else a["geo_cheap"][t] for t in range(len(m))]
        rl = T.replay_lateral(geo, a["v"], a["geo_gt"], lp, lc, tp)
        tot["J_long"] += r["J_seq"]; tot["switches"] += r["switches"]
        tot["collisions"] += r["collisions"]; tot["sustained"] += r["sustained_under"]
        tot["repeat"] += r["repeat_brakes"]
        tot["J_lat"] += rl["J_seq"]; tot["lat_switches"] += rl["switches"]
        tot["lat_collisions"] += rl["collisions"]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", choices=["kitti", "nuscenes"])
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--tag", default="temporal")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    lp, lc, tp = P.LateralParams(), P.LateralCostParams(), T.TemporalParams()

    if args.dataset == "kitti":
        ad = decision.KittiAdapter
    else:
        db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
        ad = make_adapter(db)
    seqs = [p.stem for p in sorted((Path(args.det) / args.cheap).glob("*.npz"))]

    print(f"building per-frame table over {len(seqs)} sequences ...")
    df = decision.build(Path(args.det), args.cheap, args.full, seqs, cfg, pp, cp,
                        G.PRIMARY, adapter=ad)
    df = decision.add_perception_gain(df, Path(args.det), args.cheap, args.full, seqs,
                                      cfg, adapter=ad)
    inputs = sequence_inputs(args.det, args.cheap, args.full, seqs, cfg, ad, pp, lp)

    scores = {
        "random": np.random.default_rng(0).random(len(df)),
        "uncertainty": df.unc_sum.to_numpy(),
        "criticality": df.crit_sum.to_numpy(),
        "ego speed": df.v_ego.to_numpy(),
        "PERCEPTION-GAIN ORACLE": df.dE.to_numpy(),
        "DECISION-VALUE ORACLE (long)": (df.J_cheap - df.J_full).to_numpy(),
        "DECISION-VALUE ORACLE (lat)": (df.Jlat_cheap - df.Jlat_full).to_numpy(),
    }
    pos = {s: np.flatnonzero(df["seq"].to_numpy() == s) for s in seqs}

    def masks(score, quota):
        sel = budget.select_pooled(np.asarray(score, float), quota)
        return {s: sel[pos[s]] for s in seqs}

    none = {s: np.zeros(len(pos[s]), bool) for s in seqs}
    allf = {s: np.ones(len(pos[s]), bool) for s in seqs}
    base = replay_all(inputs, none, pp, cp, lp, lc, tp)
    full = replay_all(inputs, allf, pp, cp, lp, lc, tp)
    print(f"\nall-CHEAP  J_long={base['J_long']:.0f} J_lat={base['J_lat']:.0f} "
          f"collisions={base['collisions']:.0f}/{base['lat_collisions']:.0f} "
          f"switches={base['switches']:.0f}")
    print(f"all-FULL   J_long={full['J_long']:.0f} J_lat={full['J_lat']:.0f} "
          f"collisions={full['collisions']:.0f}/{full['lat_collisions']:.0f} "
          f"switches={full['switches']:.0f}")

    rows = []
    for q in QUOTAS:
        orc_l = replay_all(inputs, masks(scores["DECISION-VALUE ORACLE (long)"], q),
                           pp, cp, lp, lc, tp)
        orc_t = replay_all(inputs, masks(scores["DECISION-VALUE ORACLE (lat)"], q),
                           pp, cp, lp, lc, tp)
        span_l = base["J_long"] - orc_l["J_long"]
        span_t = base["J_lat"] - orc_t["J_lat"]
        for name, sc in scores.items():
            r = replay_all(inputs, masks(sc, q), pp, cp, lp, lc, tp)
            rows.append({
                "quota": q, "policy": name,
                "J_long_seq": r["J_long"], "J_lat_seq": r["J_lat"],
                "eta_long": (base["J_long"] - r["J_long"]) / span_l if span_l > 1e-9 else np.nan,
                "eta_lat": (base["J_lat"] - r["J_lat"]) / span_t if span_t > 1e-9 else np.nan,
                "collisions": r["collisions"], "lat_collisions": r["lat_collisions"],
                "switches": r["switches"], "sustained": r["sustained"],
                "repeat_brakes": r["repeat"]})
    res = pd.DataFrame(rows)
    res.to_csv(run / "temporal.csv", index=False)
    print("\n=== TEMPORAL REPLAY: eta (longitudinal | lateral) ===")
    for q in QUOTAS:
        print(f"  quota {int(q*100)}%")
        for _, r in res[res.quota == q].iterrows():
            print(f"    {r.policy:30s} {r.eta_long:+.3f} | {r.eta_lat:+.3f}   "
                  f"collisions {r.collisions:.0f}/{r.lat_collisions:.0f}  switches {r.switches:.0f}")

    (run / "summary.json").write_text(json.dumps(
        {"all_cheap": base, "all_full": full, "rows": rows}, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()
