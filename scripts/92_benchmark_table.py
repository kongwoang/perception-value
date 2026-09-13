#!/usr/bin/env python
"""The benchmark table: every cell, every baseline, one protocol, one run.

Pre-registered in RESEARCH_LOG.md (2026-09-13 23:55), splits in configs/benchmark_splits.json.
Before this script, baseline numbers came from separate runs with different tie handling, bootstrap
counts and frame pools; this is now the only source for the benchmark table.

Protocol, as registered:
  * every eta on the TEST split; learned gates fit on train + val and scored on test once;
  * eta = gain / oracle prize at q in {10, 20, 30, 50}% of the frames, where a signal's gain is its
    expectation over uniformly random tie-breaks, computed exactly -- so there is no tie seed to
    choose, and random is the all-tied special case k/n * sum(V);
  * bootstrap over units (nuScenes scene, KITTI sequence, nuPlan log), 1000 draws, seed 0, the
    difference to random paired inside each draw; draws whose prize falls under a quarter of the
    full-sample prize are dropped and counted;
  * tie_frac and responsive_frac per row; cell descriptives including the destroyed-benefit ratio
    D = sum(max(-V,0)) / sum(max(V,0)) on test and on all units; self-agreement share S(M) for
    PKL and TIP.
Non-learned signals are also scored on all units (split = "all"), for continuity with the numbers
already reported.  A signal that cannot exist in a cell is written as a row with available=False
and its reason, so no blank in the table is ambiguous.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta                                                # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.tables import feature_columns                                          # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g84 = _load("gate84", "84_deployable_gate.py")
x83 = _load("ext83", "83_external_transfer.py")

RAW = ROOT / "results" / "raw"
NUSC_JOINED = {"oracle": RAW / "20260913_211441_phase0g_eta_fde_oracle" / "joined_frames.pkl",
               "mono": RAW / "20260913_214436_phase0g_eta_fde_mono" / "joined_frames.pkl"}
CORE = RAW / "20260913_133004_core_matrix_postreview"
PLANB = RAW / "20260912_111225_planner_b_static_fixed"
NUPLAN_SIGNALS = Path(RESULTS) / "final" / "benchmark_nuplan_signals.csv"
QUOTAS = (0.10, 0.20, 0.30, 0.50)
EPS = 1e-9
DE_COLS = ["dE", "dE_E1_fn_only", "dE_E2_fn_fp", "dE_E3_class_aware", "dE_E4_localization",
           "dE_E5_combined", "dE_E5_fp_heavy", "dE_E5_loc_heavy", "dE_E6_risk_weighted"]
SIGNALS = (["random", "uncertainty", "criticality_cheap", "criticality_gt"]
           + ["dE_exact" if c == "dE" else c for c in DE_COLS]
           + ["PKL", "TIP", "gate_ridge", "gate_gbm", "oracle"])
DEPLOYABLE = {"random": True, "uncertainty": True, "criticality_cheap": True, "gate_ridge": True,
              "gate_gbm": True, "oracle": False}
NOTE = {"brake": "rule-based braking controller (ours)",
        "traj": "Planner B, rollout planner (ours)",
        "plan_ade": "PKL's published planner vs real trajectory; FAILS viability (worse than constant velocity)",
        "plan_fde": "PKL's published planner vs real trajectory; FAILS viability (worse than constant velocity)",
        "pdm_closed": "PDM-Closed, published (tuPlan Garage)", "idm": "IDMPlanner, published (nuPlan devkit)"}


# ----------------------------------------------------------------------------------------------
# statistic

def topk_expect(s, weights, ks):
    """Expected sums of each weight vector over the top-k frames by s, ties broken uniformly.

    Frames strictly above the cut value count fully; the tie group containing the k-th frame
    contributes its total times the share of it that fits.  Also returns the share of each
    selection decided by that tie rather than by the score.
    """
    s = np.round(np.asarray(s, float), 9)
    o = np.argsort(-s, kind="stable")
    ss = s[o]
    new = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(new) - 1
    gs = np.flatnonzero(new)
    ge = np.r_[gs[1:], len(ss)]
    ks = np.asarray(ks, int)
    g = gid[np.clip(ks, 1, len(ss)) - 1]
    a, b = gs[g], ge[g]
    frac = (ks - a) / np.maximum(b - a, 1)
    out = []
    for w in weights:
        c = np.r_[0.0, np.cumsum(np.asarray(w, float)[o])]
        out.append(c[a] + (c[b] - c[a]) * frac)
    tie = (ks - a) / np.maximum(ks, 1)                    # the definition in budget.tie_fraction
    return out, tie


def quota_k(n):
    return np.array([max(int(round(q * n)), 1) for q in QUOTAS])


def evaluate(v, units, scores, nboot, rng):
    """Point estimates and paired bootstrap for every signal on one frame set."""
    v = np.asarray(v, float)
    nz = (np.abs(v) > EPS).astype(float)
    n = len(v)
    ks = quota_k(n)
    prize0 = topk_expect(v, [v], ks)[0][0]
    point = {}
    for name, s in scores.items():
        if s is None:                                      # random: every frame tied
            gain = ks / n * v.sum()
            resp, tie = np.full(len(ks), nz.mean()), np.full(len(ks), np.nan)
        else:
            (gain, rsum), tie = topk_expect(s, [v, nz], ks)
            resp = rsum / ks
        point[name] = dict(gain=gain, tie=tie, resp=resp,
                           eta=np.where(prize0 > EPS, gain / np.where(prize0 > EPS, prize0, 1), np.nan))

    uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    draws = {name: np.full((nboot, len(ks)), np.nan) for name in scores}
    dropped = np.zeros(len(ks), int)
    for b in range(nboot):
        t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vt = v[t]
        kt = quota_k(len(t))
        pz = topk_expect(vt, [vt], kt)[0][0]
        ok = pz > np.maximum(EPS, 0.25 * prize0)
        dropped += ~ok
        den = np.where(ok, pz, 1.0)
        for name, s in scores.items():
            g = kt / len(t) * vt.sum() if s is None else topk_expect(s[t], [vt], kt)[0][0]
            draws[name][b] = np.where(ok, g / den, np.nan)
    return ks, prize0, point, draws, dropped


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) >= 50 else (np.nan, np.nan)


def describe(v, cheap):
    v = np.asarray(v, float)
    pos, neg = v[v > EPS].sum(), -v[v < -EPS].sum()
    aff = np.abs(v) > EPS
    ks = quota_k(len(v))
    prize = topk_expect(v, [v], ks)[0][0]
    tot = float(np.sum(cheap))
    r = {"n_frames": len(v), "affected_share": float(aff.mean()),
         "harmed_among_affected": float((v < -EPS).sum() / max(aff.sum(), 1)),
         "destroyed_benefit_D": float(neg / pos) if pos > EPS else np.nan,
         "gross_benefit": float(pos), "gross_harm": float(neg),
         "all_full_reduction": float(v.sum() / tot) if tot > EPS else np.nan}
    for q, p in zip(QUOTAS, prize):
        r[f"oracle{int(q * 100)}_reduction"] = float(p / tot) if tot > EPS else np.nan
    return r


# ----------------------------------------------------------------------------------------------
# cells

def nuscenes_cells(splits):
    feats = g84.build_features(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", RiskConfig())
    fcols = feature_columns(feats)                         # assert_no_leakage runs here
    split_of = {u: k for k in ("train", "val", "test") for u in splits["nuscenes"][k]}
    cols = {"uncertainty": "unc_sum", "criticality_cheap": "feat_crit_sum", "criticality_gt": "crit_sum",
            "PKL": "G_PKL", "TIP": "G_TIP", **{("dE_exact" if c == "dE" else c): c for c in DE_COLS}}
    for geom, path in NUSC_JOINED.items():
        j = pd.read_pickle(path).merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one")
        assert len(j) == 3376, len(j)
        j["unit"], j["split"] = j.seq, j.seq.map(split_of)
        for system, cc, cf in (("brake", "J_cheap", "J_full"), ("plan_ade", "JC_ade_cheap", "JC_ade_full"),
                               ("plan_fde", "JC_fde_cheap", "JC_fde_full")):
            d = j[j[cc].notna()].reset_index(drop=True)
            yield dict(track="nuScenes", geometry=geom, system=system, target=cc.replace("_cheap", ""),
                       d=d, cheap=cc, full=cf, fcols=fcols, cols=cols, na={})


def kitti_cells(splits):
    feats = g84.build_features(Path(CACHE) / "det", "cheap_320", RiskConfig())
    fcols = feature_columns(feats)
    feats["seq"], feats["frame"] = feats.seq.astype(str), feats.frame.astype(int)
    split_of = {u: k for k in ("train", "val", "test") for u in splits["kitti"][k]}
    cols = {"uncertainty": "unc_sum", "criticality_cheap": "feat_crit_sum", "criticality_gt": "crit_sum",
            **{("dE_exact" if c == "dE" else c): c for c in DE_COLS}}
    na = {m: "PKL and TIP need nuScenes map rasters and PKL's planner; KITTI has neither" for m in ("PKL", "TIP")}
    for geom in ("oracle", "mono"):
        t = pd.read_pickle(CORE / f"KITTI__YOLOv8s__cheap_320tofull_640__{geom}.pkl")
        b = pd.read_pickle(PLANB / f"planB__KITTI__YOLOv8s__cheap_320tofull_640__{geom}.pkl")
        for x in (t, b):
            x["seq"], x["frame"] = x.seq.astype(str), x.frame.astype(int)
        d = (t.merge(b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
              .merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one"))
        assert len(d) == 8008, len(d)
        d["unit"], d["split"] = d.seq, d.seq.map(split_of)
        for system, cc, cf in (("brake", "J_cheap", "J_full"), ("traj", "JB_cheap", "JB_full")):
            yield dict(track="KITTI", geometry=geom, system=system, target=cc.replace("_cheap", ""),
                       d=d, cheap=cc, full=cf, fcols=fcols, cols=cols, na=na)


def nuplan_cells(splits):
    sig = pd.read_csv(NUPLAN_SIGNALS)
    roles = json.loads(NUPLAN_SIGNALS.with_suffix(".json").read_text())
    fcols = list(roles["gate_features"])
    banned = set(roles["privileged"]) | set(roles["diagnostic"])
    assert not set(fcols) & banned
    assert not [c for c in fcols if any(w in c for w in ("full", "reference", "_gt", "n_tracks"))]
    split_of = {u: k for k in ("train", "val", "test") for u in splits["nuplan"][k]}
    cols = {"uncertainty": "unc_proxy", "criticality_cheap": "crit_cheap_sum",
            "criticality_gt": "crit_sum_gt", "dE_E1_fn_only": "dE_E1_fn_only",
            "dE_E6_risk_weighted": "dE_E6_risk_weighted"}
    na = {c: "the miss model makes no false positives, class confusions or localisation errors, so "
             "this variant is undefined or identical to E1" for c in
          ["dE_E2_fn_fp", "dE_E3_class_aware", "dE_E4_localization", "dE_E5_combined",
           "dE_E5_fp_heavy", "dE_E5_loc_heavy"]}
    na["dE_exact"] = "no detection-level error table exists for simulated tracks"
    na.update({m: "PKL and TIP are defined on nuScenes rasters only" for m in ("PKL", "TIP")})
    for planner in ("pdm_closed", "idm"):
        r = pd.read_csv(Path(RESULTS) / "final" / f"phase0g_external_{planner}_raw.csv")
        assert (r[["ok_cheap", "ok_full"]] == 1).all().all()
        d = r.merge(sig, on=["scenario", "log", "iteration"], how="inner", validate="one_to_one")
        assert len(d) == 1440, len(d)
        c_cheap, c_full = x83.costs(d, "cheap"), x83.costs(d, "full")
        d["unit"], d["split"] = d.log, d.log.map(split_of)
        for target in ("safety", "scalar_J"):
            d[f"{target}_cheap"], d[f"{target}_full"] = c_cheap[target], c_full[target]
        for target in ("safety", "scalar_J"):
            yield dict(track="nuPlan", geometry="n/a", system=planner, target=target, d=d.copy(),
                       cheap=f"{target}_cheap", full=f"{target}_full", fcols=fcols, cols=cols, na=na,
                       privileged={"uncertainty"})


# ----------------------------------------------------------------------------------------------

def gate_predictions(d, fcols, v):
    fit = d.split.isin(["train", "val"]).to_numpy()
    X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                      nan=0.0, posinf=1e6, neginf=-1e6)
    out = {}
    for name, mdl in (("gate_ridge", "linear"), ("gate_gbm", "gbm")):
        m = predict.make_model(mdl, "reg", 0)
        m.fit(X[fit], v[fit])
        out[name] = m.predict(X)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--tag", default="benchmark_table")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    rng = np.random.default_rng(0)

    rows, cells, selfag = [], [], []
    for gen in (nuscenes_cells, kitti_cells, nuplan_cells):
        for c in gen(splits):
            d = c["d"]
            assert d.split.notna().all(), "a unit is missing from the frozen split"
            v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            gates = gate_predictions(d, c["fcols"], v_all)
            key = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                       system_note=NOTE[c["system"]])
            for split in ("test", "all"):
                m = (d.split == "test").to_numpy() if split == "test" else np.ones(len(d), bool)
                v, units = v_all[m], d.unit.to_numpy()[m]
                cells.append({**key, "split": split, "n_units": int(len(np.unique(units))),
                              **describe(v, d[c["cheap"]].to_numpy(float)[m])})
                scores, avail = {}, {}
                for s in SIGNALS:
                    if s == "random":
                        scores[s] = None
                    elif s == "oracle":
                        scores[s] = v
                    elif s.startswith("gate_"):
                        if split == "test":
                            scores[s] = gates[s][m]
                        else:
                            avail[s] = "learned baselines are scored on the test split only"
                    elif s in c["cols"]:
                        scores[s] = pd.to_numeric(d[c["cols"][s]], errors="coerce").fillna(-np.inf).to_numpy()[m]
                    else:
                        avail[s] = c["na"].get(s, "not defined for this track")
                ks, prize0, point, draws, dropped = evaluate(v, units, scores, args.nboot, rng)
                for s in SIGNALS:
                    base = {**key, "split": split, "signal": s,
                            "deployable": DEPLOYABLE.get(s, False),
                            "privileged": s in c.get("privileged", set()),
                            "n_units": int(len(np.unique(units))), "n_frames": int(len(v))}
                    if s not in scores:
                        rows.append({**base, "available": False, "na_reason": avail[s]})
                        continue
                    p = point[s]
                    for qi, q in enumerate(QUOTAS):
                        dr = draws[s][:, qi] - draws["random"][:, qi]
                        lo, hi = ci(draws[s][:, qi])
                        dlo, dhi = ci(dr)
                        rows.append({**base, "available": True, "quota": q, "k": int(ks[qi]),
                                     "eta": float(p["eta"][qi]), "eta_lo": lo, "eta_hi": hi,
                                     "minus_random": float(np.nanmean(dr)), "minus_random_lo": dlo,
                                     "minus_random_hi": dhi,
                                     "p_le_random": float(np.mean(dr[np.isfinite(dr)] <= 0)) if s != "random" else np.nan,
                                     "gain": float(p["gain"][qi]), "prize": float(prize0[qi]),
                                     "reduction_frac": float(p["gain"][qi] / d[c["cheap"]].to_numpy(float)[m].sum()),
                                     "tie_frac": float(p["tie"][qi]), "responsive_frac": float(p["resp"][qi]),
                                     "boot_dropped": int(dropped[qi])})
                # S(M): PKL/TIP against their own planner's path deviation vs against the real trajectory
                if c["track"] == "nuScenes" and c["system"].startswith("plan_"):
                    vs = d["_dJ_plannerC_path_dev"].to_numpy(float)[m]
                    sc = {"PKL": scores["PKL"], "TIP": scores["TIP"]}
                    _, _, ps, ds, _ = evaluate(vs, units, sc, args.nboot, rng)
                    for mname in ("PKL", "TIP"):
                        for qi, q in enumerate(QUOTAS):
                            e_self, e_truth = ps[mname]["eta"][qi], point[mname]["eta"][qi]
                            s_draw = 1.0 - draws[mname][:, qi] / ds[mname][:, qi]
                            lo, hi = ci(s_draw)
                            selfag.append({**key, "split": split, "metric": mname, "quota": q,
                                           "eta_self": float(e_self), "eta_truth": float(e_truth),
                                           "S": float(1.0 - e_truth / e_self), "S_lo": lo, "S_hi": hi,
                                           "n_frames": int(len(v))})
            r20 = [r for r in rows if r.get("quota") == 0.2 and r["split"] == "test"
                   and all(r[k] == key[k] for k in ("track", "geometry", "system", "target"))]
            txt = "  ".join(f"{r['signal']}={r['eta']:+.2f}" for r in r20
                            if r["signal"] in ("random", "uncertainty", "criticality_cheap", "gate_ridge",
                                               "gate_gbm", "dE_E6_risk_weighted", "PKL"))
            print(f"  {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:9s} test@20 {txt}", flush=True)

    out = Path(RESULTS) / "final"
    for name, df in (("benchmark_table", pd.DataFrame(rows)), ("benchmark_cells", pd.DataFrame(cells)),
                     ("benchmark_self_agreement", pd.DataFrame(selfag))):
        df.to_csv(out / f"{name}.csv", index=False)
        df.to_csv(run / f"{name}.csv", index=False)
        print(f"  wrote {out / (name + '.csv')} ({len(df)} rows)")


if __name__ == "__main__":
    main()
