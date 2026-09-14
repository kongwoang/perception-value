#!/usr/bin/env python
"""Task 1: thresholds, cells, sweep and the pre-registered reading for per-mode operating points.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 1 pre-registration").  Reads the per-mode
outcomes of 100 and 101 and refuses to run unless every equivalence check there passed.

  schemes  S0-S3 from 100 (train+val only); S4 chosen here, per mode, downstream system and
           geometry, as the 0.05-grid threshold minimising that mode's mean loss on train+val
           frames (exact ties -> the value closest to 0.25)
  cells    14; V = J_CHEAP(t_c) - J_FULL(t_f) composed from per-mode runs
  stats    per scheme x cell x split: thresholds, detections/frame, precision, recall per mode,
           affected count, harm rate P(V<0 | V!=0), rho = sum max(-V,0) / sum max(V,0), all-FULL and
           oracle@20 loss reduction, unit-bootstrap 95% CIs (1000 draws) for harm rate and rho;
           nuScenes cells also sign disagreement, Spearman correlation and harmed|gain>0 for four
           perception gains, and the brake-vs-planner opposite-sign count
  sweep    (t_c, t_f) in {0.15..0.55}^2, all units, harm rate and rho
  reading  survives / collapses / mixed, as registered
"""
from __future__ import annotations

import glob, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import percep_metrics as PM, runmeta                                   # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

_s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
t92 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t92)

EPS = 1e-9
GRID05 = [round(0.10 + 0.05 * i, 2) for i in range(13)]
SWEEP = [0.15, 0.25, 0.35, 0.45, 0.55]
NBOOT = 1000
SCHEMES = ("S0", "S1", "S2", "S3", "S4")
GAINS = {"exact_FN": {"fn": 1.0}, "FN_FP": PM.METRICS["E2_fn_fp"],
         "E5_combined": PM.METRICS["E5_combined"], "E_risk": PM.METRICS["E6_risk_weighted"]}
# name, dataset, outcome spec, system, geometry, moderate-gap KITTI cell
CELLS = [("nuScenes oracle Y8 320->640 q_brake", "nuScenes", "nusc_oracle", "brake", "oracle", False),
         ("nuScenes oracle Y8 320->640 q_plan", "nuScenes", "nusc_oracle", "plan", "oracle", False),
         ("nuScenes mono Y8 320->640 q_brake", "nuScenes", "nusc_mono", "brake", "mono", False),
         ("nuScenes mono Y8 320->640 q_plan", "nuScenes", "nusc_mono", "plan", "mono", False)]
for spec, label, moderate in (("kitti_y8_320_mono", "Y8 320->640", False), ("kitti_y8_384_mono", "Y8 384->640", True),
                              ("kitti_y8_512_mono", "Y8 512->640", True), ("kitti_rt_480_mono", "RT-DETR 480->640", True)):
    for system in ("traj", "brake"):
        CELLS.append((f"KITTI mono {label} q_{system}", "KITTI", spec, system, "mono", moderate))
for system in ("traj", "brake"):
    CELLS.append((f"KITTI oracle Y8 320->640 q_{system}", "KITTI", "kitti_y8_320_oracle", system, "oracle", False))


def latest(pattern):
    runs = sorted(glob.glob(str(ROOT / "results" / "raw" / pattern)))
    assert runs, f"no run matches {pattern}"
    return Path(runs[-1])


class Store:
    """Per-mode outcome tables of 100 (detector pipeline) and 101 (planner), loaded lazily."""

    def __init__(self, orun, prun):
        self.orun, self.prun = orun, prun
        self.oidx = pd.read_csv(orun / "outcome_index.csv")
        self.pidx = pd.read_csv(prun / "plan_index.csv")
        self.cache = {}
        self.tok = pd.read_csv(Path(CACHE) / "nusc_token_map.csv")

    def mode(self, spec, role, t):
        key = ("o", spec, role, round(t, 9))
        if key not in self.cache:
            r = self.oidx[(self.oidx.spec == spec) & np.isclose(self.oidx.t_cheap, t, atol=1e-9)
                          & np.isclose(self.oidx.t_full, t, atol=1e-9)]
            assert len(r) == 1, (spec, role, t, len(r))
            z = np.load(self.orun / "outcomes" / r.file.iloc[0], allow_pickle=False)
            d = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int)})
            for c in ("J", "Jlat", "n", "JB"):
                d[c] = z[f"{c}_{role}"]
            for c in ("fn", "fp", "loc", "cls", "crit_fn", "n_det", "n_gt"):
                d[c] = z[f"{role}_{c}"]
            self.cache[key] = d
        return self.cache[key]

    def plan(self, geometry, role, t):
        key = ("p", geometry, role, round(t, 9))
        if key not in self.cache:
            r = self.pidx[(self.pidx.variant == geometry) & (self.pidx.role == role)
                          & np.isclose(self.pidx.threshold, t, atol=1e-9)]
            assert len(r) == 1, (geometry, role, t, len(r))
            z = np.load(self.prun / r.file.iloc[0], allow_pickle=False)
            d = pd.DataFrame({"sample_token": z["sample_token"].astype(str), "scene": z["scene"].astype(str),
                              "JC": z["JC_ade"]})
            self.cache[key] = d.merge(self.tok, on="sample_token", how="left", validate="one_to_one")
        return self.cache[key]


def cell_frame(S, cell, tc, tf):
    """Frames of one cell at (t_c, t_f): V, units, and per-mode detector columns."""
    _, dataset, spec, system, geometry, _ = cell
    c, f = S.mode(spec, "cheap", tc), S.mode(spec, "full", tf)
    d = c.merge(f, on=["seq", "frame"], suffixes=("_c", "_f"), validate="one_to_one")
    if system == "plan":
        pc, pf = S.plan(geometry, "cheap", tc), S.plan(geometry, "full", tf)
        p = pc.merge(pf[["sample_token", "JC"]], on="sample_token", suffixes=("_c", "_f"), validate="one_to_one")
        d = p.merge(d, on=["seq", "frame"], how="inner", validate="one_to_one")
        d["Jc"], d["Jf"], d["unit"] = d.JC_c, d.JC_f, d.scene
    else:
        col = "JB" if system == "traj" else "J"
        d["Jc"], d["Jf"], d["unit"] = d[f"{col}_c"], d[f"{col}_f"], d.seq
    d["V"] = d.Jc - d.Jf
    return d


def harm_stats(v, units, rng, nboot=NBOOT):
    v = np.asarray(v, float)
    aff, neg = np.abs(v) > EPS, v < -EPS
    pos_m, neg_m = np.where(v > EPS, v, 0.0), np.where(neg, -v, 0.0)
    uniq, inv = np.unique(units, return_inverse=True)
    per = np.stack([np.bincount(inv, aff, len(uniq)), np.bincount(inv, neg, len(uniq)),
                    np.bincount(inv, pos_m, len(uniq)), np.bincount(inv, neg_m, len(uniq))], 1)
    tot = per.sum(0)
    harm = tot[1] / tot[0] if tot[0] > 0 else np.nan
    rho = tot[3] / tot[2] if tot[2] > EPS else np.nan
    W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                  for _ in range(nboot)]).astype(float)
    bt = W @ per
    with np.errstate(divide="ignore", invalid="ignore"):
        hb = np.where(bt[:, 0] > 0, bt[:, 1] / bt[:, 0], np.nan)
        rb = np.where(bt[:, 2] > EPS, bt[:, 3] / bt[:, 2], np.nan)
    q = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))) if np.isfinite(x).sum() >= 50 else (np.nan, np.nan)  # noqa: E731
    return {"affected": int(aff.sum()), "harmed": int(neg.sum()), "harm_rate": float(harm),
            "harm_rate_lo": q(hb)[0], "harm_rate_hi": q(hb)[1], "rho": float(rho),
            "rho_lo": q(rb)[0], "rho_hi": q(rb)[1]}


def detector_stats(d, sfx):
    nd, fp, fn, ng = (d[f"{c}_{sfx}"].to_numpy(float) for c in ("n_det", "fp", "fn", "n_gt"))
    return {"det_per_frame": float(nd.mean()), "precision": float((nd - fp).sum() / max(nd.sum(), 1)),
            "recall": float((ng - fn).sum() / max(ng.sum(), 1))}


def gain(d, weights):
    return (sum(w * d[f"{k}_c"] for k, w in weights.items()) - sum(w * d[f"{k}_f"] for k, w in weights.items())).to_numpy(float)


def s4_threshold(S, cell, role, trval):
    _, dataset, spec, system, geometry, _ = cell
    means = {}
    for t in GRID05:
        if system == "plan":
            p = S.plan(geometry, role, t)
            means[t] = float(p[p.scene.isin(trval)].JC.mean())
        else:
            m = S.mode(spec, role, t)
            means[t] = float(m[m.seq.isin(trval)]["JB" if system == "traj" else "J"].mean())
    best = min(means.values())
    ties = [t for t, m in means.items() if abs(m - best) <= 1e-12]
    return min(ties, key=lambda t: (abs(t - 0.25), t)), len(ties) > 1, means


def main():
    orun, prun = latest("*_calibration_outcomes"), latest("*_calibration_plan")
    for r in (orun, prun):
        ch = json.loads((r / "checks.json").read_text())
        assert ch["all_pass"], f"equivalence checks failed in {r.name}; nothing may be scored"
    run = runmeta.new_run("calibration_cells", {"outcomes": orun.name, "plan": prun.name})
    thr = json.loads((orun / "thresholds_S0_S3.json").read_text())
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    trval = {"nuScenes": set(splits["nuscenes"]["train"] + splits["nuscenes"]["val"]),
             "KITTI": set(splits["kitti"]["train"] + splits["kitti"]["val"])}
    test = {"nuScenes": set(splits["nuscenes"]["test"]), "KITTI": set(splits["kitti"]["test"])}
    S = Store(orun, prun)
    rng = np.random.default_rng(0)

    # thresholds per cell and scheme
    trows, chosen = [], {}
    for cell in CELLS:
        name, dataset, spec, system, geometry, _ = cell
        t4c, tie_c, mc = s4_threshold(S, cell, "cheap", trval[dataset])
        t4f, tie_f, mf = s4_threshold(S, cell, "full", trval[dataset])
        sch = {s: tuple(thr[spec][s]) for s in ("S0", "S1", "S2", "S3")}
        sch["S4"] = (t4c, t4f)
        chosen[name] = sch
        for s, (a, b) in sch.items():
            trows.append({"cell": name, "scheme": s, "t_cheap": a, "t_full": b,
                          "at_grid_floor": bool(min(a, b) <= 0.10 + 1e-9) if s in ("S2", "S4") else False,
                          "S3_target_not_reached": bool(thr[spec]["S3_none_reached_target"]) if s == "S3" else False,
                          "S4_tie": bool(tie_c or tie_f) if s == "S4" else False})
    tdf = pd.DataFrame(trows)

    # cells
    rows = []
    for cell in CELLS:
        name, dataset, spec, system, geometry, moderate = cell
        for s in SCHEMES:
            tc, tf = chosen[name][s]
            d = cell_frame(S, cell, tc, tf)
            for split in ("all", "test"):
                m = np.ones(len(d), bool) if split == "all" else d.unit.astype(str).isin(test[dataset]).to_numpy()
                x = d[m]
                v = x.V.to_numpy(float)
                r = {"cell": name, "dataset": dataset, "system": system, "geometry": geometry,
                     "moderate_gap_kitti": moderate, "scheme": s, "split": split, "t_cheap": tc, "t_full": tf,
                     "n_frames": int(len(x)), "n_units": int(x.unit.nunique())}
                for sfx, role in (("c", "cheap"), ("f", "full")):
                    for k, val in detector_stats(x, sfx).items():
                        r[f"{role}_{k}"] = val
                r.update(harm_stats(v, x.unit.to_numpy(), rng))
                tot = float(x.Jc.sum())
                k20 = max(int(round(0.2 * len(v))), 1)
                prize = float(t92.topk_expect(v, [v], [k20])[0][0][0])
                r["all_full_reduction"] = float(v.sum() / tot) if tot > EPS else np.nan
                r["oracle20_reduction"] = float(prize / tot) if tot > EPS else np.nan
                if dataset == "nuScenes":
                    for gname, w in GAINS.items():
                        g = gain(x, w)
                        both = (np.abs(g) > EPS) & (np.abs(v) > EPS)
                        gp = g > EPS
                        r[f"{gname}_sign_disagreement"] = float(np.mean(np.sign(g[both]) != np.sign(v[both]))) if both.any() else np.nan
                        r[f"{gname}_n_both_nonzero"] = int(both.sum())
                        r[f"{gname}_spearman"] = float(stats.spearmanr(g, v).correlation)
                        r[f"{gname}_harmed_given_gain_pos"] = float(np.mean(v[gp] < -EPS)) if gp.any() else np.nan
                        aff_gp = gp & (np.abs(v) > EPS)
                        r[f"{gname}_harmed_given_gain_pos_affected"] = float(np.mean(v[aff_gp] < -EPS)) if aff_gp.any() else np.nan
                rows.append(r)
            print(f"  {name:38s} {s}  t=({tc:.3f}, {tf:.3f})  harm {rows[-2]['harm_rate']:.3f}  rho {rows[-2]['rho']:.3f}  "
                  f"affected {rows[-2]['affected']}", flush=True)
    cdf = pd.DataFrame(rows)

    # brake-vs-planner opposite signs, nuScenes, per geometry and scheme
    opp = []
    for geometry in ("oracle", "mono"):
        cb = next(c for c in CELLS if c[1] == "nuScenes" and c[3] == "brake" and c[4] == geometry)
        cp = next(c for c in CELLS if c[1] == "nuScenes" and c[3] == "plan" and c[4] == geometry)
        for s in SCHEMES:
            db = cell_frame(S, cb, *chosen[cb[0]][s])[["seq", "frame", "V"]]
            dp = cell_frame(S, cp, *chosen[cp[0]][s])[["seq", "frame", "V", "scene"]]
            j = dp.merge(db, on=["seq", "frame"], suffixes=("_plan", "_brake"), validate="one_to_one")
            for split in ("all", "test"):
                x = j if split == "all" else j[j.scene.isin(test["nuScenes"])]
                both = (np.abs(x.V_plan) > EPS) & (np.abs(x.V_brake) > EPS)
                opp.append({"geometry": geometry, "scheme": s, "split": split, "n_plan_frames": len(x),
                            "both_nonzero": int(both.sum()),
                            "opposite_sign": int((both & (np.sign(x.V_plan) != np.sign(x.V_brake))).sum())})
    odf = pd.DataFrame(opp)
    for _, r in odf.iterrows():
        mask = (cdf.dataset == "nuScenes") & (cdf.geometry == r.geometry) & (cdf.scheme == r.scheme) & (cdf.split == r.split)
        cdf.loc[mask, "brake_vs_plan_both_nonzero"] = r.both_nonzero
        cdf.loc[mask, "brake_vs_plan_opposite_sign"] = r.opposite_sign

    # sweep
    srows = []
    for cell in CELLS:
        for tc in SWEEP:
            for tf in SWEEP:
                d = cell_frame(S, cell, tc, tf)
                v = d.V.to_numpy(float)
                aff, neg = np.abs(v) > EPS, v < -EPS
                pos_m, neg_m = v[v > EPS].sum(), -v[neg].sum()
                srows.append({"cell": cell[0], "t_cheap": tc, "t_full": tf, "affected": int(aff.sum()),
                              "harm_rate": float(neg.sum() / aff.sum()) if aff.any() else np.nan,
                              "rho": float(neg_m / pos_m) if pos_m > EPS else np.nan})
    sdf = pd.DataFrame(srows)

    # pre-registered reading
    reading = {}
    a = cdf[cdf.split == "all"]
    for s in ("S1", "S2", "S3"):
        x = a[a.scheme == s]
        ok = (x.harm_rate >= 0.20) & (x.rho >= 0.20)
        nus = int(ok[x.dataset == "nuScenes"].sum())
        kmod = int(ok[x.moderate_gap_kitti].sum())
        collapse = int(((x.harm_rate < 0.10) | (x.rho < 0.10)).sum())
        reading[s] = {"nuScenes_cells_passing": nus, "kitti_moderate_cells_passing": kmod,
                      "cells_below_collapse_line": collapse, "n_cells": len(x),
                      "survives": bool(nus >= 3 and kmod == 6),
                      "survives_most_kitti_reading": bool(nus >= 3 and kmod >= 4),
                      "collapses": bool(collapse > len(x) / 2)}
    verdict = ("survives" if all(r["survives"] for r in reading.values()) else
               "collapses" if all(r["collapses"] for r in reading.values()) else "mixed")
    reading["verdict"] = verdict
    print("\n  reading:", json.dumps(reading, indent=1))

    out = Path(RESULTS) / "final"
    for nm, df in (("calibration_thresholds", tdf), ("calibration_cells", cdf), ("calibration_sweep", sdf),
                   ("calibration_brake_vs_plan", odf)):
        df.to_csv(out / f"{nm}.csv", index=False)
        df.to_csv(run / f"{nm}.csv", index=False)
    (out / "calibration_reading.json").write_text(json.dumps(reading, indent=1))
    (run / "calibration_reading.json").write_text(json.dumps(reading, indent=1))
    print("  wrote", out / "calibration_cells.csv")


if __name__ == "__main__":
    main()
