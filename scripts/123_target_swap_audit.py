#!/usr/bin/env python
"""Audit of the Task 9 run (scripts/122_target_swap.py), requested after the results were read: is anything computed
wrongly?  Nothing registered is re-decided and no result file is changed.

  A1  the code path that trained the G-target models, given V as its label, reproduces the official V-target scores
      (gates and R1, every cell): inputs, split masks, hyperparameters and seeds of the G fits are the official ones
  A2  the G labels per cell: orientation, non-zero shares, rank correlation with V on the fitting units, and the nDG
      of the label itself against V on test (what a perfect G predictor would recover)
  A3  degenerate G-target scores (constant, few distinct values, ties at the 20% cut); recomputed nDG equals 122's
  A4  per-cell 95% CIs of the V-target at 20% from 122's joint bootstrap against the official per-cell CIs
  A5  122's pooled bootstrap re-run with the same seed (must reproduce its CIs), and an exploratory look at how
      leaving cells out of some draws shapes the pooled CI (not part of the registered test)

Writes results/raw/<time>_target_swap_audit/.
"""
from __future__ import annotations

import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m122 = _load("m122", "122_target_swap.py")
t92, t103, t120 = m122.t92, m122.t103, m122.t120
EPS, QUOTAS, ARCH, G_VARIANTS = m122.EPS, m122.QUOTAS, m122.ARCH, m122.G_VARIANTS
FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
QI = QUOTAS.index(0.2)
NBOOT = 1000


def cname(c):
    return f"{c['track']}|{c['geometry']}|{c['system']}|{c['target']}"


def eta_at(s, v, k, prize):
    return float(t92.topk_expect(s, [v], [k])[0][0][0] / prize) if prize > EPS else np.nan


def official_ci():
    out = {}
    for name, real in (("benchmark_table.csv", False), ("benchmark_table_routers.csv", False), ("benchmark_table_nuplan_real.csv", True)):
        x = pd.read_csv(FINAL / name)
        x["geometry"] = x.geometry.fillna("n/a")
        x = x if real else x[x.track != "nuPlan"]
        x = x[(x.split == "test") & x.signal.isin(ARCH) & np.isclose(x.quota, 0.2)]
        for r in x.itertuples():
            out[(f"{r.track}|{r.geometry}|{r.system}|{r.target}", r.signal)] = (r.eta_lo, r.eta_hi)
    return out


def main():
    swap = sorted(RAW.glob("*_target_swap"))[-1]
    run = runmeta.new_run("target_swap_audit", {"swap_run": swap.name})
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t120.register_features()
    summ = json.loads((swap / "benchmark_target_swap_summary.json").read_text())
    tab = pd.read_csv(swap / "benchmark_target_swap.csv")
    tab["geometry"] = tab.geometry.fillna("n/a")
    tab["cell"] = tab.track + "|" + tab.geometry + "|" + tab.system + "|" + tab.target

    a1, a2, a3, cells = [], [], [], []
    for gen in (m122.core_cells(splits), m122.nuplan_cells(splits, nr_fcols)):
        for c in gen:
            t0 = time.time()
            d, v = c["d"], c["v_all"]
            fit, test = d.split.isin(["train", "val"]).to_numpy(), (d.split == "test").to_numpy()
            vt = v[test]
            k20 = int(t92.quota_k(len(vt))[QI])
            p20 = float(t92.topk_expect(vt, [vt], [k20])[0][0][0])
            n_aff = int((np.abs(vt) > EPS).sum())
            defined = not (c["track"] == "nuPlan" and t120.undefined(p20, n_aff))
            name = cname(c)
            # A1: the G-target code path with V as label
            refit = {**t92.gate_predictions(d, c["fcols"], v), **t103.fit_score(c["X"], v, fit)}
            for a in ARCH:
                off, new = np.asarray(c["official"][a], float), np.asarray(refit[a], float)
                a1.append({"cell": name, "arch": a, "identical": bool(np.array_equal(off, new)),
                           "max_abs_score_diff": float(np.max(np.abs(off - new))),
                           "eta20_official": eta_at(off[test], vt, k20, p20) if defined else np.nan,
                           "eta20_refit": eta_at(new[test], vt, k20, p20) if defined else np.nan})
            # the G-target scores 122 trained and saved
            z = np.load(swap / f"gscores__{c['track']}__{c['geometry'].replace('/', '')}__{c['system']}__{c['target']}.npz",
                        allow_pickle=False)
            assert np.array_equal(z["V"], v) and np.array_equal(z["split"], d.split.to_numpy().astype("U8"))
            aff = fit & (np.abs(v) > EPS)
            for gv in G_VARIANTS:
                g = z[f"G_{gv}"]
                assert np.array_equal(g, c["G"][gv])
                a2.append({"cell": name, "g_variant": gv, "fit_mean_G": float(g[fit].mean()),
                           "fit_share_G_pos": float((g[fit] > EPS).mean()), "fit_share_G_neg": float((g[fit] < -EPS).mean()),
                           "fit_share_V_nonzero": float((np.abs(v[fit]) > EPS).mean()),
                           "spearman_G_V_fit": float(spearmanr(g[fit], v[fit]).correlation),
                           "spearman_G_V_fit_affected": float(spearmanr(g[aff], v[aff]).correlation) if aff.sum() > 2 else np.nan,
                           "label_G_itself_eta20_test": eta_at(g[test], vt, k20, p20) if defined else np.nan})
                for a in ARCH:
                    s = z[f"{gv}__{a}"][test]
                    e = eta_at(s, vt, k20, p20) if defined else np.nan
                    ref = tab[(tab.cell == name) & (tab.g_variant == gv) & (tab.arch == a) & (tab.setting == "quota")
                              & np.isclose(tab.quota, 0.2)].eta_G.iloc[0]
                    a3.append({"cell": name, "g_variant": gv, "arch": a, "constant": bool(np.ptp(s) == 0),
                               "n_distinct_test": int(len(np.unique(np.round(s, 9)))),
                               "tie_frac_20": float(t92.topk_expect(s, [vt], [k20])[1][0]),
                               "eta20_recomputed": e, "eta20_in_122": ref,
                               "matches_122": bool((np.isnan(e) and pd.isna(ref)) or abs(e - ref) < 1e-9)})
            for a in ARCH:
                s = np.asarray(c["official"][a], float)[test]
                a3.append({"cell": name, "g_variant": "V (official)", "arch": a, "constant": bool(np.ptp(s) == 0),
                           "n_distinct_test": int(len(np.unique(np.round(s, 9)))),
                           "tie_frac_20": float(t92.topk_expect(s, [vt], [k20])[1][0])})
            c.update(test=test, defined=defined, sG={a: np.asarray(z[f"primary__{a}"], float)[test] for a in ARCH})
            for k in ("X",):
                c.pop(k)
            cells.append(c)
            print(f"  {name:36s} A1 identical {sum(r['identical'] for r in a1[-len(ARCH):])}/{len(ARCH)}  [{time.time() - t0:.0f}s]",
                  flush=True)

    # A4/A5: 122's joint bootstrap, same seed and order, V-target and primary G-target at 20%
    for c in cells:
        m = c["test"]
        c["v"], c["units"] = c["v_all"][m], c["d"].unit.astype(str).to_numpy()[m]
        c["uidx"] = {u: np.flatnonzero(c["units"] == u) for u in np.unique(c["units"])}
        c["prize0"] = t92.topk_expect(c["v"], [c["v"]], t92.quota_k(len(c["v"])))[0][0]
        c["sV"] = {a: np.asarray(c["official"][a], float)[m] for a in ARCH}
        c["eV"] = {a: np.full(NBOOT, np.nan) for a in ARCH}
        c["eG"] = {a: np.full(NBOOT, np.nan) for a in ARCH}
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units = {ds: sorted(set().union(*[set(c["units"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    rng = np.random.default_rng(0)
    for b in range(NBOOT):
        pick = {ds: rng.integers(0, len(units[ds]), len(units[ds])) for ds in datasets}
        for c in cells:
            us = [units[c["dataset"]][i] for i in pick[c["dataset"]]]
            t = np.concatenate([c["uidx"][u] for u in us if u in c["uidx"]] or [np.zeros(0, int)])
            if len(t) == 0:
                continue
            vt = c["v"][t]
            kt = t92.quota_k(len(t))
            pz = t92.topk_expect(vt, [vt], kt)[0][0]
            ok = pz > np.maximum(EPS, 0.25 * c["prize0"])
            if not ok[QI]:
                continue
            for a in ARCH:
                c["eV"][a][b] = t92.topk_expect(c["sV"][a][t], [vt], [kt[QI]])[0][0][0] / pz[QI]
                c["eG"][a][b] = t92.topk_expect(c["sG"][a][t], [vt], [kt[QI]])[0][0][0] / pz[QI]

    offci = official_ci()
    a4 = []
    for c in cells:
        if not c["defined"]:
            continue
        for a in ARCH:
            lo, hi = t92.ci(c["eV"][a])
            olo, ohi = offci[(cname(c), a)]
            a4.append({"cell": cname(c), "arch": a, "joint_lo": lo, "joint_hi": hi, "official_lo": olo, "official_hi": ohi,
                       "width_ratio_joint_over_official": (hi - lo) / (ohi - olo) if (ohi - olo) > 0 else np.nan})

    pooled = [c for c in cells if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    core = [c for c in pooled if c["track"] != "nuPlan"]
    pdm = [c for c in pooled if c["track"] == "nuPlan"]
    a5 = {"reproduces_122": {}, "exploratory": {}}
    for a in ARCH:
        dr = np.array([c["eV"][a] - c["eG"][a] for c in pooled])
        with np.errstate(all="ignore"):
            pool = np.nanmean(dr, axis=0)
        lo, hi = t92.ci(pool)
        ref = summ["pooled"]["primary"]["per_architecture"][a]
        a5["reproduces_122"][a] = {"lo": lo, "hi": hi, "lo_122": ref["ci_lo"], "hi_122": ref["ci_hi"],
                                   "equal": bool(abs(lo - ref["ci_lo"]) < 1e-9 and abs(hi - ref["ci_hi"]) < 1e-9)}
        full = ~np.isnan(dr).any(axis=0)
        pdm_in = ~np.isnan(np.array([c["eV"][a] - c["eG"][a] for c in pdm])).any(axis=0)
        dcore = np.array([c["eV"][a] - c["eG"][a] for c in core])
        dpdm = np.array([c["eV"][a] - c["eG"][a] for c in pdm])
        with np.errstate(all="ignore"):
            core_mean, pdm_mean = np.nanmean(dcore, axis=0), np.nanmean(dpdm, axis=0)
        pt = lambda cs: float(np.mean([t92.topk_expect(c["sV"][a], [c["v"]], [int(t92.quota_k(len(c["v"]))[QI])])[0][0][0] / c["prize0"][QI]
                                       - t92.topk_expect(c["sG"][a], [c["v"]], [int(t92.quota_k(len(c["v"]))[QI])])[0][0][0] / c["prize0"][QI]
                                       for c in cs]))
        a5["exploratory"][a] = {
            "draws_all_12_present": int(full.sum()),
            "all_12_present_ci": t92.ci(pool[full]) if full.sum() >= 50 else (np.nan, np.nan),
            "mean_pooled_when_pdm_in": float(np.nanmean(pool[pdm_in])), "mean_pooled_when_pdm_out": float(np.nanmean(pool[~pdm_in])),
            "core10_point": pt(core), "core10_ci": t92.ci(core_mean),
            "pdm2_point": pt(pdm), "pdm2_ci": t92.ci(pdm_mean[pdm_in]) if pdm_in.sum() >= 50 else (np.nan, np.nan)}

    A1, A2, A3, A4 = (pd.DataFrame(x) for x in (a1, a2, a3, a4))
    for name, df in (("a1_refit_V_through_G_path.csv", A1), ("a2_G_labels.csv", A2), ("a3_score_degeneracy.csv", A3),
                     ("a4_ci_width.csv", A4)):
        df.to_csv(run / name, index=False)
    g3 = A3[A3.g_variant != "V (official)"]
    summary = {
        "A1_identical_scores": f"{int(A1.identical.sum())}/{len(A1)}",
        "A1_max_abs_score_diff": float(A1.max_abs_score_diff.max()),
        "A1_eta20_max_abs_diff": float(np.nanmax(np.abs(A1.eta20_official - A1.eta20_refit))),
        "A3_constant_G_scores": g3[g3.constant][["cell", "g_variant", "arch"]].to_dict("records"),
        "A3_constant_V_scores": A3[(A3.g_variant == "V (official)") & A3.constant][["cell", "arch"]].to_dict("records"),
        "A3_eta20_matches_122": f"{int(g3.matches_122.sum())}/{len(g3)}",
        "A4_median_width_ratio": float(A4.width_ratio_joint_over_official.median()),
        "A4_width_ratio_range": [float(A4.width_ratio_joint_over_official.min()), float(A4.width_ratio_joint_over_official.max())],
        "A5": a5}
    (run / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "A5"}, indent=1, default=float))
    print(json.dumps(a5, indent=1, default=float))
    print(A2[A2.g_variant == "primary"].round(3).to_string(index=False))
    print("wrote", run)


if __name__ == "__main__":
    main()
