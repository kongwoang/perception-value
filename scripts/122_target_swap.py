#!/usr/bin/env python
"""Task 9: same architecture, different target -- perception gain G vs decision value V.

Pre-registered in RESEARCH_LOG.md (Task 9 pre-registration), committed before any G-target model was trained.
Every learned allocator of the benchmark (gate_ridge, gate_gbm, R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf) is
scored against V twice: with its official V-target scores, and retrained by the same code on a perception-gain label
G.  Nothing official is written:

  results/final/benchmark_target_swap.csv           G variant x architecture x cell x (quota | 20% ms budget)
  results/final/benchmark_target_swap_summary.json  sanity, train-split sign agreement, pooled test, reading

Stages: (1) build cells and official V scores, check S1-S3 and stop on any failure; (2) fit G-target models;
(3) one joint bootstrap (units resampled once per dataset per draw) for per-cell, paired and pooled statistics.
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


t120 = _load("t120", "120_nuplan_real_allocation.py")
t92, t93, t103 = t120.t92, t120.t93, t120.t103

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS, QUOTAS = t92.EPS, t92.QUOTAS
GATES = ["gate_ridge", "gate_gbm"]
R1 = ["R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
ARCH = GATES + R1
BUDGET_ARCH = ARCH + ["gate_gbm_batched"]                  # same scores as gate_gbm, batched inference overhead
G_VARIANTS = ["primary", "dE_exact", "dE_E6_risk_weighted"]
CORE_G = {"primary": "dE_E5_combined", "dE_exact": "dE", "dE_E6_risk_weighted": "dE_E6_risk_weighted"}
R1_RUN = RAW / "20260915_014442_routers_r1"
NR_RUN = RAW / "20260915_033311_nuplan_real_allocation"
BUDGET_UNIT, BUDGET_LEVEL, POOLED_QUOTA = "ms", 0.2, 0.2


# ------------------------------------------------------------------------------------------------ cells

def core_cells(splits):
    mats = {"nuScenes": t103.frame_matrix(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "nuScenes"),
            "KITTI": t103.frame_matrix(Path(CACHE) / "det", "cheap_320", "KITTI")}
    cost = {"nuScenes": t93.profile_costs("nuScenes"), "KITTI": t93.profile_costs("KITTI")}
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            keys, M = mats[c["track"]]
            j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(
                keys.assign(_row=np.arange(len(keys))), on=["seq", "frame"], how="left", validate="one_to_one")
            assert j._row.notna().all()
            X = M[j._row.to_numpy(int)]
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            z = np.load(R1_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz", allow_pickle=False)
            assert (z["seq"].astype(str) == d.seq.astype(str).to_numpy()).all() and (z["frame"] == d.frame.to_numpy()).all(), "S2"
            assert np.array_equal(z["V"], v), "S2: official V differs"
            official = {**t92.gate_predictions(d, c["fcols"], v), **{r: z[r] for r in R1}}
            G = {g: d[col].to_numpy(float) for g, col in CORE_G.items()}
            assert all(np.isfinite(x).all() for x in G.values())
            assert not set(CORE_G.values()) & set(c["fcols"]), "S3: a G column is a gate feature"
            yield dict(dataset=c["track"], track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                       d=d, v_all=v, X=X, fcols=c["fcols"], cheap=c["cheap"], official=official, G=G,
                       cost={u: (cost[c["track"]]["cheap"][u], cost[c["track"]]["640"][u]) for u in ("ms", "mJ")})


def nuplan_cells(splits, fcols):
    spec = pickle.load(open(Path(CACHE) / "nuplan_real" / "branches.pkl", "rb"))["spec"]
    det = json.loads((RAW / "nuplan_task5" / "detect_summary.json").read_text())
    cost = {"ms": (det["ns_cheap_320"]["ms_total_median"], det["ns_full_640"]["ms_total_median"]),
            "mJ": (det["energy"]["ns_cheap_320"]["cpu_gpu_mj_per_frame"], det["energy"]["ns_full_640"]["cpu_gpu_mj_per_frame"])}
    for c in t120.cells(splits):
        d = c["d"].reset_index(drop=True)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(NR_RUN / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        assert (z["scenario"].astype(str) == d.scenario.astype(str).to_numpy()).all()
        assert (z["iteration"] == d.iteration.to_numpy()).all(), "S2"
        assert np.array_equal(z["V"], v), "S2: official V differs"
        err = {}
        for mode in ("cheap", "full"):
            sp = [spec[(s, int(i), mode)] for s, i in zip(d.scenario, d.iteration)]
            miss, fp = np.array([len(x[0]) for x in sp], float), np.array([len(x[1]) for x in sp], float)
            assert np.array_equal(fp, (d[f"n_tracks_{mode}"] - d[f"n_tracks_{mode}_nofp"]).to_numpy(float))
            err[mode] = (miss, fp)
        G = {"primary": (err["cheap"][0] + err["cheap"][1]) - (err["full"][0] + err["full"][1]),
             "dE_exact": err["cheap"][0] - err["full"][0],
             "dE_E6_risk_weighted": d["dE_E6_risk_weighted"].to_numpy(float)}
        assert np.allclose(G["dE_exact"], d["dE_E1_fn_only"].to_numpy(float)), "missed-track change differs from 119"
        assert not {"dE_E1_fn_only", "dE_E6_risk_weighted"} & set(fcols), "S3"
        yield dict(dataset="nuPlan", track="nuPlan", geometry="n/a", system=c["system"], target=c["target"], d=d, v_all=v,
                   X=c["X"], fcols=fcols, cheap=c["cheap"], official={a: z[a] for a in ARCH}, G=G, cost=cost)


# ------------------------------------------------------------------------------------------------ helpers

def gain_at(sc, v, frac):
    k = int(np.floor(frac * len(v) + 1e-9))
    if k <= 0:
        return 0.0
    if sc is None:
        return k / len(v) * v.sum()
    return float(t92.topk_expect(sc, [v], [k])[0][0][0])


def overhead(ov, track, arch):
    ms, mj, _ = t93.signal_overhead(ov, track, arch)
    return {"ms": ms, "mJ": mj}[BUDGET_UNIT]


def official_tables():
    def rd(name, real_nuplan):
        x = pd.read_csv(FINAL / name)
        x["geometry"] = x.geometry.fillna("n/a")               # read_csv turns nuPlan's "n/a" into NaN
        # the detection-profile tables also hold nuPlan rows under the same keys: take nuPlan only from the real track
        return x if real_nuplan else x[x.track != "nuPlan"]
    tab = pd.concat([rd("benchmark_table.csv", False), rd("benchmark_table_routers.csv", False),
                     rd("benchmark_table_nuplan_real.csv", True)], ignore_index=True)
    tab = tab[(tab.split == "test") & tab.signal.isin(ARCH) & tab.quota.notna()]
    bud = pd.concat([rd("benchmark_budget_routers.csv", False), rd("benchmark_budget_nuplan_real.csv", True)], ignore_index=True)
    bud = bud[(bud.unit == BUDGET_UNIT) & np.isclose(bud.budget_level, BUDGET_LEVEL) & bud.signal.isin(BUDGET_ARCH)]
    key = ["track", "geometry", "system", "target", "signal"]
    return ({tuple(r[k] for k in key) + (round(r.quota, 2),): r.eta for _, r in tab.iterrows()},
            {tuple(r[k] for k in key): r.eta for _, r in bud.iterrows()})


def agree(g, v, m):
    both = (np.abs(g) > EPS) & (np.abs(v) > EPS) & m
    n = int(both.sum())
    return (float(np.mean(np.sign(g[both]) == np.sign(v[both]))) if n else np.nan), n


def ci(x):
    return t92.ci(np.asarray(x, float))


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()
    for tag, run_dir in (("routers_r1", R1_RUN), ("nuplan_real_allocation", NR_RUN)):
        assert sorted(RAW.glob(f"*_{tag}"))[-1] == run_dir, f"{run_dir.name} is not the latest official {tag} run"
    run = runmeta.new_run("target_swap", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t120.register_features()
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    off_q, off_b = official_tables()

    # ---- stage 1: cells, official V scores, S1
    cells, sanity = [], []
    for gen in (core_cells(splits), nuplan_cells(splits, nr_fcols)):
        for c in gen:
            d = c["d"]
            m = (d.split == "test").to_numpy()
            c["fit"] = d.split.isin(["train", "val"]).to_numpy()
            c["v"], c["units"] = c["v_all"][m], d.unit.astype(str).to_numpy()[m]
            c["test"] = m
            c["n_aff"] = int((np.abs(c["v"]) > EPS).sum())
            ks = t92.quota_k(len(c["v"]))
            c["ks"], c["prize0"] = ks, t92.topk_expect(c["v"], [c["v"]], ks)[0][0]
            tot = float(d[c["cheap"]].to_numpy(float)[m].sum())
            c["oracle_red"] = c["prize0"] / tot if tot > EPS else np.full(len(ks), np.nan)
            c["undef"] = ([t120.undefined(p, c["n_aff"]) for p in c["prize0"]] if c["track"] == "nuPlan" else [""] * len(ks))
            c0, c1 = c["cost"][BUDGET_UNIT]
            c["prize_b"] = gain_at(c["v"], c["v"], BUDGET_LEVEL)
            c["tot"] = tot
            c["undef_b"] = t120.undefined(c["prize_b"], c["n_aff"]) if c["track"] == "nuPlan" else ""
            c["frac"] = {a: max((c0 + BUDGET_LEVEL * c1 - c0 - overhead(ov, c["track"], a)) / c1, 0.0) for a in BUDGET_ARCH}
            key = (c["track"], c["geometry"], c["system"], c["target"])
            for a in ARCH:
                gain = t92.topk_expect(c["official"][a][m], [c["v"]], ks)[0][0]
                for qi, q in enumerate(QUOTAS):
                    eta = np.nan if c["undef"][qi] else (gain[qi] / c["prize0"][qi] if c["prize0"][qi] > EPS else np.nan)
                    ref = off_q.get(key + (a, round(q, 2)), "missing")
                    ok = ref != "missing" and ((np.isnan(eta) and pd.isna(ref)) or
                                               (not np.isnan(eta) and not pd.isna(ref) and round(eta, 3) == round(ref, 3)))
                    sanity.append(dict(zip(("track", "geometry", "system", "target"), key), setting="quota", quota=q,
                                       arch=a, eta=eta, official=None if ref == "missing" else ref, ok=bool(ok)))
            for a in BUDGET_ARCH:
                sc = c["official"]["gate_gbm" if a == "gate_gbm_batched" else a][m]
                g = gain_at(sc, c["v"], c["frac"][a])
                eta = np.nan if c["undef_b"] else (g / c["prize_b"] if c["prize_b"] > EPS else np.nan)
                ref = off_b.get(key + (a,), "missing")
                ok = ref != "missing" and ((np.isnan(eta) and pd.isna(ref)) or
                                           (not np.isnan(eta) and not pd.isna(ref) and round(eta, 3) == round(ref, 3)))
                sanity.append(dict(zip(("track", "geometry", "system", "target"), key), setting="budget_ms_20", quota=BUDGET_LEVEL,
                                   arch=a, eta=eta, official=None if ref == "missing" else ref, ok=bool(ok)))
            cells.append(c)
            print(f"  cell {len(cells):2d} {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:8s} "
                  f"test frames {m.sum()}  affected {c['n_aff']}", flush=True)
    san = pd.DataFrame(sanity)
    san.to_csv(run / "sanity_S1.csv", index=False)
    bad = san[~san.ok]
    print(f"  S1: {len(san) - len(bad)}/{len(san)} official nDG values reproduced to 3 decimals", flush=True)
    summary = {"sanity_S1": {"n": int(len(san)), "reproduced": int(len(san) - len(bad)),
                             "max_abs_diff": float(np.nanmax(np.abs(san.eta.astype(float) - san.official.astype(float)))),
                             "failures": bad.to_dict("records")},
               "sanity_S2_S3": "asserted (official V, keys and frame order equal; no G column among inputs)"}
    if len(bad):
        (run / "summary_S1_failed.json").write_text(json.dumps(summary, indent=1, default=float))
        print(bad.to_string())
        print("S1 FAILED: the official models are not reproduced; stopping before any G-target model is trained")
        sys.exit(3)

    # ---- stage 2: G-target models
    for c in cells:
        t0 = time.time()
        c["gscores"] = {}
        for gv in G_VARIANTS:
            g = c["G"][gv]
            c["gscores"][gv] = {**t92.gate_predictions(c["d"], c["fcols"], g), **t103.fit_score(c["X"], g, c["fit"])}
        np.savez_compressed(run / f"gscores__{c['track']}__{c['geometry'].replace('/', '')}__{c['system']}__{c['target']}.npz",
                            unit=c["d"].unit.astype(str).to_numpy().astype("U40"), split=c["d"].split.to_numpy().astype("U8"),
                            V=c["v_all"], **{f"G_{gv}": c["G"][gv] for gv in G_VARIANTS},
                            **{f"{gv}__{a}": s for gv in G_VARIANTS for a, s in c["gscores"][gv].items()})
        print(f"  G-target fits {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:8s} [{time.time() - t0:.0f}s]", flush=True)

    # ---- stage 3: joint bootstrap
    names = [("V", a) for a in ARCH] + [(gv, a) for gv in G_VARIANTS for a in ARCH]
    bnames = [("V", a) for a in BUDGET_ARCH] + [("primary", a) for a in BUDGET_ARCH]
    for c in cells:
        m = c["test"]
        c["s"] = {("V", a): c["official"][a][m] for a in ARCH}
        c["s"].update({(gv, a): c["gscores"][gv][a][m] for gv in G_VARIANTS for a in ARCH})
        c["sb"] = {(t, a): c["s"][(t, "gate_gbm" if a == "gate_gbm_batched" else a)] for t, a in bnames}
        c["uidx"] = {u: np.flatnonzero(c["units"] == u) for u in np.unique(c["units"])}
        c["dq"] = {nm: np.full((args.nboot, len(QUOTAS)), np.nan) for nm in names}
        c["dq_rand"] = np.full((args.nboot, len(QUOTAS)), np.nan)
        c["db"] = {nm: np.full(args.nboot, np.nan) for nm in bnames}
        c["db_rand"] = np.full(args.nboot, np.nan)
        c["drop_q"], c["drop_b"] = np.zeros(len(QUOTAS), int), 0
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units = {ds: sorted(set().union(*[set(c["units"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for b in range(args.nboot):
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
            c["drop_q"] += ~ok
            den = np.where(ok, pz, 1.0)
            c["dq_rand"][b] = np.where(ok, kt / len(t) * vt.sum() / den, np.nan)
            for nm in names:
                c["dq"][nm][b] = np.where(ok, t92.topk_expect(c["s"][nm][t], [vt], kt)[0][0] / den, np.nan)
            pb = gain_at(vt, vt, BUDGET_LEVEL)
            if pb > max(EPS, 0.25 * c["prize_b"]):
                c["db_rand"][b] = gain_at(None, vt, BUDGET_LEVEL) / pb
                for nm in bnames:
                    c["db"][nm][b] = gain_at(c["sb"][nm][t], vt, c["frac"][nm[1]]) / pb
            else:
                c["drop_b"] += 1
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{args.nboot} [{time.time() - t0:.0f}s]", flush=True)

    # ---- rows
    rows = []
    for c in cells:
        key = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                   n_units=int(len(c["uidx"])), n_frames=int(len(c["v"])), n_affected=c["n_aff"])
        m = c["test"]
        for gv in G_VARIANTS:
            ag_tr = agree(c["G"][gv], c["v_all"], (c["d"].split == "train").to_numpy())
            ag_fit = agree(c["G"][gv], c["v_all"], c["fit"])
            for a in ARCH:
                sV, sG = c["s"][("V", a)], c["s"][(gv, a)]
                gV = t92.topk_expect(sV, [c["v"]], c["ks"])[0][0]
                gG = t92.topk_expect(sG, [c["v"]], c["ks"])[0][0]
                for qi, q in enumerate(QUOTAS):
                    why = c["undef"][qi]
                    p0 = c["prize0"][qi]
                    eV = np.nan if (why or p0 <= EPS) else gV[qi] / p0
                    eG = np.nan if (why or p0 <= EPS) else gG[qi] / p0
                    xV, xG, xr = c["dq"][("V", a)][:, qi], c["dq"][(gv, a)][:, qi], c["dq_rand"][:, qi]
                    rows.append({**key, "g_variant": gv, "arch": a, "setting": "quota", "quota": q, "ndg_defined": not why,
                                 "undefined_reason": why, **_stats(eV, eG, xV, xG, xr, why),
                                 "oracle_reduction": float(c["oracle_red"][qi]),
                                 "abs_reduction_V": eV * c["oracle_red"][qi], "abs_reduction_G": eG * c["oracle_red"][qi],
                                 "boot_dropped": int(c["drop_q"][qi]), "escalated_frac": np.nan,
                                 "sign_agree_train": ag_tr[0], "sign_agree_train_n": ag_tr[1],
                                 "sign_agree_fit": ag_fit[0], "sign_agree_fit_n": ag_fit[1]})
            if gv != "primary":
                continue
            why = c["undef_b"]
            for a in BUDGET_ARCH:
                pb = c["prize_b"]
                eV = np.nan if (why or pb <= EPS) else gain_at(c["sb"][("V", a)], c["v"], c["frac"][a]) / pb
                eG = np.nan if (why or pb <= EPS) else gain_at(c["sb"][("primary", a)], c["v"], c["frac"][a]) / pb
                red = pb / c["tot"] if c["tot"] > EPS else np.nan
                rows.append({**key, "g_variant": gv, "arch": a, "setting": "budget_ms_20", "quota": BUDGET_LEVEL,
                             "ndg_defined": not why, "undefined_reason": why,
                             **_stats(eV, eG, c["db"][("V", a)], c["db"][("primary", a)], c["db_rand"], why),
                             "oracle_reduction": red, "abs_reduction_V": eV * red, "abs_reduction_G": eG * red,
                             "boot_dropped": int(c["drop_b"]), "escalated_frac": c["frac"][a],
                             "sign_agree_train": ag_tr[0], "sign_agree_train_n": ag_tr[1],
                             "sign_agree_fit": ag_fit[0], "sign_agree_fit_n": ag_fit[1]})
    df = pd.DataFrame(rows)

    # ---- pooled test and reading
    pooled_cells = [c for c in cells if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    assert len(pooled_cells) == 12
    qi = QUOTAS.index(POOLED_QUOTA)
    summary["pooled"] = {}
    for gv in G_VARIANTS:
        res = {}
        for a in ARCH:
            pt = [(lambda r: r["diff"])(df[(df.g_variant == gv) & (df.arch == a) & (df.setting == "quota") & (df.quota == POOLED_QUOTA)
                                        & (df.track == c["track"]) & (df.geometry == c["geometry"]) & (df.system == c["system"])
                                        & (df.target == c["target"])].iloc[0]) for c in pooled_cells]
            dr = np.array([c["dq"][("V", a)][:, qi] - c["dq"][(gv, a)][:, qi] for c in pooled_cells])
            left_out = np.isnan(dr).any(axis=0)
            with np.errstate(all="ignore"):
                pool = np.nanmean(dr, axis=0)
            lo, hi = ci(pool)
            sub = df[(df.g_variant == gv) & (df.arch == a) & (df.setting == "quota") & (df.quota == POOLED_QUOTA)]
            sub = sub[(sub.track != "nuPlan") | (sub.system == "pdm_closed")]
            res[a] = {"mean_diff_V_minus_G": float(np.mean(pt)), "ci_lo": lo, "ci_hi": hi,
                      "draws_with_a_cell_left_out": int(left_out.sum()), "cells": len(pooled_cells),
                      "cells_V_gt_G": int(np.sum(np.array(pt) > 0)), "cells_diff_ci_above_0": int((sub.diff_lo > 0).sum()),
                      "cells_diff_ci_below_0": int((sub.diff_hi < 0).sum())}
        pos = {a: (res[a]["ci_lo"] > 0) for a in ARCH}
        inc0 = all(res[a]["ci_lo"] <= 0 <= res[a]["ci_hi"] for a in ARCH)
        reading = ("objective matters" if any(pos[r] for r in R1) and any(pos[g] for g in GATES)
                   else "architecture-driven" if inc0 else "mixed")
        summary["pooled"][gv] = {"per_architecture": res, "reading": reading, "deciding": gv == "primary"}
    summary["reading_primary"] = summary["pooled"]["primary"]["reading"]
    summary["G_beats_V_significantly"] = df[df.diff_hi < 0][["track", "geometry", "system", "target", "g_variant", "arch",
                                                            "setting", "quota", "eta_V", "eta_G", "diff", "diff_lo", "diff_hi"]
                                                           ].to_dict("records")
    summary["sign_agreement"] = (df[(df.setting == "quota") & (df.quota == 0.1) & (df.arch == ARCH[0])]
                                 [["track", "geometry", "system", "target", "g_variant", "sign_agree_train", "sign_agree_train_n",
                                   "sign_agree_fit", "sign_agree_fit_n"]].to_dict("records"))
    summary["drops"] = {f"{c['track']}|{c['geometry']}|{c['system']}|{c['target']}": {"quota": c["drop_q"].tolist(), "budget": int(c["drop_b"])}
                        for c in cells}
    for name, obj in (("benchmark_target_swap.csv", df), ("benchmark_target_swap_summary.json", summary)):
        for out in (FINAL / name, run / name):
            if name.endswith(".csv"):
                obj.to_csv(out, index=False)
            else:
                out.write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {FINAL / 'benchmark_target_swap.csv'} ({len(df)} rows) and the summary", flush=True)
    t = df[(df.g_variant == "primary") & (df.setting == "quota") & (df.quota == 0.2)]
    for _, r in t.iterrows():
        print(f"  {r.track:8s} {r.geometry:6s} {r.system:10s} {r.target:8s} {r.arch:10s} V {r.eta_V:+.3f}  G {r.eta_G:+.3f}  "
              f"diff {r['diff']:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]")
    for gv in G_VARIANTS:
        print(f"  pooled ({gv}): reading = {summary['pooled'][gv]['reading']}")
        for a, x in summary["pooled"][gv]["per_architecture"].items():
            print(f"    {a:10s} D {x['mean_diff_V_minus_G']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}]  V>G in {x['cells_V_gt_G']}/12")


def _stats(eV, eG, xV, xG, xr, why):
    if why:
        return {k: np.nan for k in ("eta_V", "eta_V_lo", "eta_V_hi", "V_minus_random", "V_minus_random_lo", "V_minus_random_hi",
                                    "eta_G", "eta_G_lo", "eta_G_hi", "G_minus_random", "G_minus_random_lo", "G_minus_random_hi",
                                    "diff", "diff_lo", "diff_hi", "p_diff_le_0")}
    out = {}
    for tag, e, x in (("V", eV, xV), ("G", eG, xG)):
        lo, hi = ci(x)
        dr = x - xr
        dlo, dhi = ci(dr)
        with np.errstate(all="ignore"):
            out.update({f"eta_{tag}": float(e), f"eta_{tag}_lo": lo, f"eta_{tag}_hi": hi,
                        f"{tag}_minus_random": float(np.nanmean(dr)), f"{tag}_minus_random_lo": dlo, f"{tag}_minus_random_hi": dhi})
    dd = xV - xG
    lo, hi = ci(dd)
    fin = dd[np.isfinite(dd)]
    out.update({"diff": float(eV - eG), "diff_lo": lo, "diff_hi": hi,
                "p_diff_le_0": float(np.mean(fin <= 0)) if len(fin) else np.nan})
    return out


if __name__ == "__main__":
    main()
